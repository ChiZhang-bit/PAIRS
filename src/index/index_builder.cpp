#include "rcfs/index_builder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace rcfs {
namespace {

template <typename Function>
void build_views(uint32_t count, uint32_t threads, Function build_one) {
  const uint32_t workers = std::min(count, threads);
  if (workers <= 1) {
    for (uint32_t i = 0; i < count; ++i) build_one(i);
    return;
  }
  std::atomic<uint32_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto work = [&]() {
    try {
      for (;;) {
        const auto i = next.fetch_add(1);
        if (i >= count) break;
        build_one(i);
      }
    } catch (...) {
      next.store(count);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) error = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  try {
    for (uint32_t i = 1; i < workers; ++i) pool.emplace_back(work);
    work();
  } catch (...) {
    next.store(count);
    for (auto& thread : pool) thread.join();
    throw;
  }
  for (auto& thread : pool) thread.join();
  if (error) std::rethrow_exception(error);
}

void normalize_vector(float* values, uint32_t dim) {
  double squared_norm = 0.0;
  for (uint32_t i = 0; i < dim; ++i) {
    squared_norm += static_cast<double>(values[i]) * values[i];
  }
  if (squared_norm <= 0.0) return;
  const float inverse = static_cast<float>(1.0 / std::sqrt(squared_norm));
  for (uint32_t i = 0; i < dim; ++i) values[i] *= inverse;
}

void validate_dataset(const Dataset& dataset, const BuildParams& params) {
  const uint32_t rows = dataset.vectors.rows;
  const uint32_t dim = dataset.vectors.cols;
  if (rows == 0 || dim == 0 ||
      dataset.vectors.values.size() != static_cast<size_t>(rows) * dim) {
    throw std::invalid_argument("The vector matrix must be non-empty and dense");
  }
  if (dataset.numeric_columns.empty() && dataset.categorical_columns.empty()) {
    throw std::invalid_argument(
        "RCFS needs at least one filter attribute");
  }
  if (dataset.numeric_columns.size() > 4096 || dataset.categorical_columns.size() > 4096 ||
      dataset.numeric_columns.size() >
          static_cast<size_t>(std::numeric_limits<NumericAttrId>::max()) ||
      dataset.categorical_columns.size() >
          static_cast<size_t>(std::numeric_limits<CategoricalAttrId>::max())) {
    throw std::invalid_argument("Too many attributes for the RCFS ID types");
  }
  if (params.canonical_full_range_attribute >= dataset.numeric_columns.size() + dataset.categorical_columns.size()) {
    throw std::invalid_argument("Canonical full-range attribute is out of range");
  }
  if (params.M < 2 || params.ef_construction < params.M ||
      params.ef_max < params.M) {
    throw std::invalid_argument(
        "Require M >= 2 and ef_construction/ef_max >= M");
  }

  for (const auto& column : dataset.numeric_columns) {
    if (column.name.empty() || column.values.size() != rows) {
      throw std::invalid_argument("Invalid numeric column shape/name");
    }
    for (float value : column.values) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Numeric attributes must be finite");
      }
    }
  }

  for (const auto& column : dataset.categorical_columns) {
    if (column.name.empty() || column.value_names.empty() ||
        column.values.size() != rows) {
      throw std::invalid_argument("Invalid categorical column shape/schema");
    }
    for (ValueId value : column.values) {
      if (value != kMissingValue && value >= column.value_names.size()) {
        throw std::invalid_argument(
            "Categorical value ID is outside its declared dictionary");
      }
    }
  }
}

}  // namespace

RoutingStatistics collect_routing_statistics(const UnifiedRoutingView& view) {
  RoutingStatistics s;
  s.min_key=view.order.min_key(); s.max_key=view.order.max_key();
  s.forward_blocks=view.routing.block_count(); s.stored_neighbors=view.routing.stored_neighbor_count();
  s.entry_count=static_cast<uint32_t>(view.routing.full_domain_entries().size());
  if (view.order.type()==AttributeType::kNumeric) {
    const auto& keys=view.order.sorted_values();
    for (size_t i=0;i<keys.size();++i) if (!i || keys[i]!=keys[i-1]) ++s.distinct_keys;
    for (uint32_t i=0;i<=32;++i) s.numeric_quantiles.push_back(keys[(uint64_t(keys.size()-1)*i)/32]);
  } else {
    const auto& keys=view.order.sorted_codes();
    for (size_t i=0;i<keys.size();++i) if (!i || keys[i]!=keys[i-1]) ++s.distinct_keys;
  }
  return s;
}

RcfsIndex RcfsIndexBuilder::build(Dataset dataset,
                                  const BuildParams& params,
                                  uint32_t build_threads) {
  if (!build_threads) throw std::invalid_argument("build_threads must be positive");
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  validate_dataset(dataset, params);

  RcfsIndex index;
  index.build_params_ = params;
  index.schema_ = make_schema(dataset);
  if (index.schema_.total_category_codes > (1U << 24)) throw std::invalid_argument("Category dictionary exceeds v4 format limit");
  for (size_t c = 0; c < dataset.categorical_columns.size(); ++c) {
    dataset.categorical_columns[c].code_offset =
        index.schema_.category_code_offsets[c];
  }

  if (params.normalize) {
    for (uint32_t row = 0; row < dataset.vectors.rows; ++row) {
      normalize_vector(dataset.vectors.values.data() +
                           static_cast<size_t>(row) * dataset.vectors.cols,
                       dataset.vectors.cols);
    }
  }

  index.dataset_ = std::move(dataset);
  const auto count = index.schema_.attribute_count();
  index.views_.resize(count);
  std::vector<AttributeId> owners(count);
  std::unordered_multimap<uint64_t,AttributeId> order_hashes;
  // Choose graph owners before launching workers so ownership does not depend
  // on which field happens to finish first.
  for (AttributeId attribute = 0; attribute < count; ++attribute) {
    const size_t j = attribute;
    std::cout << "[rcfs-build] attribute=" << index.schema_.attribute_name(attribute)
              << " (" << (j+1) << '/' << index.schema_.attribute_count() << ')' << std::endl;
    auto& view = index.views_[attribute];
    if (index.schema_.attribute_type(attribute) == AttributeType::kNumeric)
      view.order = AttributeOrder::build(attribute, index.dataset_.numeric_columns[attribute].values);
    else {
      const auto& c = index.dataset_.categorical_columns[attribute-index.schema_.numeric_names.size()];
      view.order = AttributeOrder::build_categorical(attribute, c.values, c.value_names);
    }

    uint64_t hash=14695981039346656037ULL;
    for (NodeId id:view.order.pos_to_id()) { hash^=id; hash*=1099511628211ULL; }
    AttributeId owner=kInvalidAttribute;
    const auto possible=order_hashes.equal_range(hash);
    // Hashes only narrow the comparison. Sharing requires an identical full
    // permutation, with the same vectors, construction parameters and seed.
    for (auto it=possible.first;it!=possible.second;++it)
      if (index.views_[it->second].order.pos_to_id()==view.order.pos_to_id()) { owner=it->second; break; }
    owners[attribute] = owner == kInvalidAttribute ? attribute : owner;
    if (owner == kInvalidAttribute) order_hashes.emplace(hash,attribute);
  }
  // Workers write separate view slots. Each physical graph still uses its
  // prescribed insertion order, parameters and seed.
  const auto ordered = Clock::now();
  std::vector<double> view_seconds(count);
  std::atomic<uint32_t> active{0}, peak{0};
  build_views(count, build_threads, [&](AttributeId attribute) {
    const auto view_start = Clock::now();
    const auto concurrent = active.fetch_add(1) + 1;
    auto previous_peak = peak.load();
    while (previous_peak < concurrent && !peak.compare_exchange_weak(previous_peak, concurrent)) {}
    auto& view = index.views_[attribute];
    if (owners[attribute] == attribute)
      view.routing = RangeRoutingIndex::build(index.dataset_.vectors,view.order,params);
    view.postings = PostingIndex::build(
        view.order, index.schema_, index.dataset_.categorical_columns);
    view_seconds[attribute] = std::chrono::duration<double>(Clock::now() - view_start).count();
    active.fetch_sub(1);
  });
  const auto constructed = Clock::now();
  // Bind shared graphs only after all owners have finished construction.
  for (AttributeId attribute = 0; attribute < count; ++attribute) {
    auto& view = index.views_[attribute];
    if (owners[attribute] != attribute)
      view.routing = index.views_[owners[attribute]].routing.shared_for(attribute);
    view.statistics = collect_routing_statistics(view);
  }

  index.validate();
  const auto finished = Clock::now();
  std::cout << "[rcfs-build] setup_seconds=" << std::chrono::duration<double>(ordered-started).count()
            << " view_phase_seconds=" << std::chrono::duration<double>(constructed-ordered).count()
            << " longest_view_seconds=" << *std::max_element(view_seconds.begin(),view_seconds.end())
            << " finalize_seconds=" << std::chrono::duration<double>(finished-constructed).count()
            << " physical_graphs=" << order_hashes.size() << " peak_view_tasks=" << peak.load() << '\n';
  return index;
}

}  // namespace rcfs
