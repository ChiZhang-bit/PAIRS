#include "rcfs/index.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include "rcfs/binary_io.h"

namespace rcfs {
namespace {

constexpr std::array<char, 8> kMetaMagic = {'R', 'C', 'F', 'S',
                                             'I', 'D', 'X', '4'};
constexpr std::array<char, 8> kVectorMagic = {'R', 'C', 'F', 'S',
                                               'V', 'E', 'C', '4'};
constexpr std::array<char, 8> kAttributeMagic = {'R', 'C', 'F', 'S',
                                                  'A', 'T', 'R', '4'};
constexpr uint32_t kIndexVersion = 4;
constexpr uint32_t kMaxAttributes = 4096;
constexpr uint32_t kMaxCategoryCodes = 1U << 24;

std::string component(const std::string& prefix, const std::string& suffix) {
  return prefix + suffix;
}

std::string order_path(const std::string& prefix, uint32_t attribute) {
  return prefix + ".attr_" + std::to_string(attribute) + ".order.bin";
}

std::string graph_path(const std::string& prefix, uint32_t attribute) {
  return prefix + ".attr_" + std::to_string(attribute) + ".graph.bin";
}

std::string posting_path(const std::string& prefix, uint32_t attribute) {
  return prefix + ".attr_" + std::to_string(attribute) + ".postings.bin";
}

struct ComponentDescriptor {
  std::string suffix;
  uint64_t bytes = 0;
  uint64_t checksum = 0;
};

std::vector<std::string> expected_component_suffixes(uint32_t attributes) {
  std::vector<std::string> suffixes = {".vectors.bin", ".attributes.bin"};
  suffixes.reserve(2ULL + 3ULL * attributes);
  for (uint32_t j = 0; j < attributes; ++j) {
    const std::string stem = ".attr_" + std::to_string(j);
    suffixes.push_back(stem + ".order.bin");
    suffixes.push_back(stem + ".graph.bin");
    suffixes.push_back(stem + ".postings.bin");
  }
  return suffixes;
}

uint64_t file_checksum(const std::string& path) {
  // FNV-1a checksum for detecting accidental file corruption.
  constexpr uint64_t kOffset = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot checksum component: " + path);
  std::array<char, 1U << 16> buffer{};
  uint64_t checksum = kOffset;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      checksum ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
      checksum *= kPrime;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Failed while checksumming component: " + path);
  }
  return checksum;
}

std::vector<ComponentDescriptor> describe_components(
    const std::string& prefix, uint32_t attributes) {
  std::vector<ComponentDescriptor> result;
  for (const std::string& suffix : expected_component_suffixes(attributes)) {
    const std::string path = prefix + suffix;
    result.push_back({suffix, file_size_bytes(path), file_checksum(path)});
  }
  return result;
}

void verify_components(const std::string& prefix,
                       const std::vector<ComponentDescriptor>& descriptors) {
  for (const ComponentDescriptor& descriptor : descriptors) {
    const std::string path = prefix + descriptor.suffix;
    if (file_size_bytes(path) != descriptor.bytes) {
      throw std::runtime_error("RCFS component length mismatch: " + path);
    }
    if (file_checksum(path) != descriptor.checksum) {
      throw std::runtime_error("RCFS component checksum mismatch: " + path);
    }
  }
}

template <size_t N>
void write_magic(std::ostream& output, const std::array<char, N>& magic,
                 const std::string& context) {
  write_exact(output, magic.data(), magic.size(), context);
}

template <size_t N>
void expect_magic(std::istream& input, const std::array<char, N>& expected,
                  const std::string& context) {
  std::array<char, N> actual{};
  read_exact(input, actual.data(), actual.size(), context);
  if (actual != expected) {
    throw std::runtime_error("Unsupported/corrupt RCFS component; rebuild v4 index: " + context);
  }
}

void ensure_parent(const std::string& prefix) {
  const std::filesystem::path parent =
      std::filesystem::path(prefix).parent_path();
  if (parent.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) {
    throw std::runtime_error("Cannot create index directory: " +
                             error.message());
  }
}

uint64_t checked_matrix_count(uint32_t rows, uint32_t cols) {
  if (cols != 0 && rows > std::numeric_limits<uint64_t>::max() / cols) {
    throw std::runtime_error("Vector matrix element count overflows uint64");
  }
  return static_cast<uint64_t>(rows) * cols;
}

void write_meta(const RcfsIndex& index,
                const std::vector<ComponentDescriptor>& descriptors,
                const std::string& path) {
  std::ofstream output = open_binary_output(path);
  write_magic(output, kMetaMagic, path);
  write_pod(output, kIndexVersion, path);
  write_pod(output, index.size(), path);
  write_pod(output, index.dim(), path);
  write_pod(output,
            static_cast<uint32_t>(index.schema().numeric_names.size()), path);
  write_pod(output,
            static_cast<uint32_t>(index.schema().categorical_names.size()),
            path);
  write_pod(output, index.schema().total_category_codes, path);

  const BuildParams& params = index.build_params();
  write_pod(output, params.M, path);
  write_pod(output, params.ef_construction, path);
  write_pod(output, params.ef_max, path);
  write_pod(output, params.random_seed, path);
  write_pod(output, params.canonical_full_range_attribute, path);
  write_pod(output, params.normalize, path);

  write_string_vector(output, index.schema().numeric_names, path);
  write_string_vector(output, index.schema().categorical_names, path);
  write_pod(output,
            static_cast<uint64_t>(
                index.schema().categorical_value_names.size()),
            path);
  for (const auto& names : index.schema().categorical_value_names) {
    write_string_vector(output, names, path);
  }
  write_vector(output, index.schema().category_code_offsets, path);
  write_pod(output,static_cast<uint32_t>(index.views().size()),path);
  for (const auto& view:index.views()) {
    const auto& s=view.statistics;
    write_pod(output,s.min_key,path); write_pod(output,s.max_key,path);
    write_pod(output,s.distinct_keys,path); write_pod(output,s.forward_blocks,path);
    write_pod(output,s.stored_neighbors,path); write_pod(output,s.entry_count,path);
    write_vector(output,s.numeric_quantiles,path);
  }
  write_pod(output, static_cast<uint32_t>(descriptors.size()), path);
  for (const ComponentDescriptor& descriptor : descriptors) {
    write_string(output, descriptor.suffix, path);
    write_pod(output, descriptor.bytes, path);
    write_pod(output, descriptor.checksum, path);
  }
}

void write_vectors(const RcfsIndex& index, const std::string& path) {
  std::ofstream output = open_binary_output(path);
  write_magic(output, kVectorMagic, path);
  write_pod(output, kIndexVersion, path);
  write_pod(output, index.size(), path);
  write_pod(output, index.dim(), path);
  write_array(output, index.dataset().vectors.values, path);
}

void write_attributes(const RcfsIndex& index, const std::string& path) {
  std::ofstream output = open_binary_output(path);
  write_magic(output, kAttributeMagic, path);
  write_pod(output, kIndexVersion, path);
  write_pod(output, index.size(), path);
  write_pod(output,
            static_cast<uint32_t>(index.dataset().numeric_columns.size()),
            path);
  write_pod(output,
            static_cast<uint32_t>(index.dataset().categorical_columns.size()),
            path);
  for (const auto& column : index.dataset().numeric_columns) {
    write_array(output, column.values, path);
  }
  for (const auto& column : index.dataset().categorical_columns) {
    write_array(output, column.values, path);
  }
}

}  // namespace

const AttributeView& RcfsIndex::view(AttributeId attribute) const {
  if (attribute >= views_.size()) {
    throw std::out_of_range("Unified attribute view is out of range");
  }
  return views_[attribute];
}

const float* RcfsIndex::vector(NodeId id) const {
  if (id >= size()) throw std::out_of_range("Vector node ID is out of range");
  return dataset_.vectors.values.data() + static_cast<size_t>(id) * dim();
}

void RcfsIndex::save(const std::string& prefix) const {
  validate();
  ensure_parent(prefix);
  write_vectors(*this, component(prefix, ".vectors.bin"));
  write_attributes(*this, component(prefix, ".attributes.bin"));

  for (uint32_t j = 0; j < views_.size(); ++j) {
    {
      std::ofstream output = open_binary_output(order_path(prefix, j));
      views_[j].order.save(output);
    }
    {
      std::ofstream output = open_binary_output(graph_path(prefix, j));
      uint32_t owner=j;
      for (uint32_t previous=0;previous<j;++previous)
        if (views_[j].routing.shares_storage(views_[previous].routing)) { owner=previous; break; }
      write_pod(output,owner);
      if (owner==j) views_[j].routing.save(output);
    }
    {
      std::ofstream output = open_binary_output(posting_path(prefix, j));
      views_[j].postings.save(output);
    }
  }
  const std::vector<ComponentDescriptor> descriptors = describe_components(
      prefix, static_cast<uint32_t>(views_.size()));
  // Record sizes and checksums only after every component has been written.
  write_meta(*this, descriptors, component(prefix, ".meta"));
}

RcfsIndex RcfsIndex::load(const std::string& prefix) {
  RcfsIndex index;
  std::vector<ComponentDescriptor> descriptors;
  const std::string meta_path = component(prefix, ".meta");
  {
    std::ifstream input = open_binary_input(meta_path);
    expect_magic(input, kMetaMagic, meta_path);
    if (read_pod<uint32_t>(input, meta_path) != kIndexVersion) {
      throw std::runtime_error("Unsupported RCFS index version; rebuild the index");
    }
    const uint32_t rows = read_pod<uint32_t>(input, meta_path);
    const uint32_t dim = read_pod<uint32_t>(input, meta_path);
    const uint32_t numeric_count = read_pod<uint32_t>(input, meta_path);
    const uint32_t categorical_count = read_pod<uint32_t>(input, meta_path);
    const uint32_t total_codes = read_pod<uint32_t>(input, meta_path);
    if (rows == 0 || dim == 0 || (numeric_count == 0 && categorical_count == 0) ||
        numeric_count > kMaxAttributes || categorical_count > kMaxAttributes ||
        total_codes > kMaxCategoryCodes) {
      throw std::runtime_error("Invalid RCFS index dimensions in meta file");
    }

    index.dataset_.vectors.rows = rows;
    index.dataset_.vectors.cols = dim;
    index.build_params_.M = read_pod<uint32_t>(input, meta_path);
    index.build_params_.ef_construction =
        read_pod<uint32_t>(input, meta_path);
    index.build_params_.ef_max = read_pod<uint32_t>(input, meta_path);
    index.build_params_.random_seed = read_pod<uint32_t>(input, meta_path);
    index.build_params_.canonical_full_range_attribute =
        read_pod<AttributeId>(input, meta_path);
    index.build_params_.normalize = read_pod<bool>(input, meta_path);

    index.schema_.numeric_names =
        read_string_vector(input, numeric_count, kDefaultMaxStringBytes,
                           meta_path);
    index.schema_.categorical_names =
        read_string_vector(input, categorical_count, kDefaultMaxStringBytes,
                           meta_path);
    const uint64_t dictionary_count = read_pod<uint64_t>(input, meta_path);
    if (dictionary_count != categorical_count) {
      throw std::runtime_error("Category dictionary count is inconsistent");
    }
    index.schema_.categorical_value_names.reserve(categorical_count);
    for (uint32_t i = 0; i < categorical_count; ++i) {
      index.schema_.categorical_value_names.push_back(read_string_vector(
          input, total_codes, kDefaultMaxStringBytes, meta_path));
    }
    index.schema_.category_code_offsets = read_vector<CategoryCode>(
        input, categorical_count, meta_path);
    index.schema_.total_category_codes = total_codes;
    if (index.schema_.numeric_names.size()!=numeric_count || index.schema_.categorical_names.size()!=categorical_count)
      throw std::runtime_error("Schema/header field counts differ");
    const uint32_t stats_count=read_pod<uint32_t>(input,meta_path);
    if (stats_count!=numeric_count+categorical_count) throw std::runtime_error("Statistics view count differs");
    index.views_.resize(stats_count);
    for (auto& view:index.views_) {
      auto& s=view.statistics;
      s.min_key=read_pod<double>(input,meta_path); s.max_key=read_pod<double>(input,meta_path);
      s.distinct_keys=read_pod<uint64_t>(input,meta_path); s.forward_blocks=read_pod<uint64_t>(input,meta_path);
      s.stored_neighbors=read_pod<uint64_t>(input,meta_path); s.entry_count=read_pod<uint32_t>(input,meta_path);
      s.numeric_quantiles=read_vector<double>(input,33,meta_path);
    }
    const std::vector<std::string> expected =
        expected_component_suffixes(numeric_count + categorical_count);
    const uint32_t descriptor_count = read_pod<uint32_t>(input, meta_path);
    if (descriptor_count != expected.size()) {
      throw std::runtime_error("RCFS component manifest count is inconsistent");
    }
    descriptors.reserve(descriptor_count);
    for (uint32_t i = 0; i < descriptor_count; ++i) {
      ComponentDescriptor descriptor;
      descriptor.suffix = read_string(input, 256, meta_path);
      descriptor.bytes = read_pod<uint64_t>(input, meta_path);
      descriptor.checksum = read_pod<uint64_t>(input, meta_path);
      if (descriptor.suffix != expected[i] || descriptor.bytes == 0) {
        throw std::runtime_error(
            "RCFS component manifest entry is inconsistent");
      }
      descriptors.push_back(std::move(descriptor));
    }
    require_eof(input, meta_path);
    index.schema_.validate();
  }

  // Check file integrity before parsing. validate() below also checks that the
  // decoded mappings, keys and postings agree with the shared attributes.
  verify_components(prefix, descriptors);

  const std::string vectors_path = component(prefix, ".vectors.bin");
  {
    std::ifstream input = open_binary_input(vectors_path);
    expect_magic(input, kVectorMagic, vectors_path);
    if (read_pod<uint32_t>(input, vectors_path) != kIndexVersion ||
        read_pod<uint32_t>(input, vectors_path) != index.size() ||
        read_pod<uint32_t>(input, vectors_path) != index.dim()) {
      throw std::runtime_error("Vector component header is inconsistent");
    }
    const uint64_t count = checked_matrix_count(index.size(), index.dim());
    index.dataset_.vectors.values =
        read_array<float>(input, count, count, vectors_path);
    require_eof(input, vectors_path);
  }

  index.dataset_.numeric_columns.resize(index.schema_.numeric_names.size());
  for (size_t j = 0; j < index.dataset_.numeric_columns.size(); ++j) {
    index.dataset_.numeric_columns[j].name = index.schema_.numeric_names[j];
  }
  index.dataset_.categorical_columns.resize(
      index.schema_.categorical_names.size());
  for (size_t c = 0; c < index.dataset_.categorical_columns.size(); ++c) {
    auto& column = index.dataset_.categorical_columns[c];
    column.name = index.schema_.categorical_names[c];
    column.value_names = index.schema_.categorical_value_names[c];
    column.code_offset = index.schema_.category_code_offsets[c];
  }

  const std::string attributes_path = component(prefix, ".attributes.bin");
  {
    std::ifstream input = open_binary_input(attributes_path);
    expect_magic(input, kAttributeMagic, attributes_path);
    if (read_pod<uint32_t>(input, attributes_path) != kIndexVersion ||
        read_pod<uint32_t>(input, attributes_path) != index.size() ||
        read_pod<uint32_t>(input, attributes_path) !=
            index.dataset_.numeric_columns.size() ||
        read_pod<uint32_t>(input, attributes_path) !=
            index.dataset_.categorical_columns.size()) {
      throw std::runtime_error("Attribute component header is inconsistent");
    }
    for (auto& column : index.dataset_.numeric_columns) {
      column.values = read_array<float>(input, index.size(), index.size(),
                                        attributes_path);
    }
    for (auto& column : index.dataset_.categorical_columns) {
      column.values = read_array<ValueId>(input, index.size(), index.size(),
                                          attributes_path);
    }
    require_eof(input, attributes_path);
  }

  index.views_.resize(index.schema_.attribute_count());
  const uint64_t max_posting_positions =
      static_cast<uint64_t>(index.size()) *
      std::max<size_t>(1, index.dataset_.categorical_columns.size());
  for (uint32_t j = 0; j < index.views_.size(); ++j) {
    const std::string current_order_path = order_path(prefix, j);
    const std::string current_graph_path = graph_path(prefix, j);
    const std::string current_posting_path = posting_path(prefix, j);
    {
      std::ifstream input = open_binary_input(current_order_path);
      index.views_[j].order = AttributeOrder::load(input, index.size());
      require_eof(input, current_order_path);
    }
    {
      std::ifstream input = open_binary_input(current_graph_path);
      const uint64_t component_bytes = file_size_bytes(current_graph_path);
      const uint32_t owner=read_pod<uint32_t>(input);
      // Shared owners precede their references, so loading can reuse an existing graph
      // without forward references or cyclic ownership.
      if (owner>j) throw std::runtime_error("Invalid shared graph owner");
      if (owner==j) index.views_[j].routing = RangeRoutingIndex::load(
          input, index.size(), component_bytes, component_bytes);
      else {
        if (index.views_[j].order.pos_to_id()!=index.views_[owner].order.pos_to_id())
          throw std::runtime_error("Shared graph order mismatch");
        index.views_[j].routing=index.views_[owner].routing.shared_for(j);
      }
      require_eof(input, current_graph_path);
    }
    {
      std::ifstream input = open_binary_input(current_posting_path);
      index.views_[j].postings = PostingIndex::load(
          input, index.size(), index.schema_.total_category_codes,
          max_posting_positions);
      require_eof(input, current_posting_path);
    }
  }

  index.validate();
  return index;
}

void RcfsIndex::validate() const {
  schema_.validate();
  const uint32_t rows = size();
  const uint32_t dimension = dim();
  if (rows == 0 || dimension == 0 ||
      dataset_.vectors.values.size() !=
          static_cast<size_t>(rows) * dimension) {
    throw std::runtime_error("Invalid shared vector store");
  }
  if (dataset_.numeric_columns.size() != schema_.numeric_names.size() ||
      dataset_.categorical_columns.size() !=
          schema_.categorical_names.size() ||
      views_.size() != schema_.attribute_count()) {
    throw std::runtime_error("Index attribute/view counts are inconsistent");
  }
  if (build_params_.canonical_full_range_attribute >= views_.size()) {
    throw std::runtime_error("Invalid canonical full-range view");
  }

  for (float value : dataset_.vectors.values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("Vector store contains a non-finite value");
    }
  }
  for (size_t j = 0; j < dataset_.numeric_columns.size(); ++j)
    if (dataset_.numeric_columns[j].values.size() != rows || dataset_.numeric_columns[j].name != schema_.numeric_names[j])
      throw std::runtime_error("Numeric column shape/schema mismatch");
  for (AttributeId attr = 0; attr < views_.size(); ++attr) {
    const auto& view = views_[attr];
    if (view.order.attribute() != attr || view.routing.attribute() != attr ||
        view.postings.order_attribute() != attr || view.order.size() != rows ||
        view.routing.size() != rows || view.order.type() != schema_.attribute_type(attr))
      throw std::runtime_error("Unified view metadata mismatch");
    view.order.validate(); view.routing.validate(); view.postings.validate(rows);
    const auto expected=collect_routing_statistics(view);
    const auto& stats=view.statistics;
    if (stats.min_key!=expected.min_key || stats.max_key!=expected.max_key ||
        stats.distinct_keys!=expected.distinct_keys || stats.forward_blocks!=expected.forward_blocks ||
        stats.stored_neighbors!=expected.stored_neighbors || stats.entry_count!=expected.entry_count ||
        stats.numeric_quantiles!=expected.numeric_quantiles)
      throw std::runtime_error("Routing statistics/component mismatch");
    // Valid arrays can still describe different datasets. Cross-check each
    // view-local key through its original ID against the shared column.
    for (Position pos = 0; pos < rows; ++pos) {
      NodeId id = view.order.node_at(pos);
      if (view.order.type() == AttributeType::kNumeric) {
        if (view.order.sorted_values()[pos] != dataset_.numeric_columns[attr].values[id])
          throw std::runtime_error("Numeric key/shared column mismatch");
      } else if (view.order.sorted_codes()[pos] != view.order.rank_of(dataset_.categorical_columns[attr-schema_.numeric_names.size()].values[id]))
        throw std::runtime_error("Category key/shared column mismatch");
    }
    if (view.order.type()==AttributeType::kCategorical) {
      const auto& column=dataset_.categorical_columns[attr-schema_.numeric_names.size()];
      const auto canonical=AttributeOrder::build_categorical(attr,column.values,column.value_names);
      if (view.order.code_to_rank()!=canonical.code_to_rank()) throw std::runtime_error("Noncanonical category ranks");
    }
    // Membership checks reject wrong entries. Combined with unique posting
    // rows, the total count also detects missing objects.
    for (size_t c = 0; c < dataset_.categorical_columns.size(); ++c) {
      const auto& col = dataset_.categorical_columns[c];
      uint64_t expected_count = 0, stored_count = 0;
      for (ValueId v : col.values) if (v != kMissingValue) ++expected_count;
      for (ValueId v = 0; v < col.value_names.size(); ++v) {
        auto part = view.postings.full_slice(schema_.category_code(static_cast<CategoricalAttrId>(c),v));
        stored_count += part.size();
        for (uint64_t i = 0; i < part.size(); ++i)
          if (col.values[view.order.node_at(view.postings.at(part,i))] != v)
            throw std::runtime_error("Posting/shared column mismatch");
      }
      if (stored_count != expected_count) throw std::runtime_error("Incomplete posting field");
    }
  }

  for (size_t c = 0; c < dataset_.categorical_columns.size(); ++c) {
    const auto& column = dataset_.categorical_columns[c];
    if (column.name != schema_.categorical_names[c] ||
        column.value_names != schema_.categorical_value_names[c] ||
        column.code_offset != schema_.category_code_offsets[c] ||
        column.values.size() != rows) {
      throw std::runtime_error("Categorical column schema is inconsistent");
    }
    for (ValueId value : column.values) {
      if (value != kMissingValue && value >= column.value_names.size()) {
        throw std::runtime_error("Categorical column contains an invalid value");
      }
    }
  }
}

}  // namespace rcfs
