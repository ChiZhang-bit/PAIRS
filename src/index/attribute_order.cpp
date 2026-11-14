#include "rcfs/attribute_order.h"
#include "rcfs/binary_io.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace rcfs {
namespace {
void check_size(size_t n) {
  if (!n || n > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Attribute order requires 1..INT_MAX rows");
}
}
AttributeOrder AttributeOrder::build(AttributeId attr, const std::vector<float>& values) {
  check_size(values.size());
  for (float v : values) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite numeric key");
  AttributeOrder out; out.attribute_ = attr;
  out.pos_to_id_.resize(values.size());
  std::iota(out.pos_to_id_.begin(), out.pos_to_id_.end(), NodeId{0});
  // Break equal-value ties by original ID without stable_sort's extra buffer.
  std::sort(out.pos_to_id_.begin(), out.pos_to_id_.end(), [&](NodeId a, NodeId b) {
    return values[a] < values[b] || (values[a] == values[b] && a < b);
  });
  auto& keys = std::get<0>(out.keys_);
  keys.resize(values.size()); out.id_to_pos_.resize(values.size());
  for (Position p = 0; p < out.size(); ++p) {
    keys[p] = values[out.pos_to_id_[p]]; out.id_to_pos_[out.pos_to_id_[p]] = p;
  }
  return out;
}
AttributeOrder AttributeOrder::build_categorical(AttributeId attr,
    const std::vector<ValueId>& values, uint32_t dictionary_size) {
  check_size(values.size());
  if (!dictionary_size || dictionary_size >= kMissingValue) throw std::invalid_argument("Invalid dictionary size");
  AttributeOrder out; out.attribute_ = attr; out.type_ = AttributeType::kCategorical;
  out.code_to_rank_.resize(dictionary_size);
  std::iota(out.code_to_rank_.begin(),out.code_to_rank_.end(),ValueId{0});
  out.keys_.emplace<1>(values.size());
  out.pos_to_id_.resize(values.size()); out.id_to_pos_.resize(values.size());
  out.category_boundaries_.assign(size_t(dictionary_size) + 2, 0);
  auto bucket = [&](ValueId v) {
    if (v != kMissingValue && v >= dictionary_size) throw std::invalid_argument("Category outside dictionary");
    return v == kMissingValue ? dictionary_size : v;
  };
  for (ValueId v : values) ++out.category_boundaries_[bucket(v) + 1];
  for (size_t i = 1; i < out.category_boundaries_.size(); ++i)
    out.category_boundaries_[i] += out.category_boundaries_[i-1];
  auto cursor = out.category_boundaries_;
  auto& keys = std::get<1>(out.keys_);
  // Distributing IDs in order keeps each category block stable without sorting.
  for (NodeId id = 0; id < values.size(); ++id) {
    Position p = cursor[bucket(values[id])]++;
    out.pos_to_id_[p] = id; out.id_to_pos_[id] = p; keys[p] = values[id];
  }
  return out;
}
AttributeOrder AttributeOrder::build_categorical(AttributeId attr,
    const std::vector<ValueId>& values,const std::vector<std::string>& dictionary) {
  std::vector<ValueId> ranks(dictionary.size()), codes(dictionary.size());
  std::iota(codes.begin(),codes.end(),ValueId{0});
  // Label byte order makes ranks independent of external code assignments.
  // Unsigned comparison also avoids platform differences in signed char.
  std::sort(codes.begin(),codes.end(),[&](ValueId a,ValueId b) {
    return std::lexicographical_compare(dictionary[a].begin(),dictionary[a].end(),
        dictionary[b].begin(),dictionary[b].end(),
        [](char x,char y){ return static_cast<unsigned char>(x)<static_cast<unsigned char>(y); });
  });
  for (ValueId r=0;r<codes.size();++r) {
    if (r && dictionary[codes[r]]==dictionary[codes[r-1]]) throw std::invalid_argument("Duplicate category label");
    ranks[codes[r]]=r;
  }
  std::vector<ValueId> keys; keys.reserve(values.size());
  for (ValueId code:values) keys.push_back(code==kMissingValue ? code : ranks.at(code));
  auto out=build_categorical(attr,keys,static_cast<uint32_t>(dictionary.size()));
  out.code_to_rank_=std::move(ranks); return out;
}
PositionRange AttributeOrder::map_equal(ValueId code) const {
  if (type_ != AttributeType::kCategorical) throw std::invalid_argument("Equality requires categorical order");
  if (code == kMissingValue || uint64_t(code) + 2 >= category_boundaries_.size()) return {};
  const ValueId rank=code_to_rank_[code];
  return {category_boundaries_[rank], category_boundaries_[size_t(rank) + 1]};
}
PositionRange AttributeOrder::map_closed(double lower, double upper) const {
  if (std::isnan(lower) || std::isnan(upper) || lower > upper || empty()) return {};
  if (type_ == AttributeType::kCategorical) {
    if (lower != upper || !std::isfinite(lower) || lower < 0 || lower >= kMissingValue || std::floor(lower) != lower)
      throw std::invalid_argument("Categorical predicate must be a singleton code interval");
    return map_equal(static_cast<ValueId>(lower));
  }
  const auto& keys = sorted_values();
  // Promote stored keys rather than rounding the query to float32. A double
  // endpoint can lie strictly between adjacent float32 values.
  auto b = std::lower_bound(keys.begin(), keys.end(), lower, [](float v, double x) { return double(v) < x; });
  auto e = std::upper_bound(b, keys.end(), upper, [](double x, float v) { return x < double(v); });
  return {static_cast<Position>(b - keys.begin()), static_cast<Position>(e - keys.begin())};
}
double AttributeOrder::min_key() const {
  if (empty()) return 0;
  return type_ == AttributeType::kNumeric ? double(sorted_values().front()) : double(sorted_codes().front());
}
double AttributeOrder::max_key() const {
  if (empty()) return 0;
  return type_ == AttributeType::kNumeric ? double(sorted_values().back()) : double(sorted_codes().back());
}
void AttributeOrder::save(std::ostream& out) const {
  validate();
  write_pod(out, uint64_t{0x3444524f53464352ULL});
  write_pod(out, attribute_); write_pod(out, type_); write_pod(out, size());
  if (type_ == AttributeType::kNumeric) write_array(out, sorted_values()); else write_array(out, sorted_codes());
  write_array(out, pos_to_id_); write_array(out, id_to_pos_); write_vector(out, category_boundaries_); write_vector(out,code_to_rank_);
}
AttributeOrder AttributeOrder::load(std::istream& in, uint64_t max_nodes) {
  if (read_pod<uint64_t>(in) != 0x3444524f53464352ULL) throw std::runtime_error("Unsupported order version; rebuild index");
  AttributeOrder out;
  out.attribute_ = read_pod<AttributeId>(in); out.type_ = read_pod<AttributeType>(in);
  uint32_t n = read_pod<uint32_t>(in);
  if (out.type_ == AttributeType::kNumeric) out.keys_ = read_array<float>(in, n, max_nodes);
  else if (out.type_ == AttributeType::kCategorical) out.keys_ = read_array<ValueId>(in, n, max_nodes);
  else throw std::runtime_error("Invalid attribute type");
  out.pos_to_id_ = read_array<NodeId>(in, n, max_nodes);
  out.id_to_pos_ = read_array<Position>(in, n, max_nodes);
  out.category_boundaries_ = read_vector<Position>(in, (uint64_t{1} << 24) + 2);
  out.code_to_rank_=read_vector<ValueId>(in,(uint64_t{1}<<24));
  out.validate(); return out;
}
void AttributeOrder::validate() const {
  check_size(size());
  if (id_to_pos_.size() != size() || std::visit([](const auto& k) { return k.size(); }, keys_) != size())
    throw std::runtime_error("Inconsistent order arrays");
  if (type_ == AttributeType::kNumeric && (!category_boundaries_.empty() || !code_to_rank_.empty() || keys_.index() != 0))
    throw std::runtime_error("Invalid numeric key storage");
  if (type_ == AttributeType::kCategorical) {
    if (keys_.index() != 1 || category_boundaries_.size() < 3 || category_boundaries_.front() != 0 ||
        category_boundaries_.back() != size() || !std::is_sorted(category_boundaries_.begin(), category_boundaries_.end()))
      throw std::runtime_error("Invalid category boundaries");
    if (code_to_rank_.size()+2!=category_boundaries_.size()) throw std::runtime_error("Invalid rank map size");
    std::vector<uint8_t> seen(code_to_rank_.size(),0);
    for (auto r:code_to_rank_) {
      if (r>=seen.size() || seen[r]++) throw std::runtime_error("Rank map is not a permutation");
    }
    for (size_t c = 0; c + 1 < category_boundaries_.size(); ++c)
      for (Position p = category_boundaries_[c]; p < category_boundaries_[c+1]; ++p)
        if (sorted_codes()[p] != (c + 2 == category_boundaries_.size() ? kMissingValue : c))
          throw std::runtime_error("Category block/code mismatch");
  }
  for (Position p = 0; p < size(); ++p) {
    NodeId id = pos_to_id_[p];
    // Matching inverse entries prove that no original ID occurs twice.
    if (id >= size() || id_to_pos_[id] != p) throw std::runtime_error("Invalid ID permutation");
    double key = type_ == AttributeType::kNumeric ? double(sorted_values()[p]) : double(sorted_codes()[p]);
    if (!std::isfinite(key)) throw std::runtime_error("Non-finite key");
    if (p) {
      double prev = type_ == AttributeType::kNumeric ? double(sorted_values()[p-1]) : double(sorted_codes()[p-1]);
      if (prev > key || (prev == key && pos_to_id_[p-1] >= id)) throw std::runtime_error("Unsorted order");
    }
  }
}
}  // namespace rcfs
