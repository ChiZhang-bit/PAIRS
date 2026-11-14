#include "rcfs/posting_index.h"
#include "rcfs/binary_io.h"
#include <algorithm>
#include <limits>
namespace rcfs {
PostingIndex PostingIndex::build(const AttributeOrder& order, const Schema& schema,
    const std::vector<CategoricalColumn>& cols) {
  PostingIndex out;
  out.order_attribute_ = order.attribute(); out.node_count_ = order.size();
  out.category_code_count_ = schema.total_category_codes;
  if (cols.size() != schema.categorical_names.size()) throw std::invalid_argument("Posting schema mismatch");
  for (const auto& c : cols) if (c.values.size() != order.size()) throw std::invalid_argument("Posting column size mismatch");
  if (order.type() == AttributeType::kCategorical) {
    auto c = order.attribute() - schema.numeric_names.size();
    if (c >= cols.size()) throw std::invalid_argument("Posting order attribute mismatch");
    out.implicit_begin_ = schema.category_code_offsets[c];
    const auto& bounds = order.category_boundaries();
    // Drop the missing-value block's end. Equality postings cover named values only.
    out.implicit_bounds_.assign(bounds.begin(), bounds.end()-1);
    out.implicit_ranks_=order.code_to_rank();
  }
  out.offsets_.assign(size_t(out.category_code_count_) + 1, 0);
  for (size_t c = 0; c < cols.size(); ++c) {
    CategoryCode base = schema.category_code_offsets[c];
    for (ValueId v : cols[c].values) {
      if (v == kMissingValue) continue;
      if (v >= cols[c].value_names.size()) throw std::invalid_argument("Invalid posting category");
      if (!out.implicit(base+v)) ++out.offsets_[size_t(base+v)+1];
    }
  }
  for (size_t i = 1; i < out.offsets_.size(); ++i) out.offsets_[i] += out.offsets_[i-1];
  if (out.offsets_.back() > std::numeric_limits<size_t>::max()/sizeof(Position)) throw std::overflow_error("Posting size");
  out.positions_.resize(size_t(out.offsets_.back()));
  auto cursor = out.offsets_;
  // Write view-local positions in increasing order so slices need only binary searches.
  for (Position pos = 0; pos < order.size(); ++pos) {
    NodeId id = order.node_at(pos);
    for (size_t c = 0; c < cols.size(); ++c) {
      ValueId v = cols[c].values[id];
      if (v == kMissingValue) continue;
      CategoryCode code = schema.category_code_offsets[c]+v;
      if (!out.implicit(code)) out.positions_[size_t(cursor[code]++)] = pos;
    }
  }
  return out;
}
uint64_t PostingIndex::posting_size(CategoryCode code) const { return full_slice(code).size(); }
PostingSlice PostingIndex::full_slice(CategoryCode code) const {
  if (!contains_code(code)) return {code,0,0};
  if (implicit(code)) {
    size_t c = implicit_ranks_[code-implicit_begin_];
    return {code,implicit_bounds_[c],implicit_bounds_[c+1]};
  }
  return {code,offsets_[code],offsets_[size_t(code)+1]};
}
PostingSlice PostingIndex::slice(CategoryCode code, const RangeBounds& bounds) const {
  if (bounds.begin > bounds.end || bounds.end > node_count_) throw std::out_of_range("Posting interval");
  PostingSlice full = full_slice(code);
  if (full.empty() || bounds.empty()) return {code,0,0};
  if (implicit(code)) {
    // This posting is a contiguous category block, so intersection needs no array.
    uint64_t b = std::max<uint64_t>(full.begin_index,bounds.begin);
    uint64_t e = std::min<uint64_t>(full.end_index,bounds.end);
    return b < e ? PostingSlice{code,b,e} : PostingSlice{code,0,0};
  }
  auto b = positions_.begin()+static_cast<ptrdiff_t>(full.begin_index);
  auto e = positions_.begin()+static_cast<ptrdiff_t>(full.end_index);
  auto l = std::lower_bound(b,e,bounds.begin), u = std::lower_bound(l,e,bounds.end);
  return {code,uint64_t(l-positions_.begin()),uint64_t(u-positions_.begin())};
}
Position PostingIndex::at(const PostingSlice& part, uint64_t relative) const {
  auto full = full_slice(part.code);
  if (!contains_code(part.code) || part.begin_index < full.begin_index || part.end_index > full.end_index ||
      part.begin_index > part.end_index || relative >= part.size()) throw std::out_of_range("Posting slice offset");
  return implicit(part.code) ? Position(part.begin_index+relative) : positions_[size_t(part.begin_index+relative)];
}
void PostingIndex::save(std::ostream& out) const {
  validate(node_count_);
  write_pod(out,uint64_t{0x3454535053464352ULL});
  write_pod(out,order_attribute_); write_pod(out,node_count_); write_pod(out,category_code_count_);
  write_pod(out,implicit_begin_); write_vector(out,implicit_bounds_); write_vector(out,implicit_ranks_);
  write_vector(out,offsets_); write_vector(out,positions_);
}
PostingIndex PostingIndex::load(std::istream& in, uint64_t max_nodes, uint32_t max_codes, uint64_t max_positions) {
  if (read_pod<uint64_t>(in) != 0x3454535053464352ULL) throw std::runtime_error("Unsupported posting version; rebuild index");
  PostingIndex out;
  out.order_attribute_ = read_pod<AttributeId>(in); out.node_count_ = read_pod<uint32_t>(in);
  out.category_code_count_ = read_pod<uint32_t>(in);
  if (!out.node_count_ || out.node_count_ > max_nodes || out.category_code_count_ > max_codes)
    throw std::runtime_error("Posting load limits");
  out.implicit_begin_ = read_pod<CategoryCode>(in);
  out.implicit_bounds_ = read_vector<Position>(in,uint64_t(max_codes)+1);
  out.implicit_ranks_=read_vector<ValueId>(in,max_codes);
  out.offsets_ = read_vector<uint64_t>(in,uint64_t(max_codes)+1);
  out.positions_ = read_vector<Position>(in,max_positions);
  out.validate(out.node_count_); return out;
}
void PostingIndex::validate(uint64_t n) const {
  if (!n || n != node_count_ || offsets_.size() != size_t(category_code_count_)+1 ||
      offsets_.front() != 0 || offsets_.back() != positions_.size() || !std::is_sorted(offsets_.begin(),offsets_.end()))
    throw std::runtime_error("Invalid posting offsets");
  if ((!implicit_bounds_.empty() && implicit_ranks_.size()+1!=implicit_bounds_.size()) ||
      (implicit_bounds_.empty() && !implicit_ranks_.empty())) throw std::runtime_error("Implicit rank map size");
  std::vector<uint8_t> seen(implicit_ranks_.size(),0);
  for (auto r:implicit_ranks_) if (r>=seen.size() || seen[r]++) throw std::runtime_error("Invalid implicit ranks");
  if (implicit_begin_ == kMissingValue) {
    if (!implicit_bounds_.empty()) throw std::runtime_error("Unexpected implicit posting");
  } else if (implicit_bounds_.size() < 2 ||
      uint64_t(implicit_begin_)+implicit_bounds_.size()-1 > category_code_count_ ||
      implicit_bounds_.front() != 0 || implicit_bounds_.back() > n ||
      !std::is_sorted(implicit_bounds_.begin(),implicit_bounds_.end()))
    throw std::runtime_error("Invalid implicit posting boundaries");
  for (CategoryCode code = 0; code < category_code_count_; ++code) {
    auto b = positions_.begin()+static_cast<ptrdiff_t>(offsets_[code]);
    auto e = positions_.begin()+static_cast<ptrdiff_t>(offsets_[size_t(code)+1]);
    if ((implicit(code) && b != e) || !std::is_sorted(b,e) || std::adjacent_find(b,e) != e ||
        (b != e && *(e-1) >= n)) throw std::runtime_error("Invalid posting row");
  }
}
}  // namespace rcfs
