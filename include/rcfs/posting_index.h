#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>

#include "rcfs/attribute_order.h"
#include "rcfs/schema.h"
#include "rcfs/types.h"

namespace rcfs {

// Half-open slice. Explicit postings use offsets into positions_, while an
// implicit posting uses positions directly. Use at() to handle either form.
struct PostingSlice {
  CategoryCode code = 0;
  uint64_t begin_index = 0;
  uint64_t end_index = 0;

  bool empty() const { return begin_index >= end_index; }
  uint64_t size() const {
    return empty() ? 0 : end_index - begin_index;
  }
};

// Categorical postings in one numeric or categorical view's position space.
// CategoryCode identifies a field and value through the shared Schema.
class PostingIndex {
 public:
  PostingIndex() = default;

  static PostingIndex build(
      const AttributeOrder& order, const Schema& schema,
      const std::vector<CategoricalColumn>& categorical_columns);

  AttributeId order_attribute() const { return order_attribute_; }
  uint32_t node_count() const { return node_count_; }
  uint32_t category_code_count() const { return category_code_count_; }
  // Counts stored positions only. Implicit category blocks need no payload.
  uint64_t occurrence_count() const { return positions_.size(); }

  bool contains_code(CategoryCode code) const {
    return code < category_code_count_;
  }
  uint64_t posting_size(CategoryCode code) const;
  PostingSlice full_slice(CategoryCode code) const;
  PostingSlice slice(CategoryCode code, const RangeBounds& bounds) const;
  uint64_t count(CategoryCode code, const RangeBounds& bounds) const {
    return slice(code, bounds).size();
  }
  Position at(const PostingSlice& slice, uint64_t relative_index) const;

  const std::vector<uint64_t>& offsets() const { return offsets_; }
  const std::vector<Position>& positions() const { return positions_; }

  void save(std::ostream& output) const;
  static PostingIndex load(std::istream& input, uint64_t max_nodes,
                           uint32_t max_category_codes,
                           uint64_t max_posting_positions);
  void validate(uint64_t node_count) const;

 private:
  AttributeId order_attribute_ = 0;
  uint32_t node_count_ = 0;
  uint32_t category_code_count_ = 0;
  std::vector<uint64_t> offsets_;
  std::vector<Position> positions_;
  // A categorical view's own field is already contiguous in its order.
  CategoryCode implicit_begin_ = kMissingValue;
  std::vector<Position> implicit_bounds_;
  std::vector<ValueId> implicit_ranks_;
  bool implicit(CategoryCode code) const {
    return implicit_begin_ != kMissingValue && code >= implicit_begin_ &&
        uint64_t(code - implicit_begin_) + 1 < implicit_bounds_.size();
  }
};

}  // namespace rcfs
