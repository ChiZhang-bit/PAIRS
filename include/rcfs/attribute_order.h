#pragma once
#include <cstdint>
#include <iosfwd>
#include <variant>
#include <vector>
#include <string>
#include "rcfs/types.h"
namespace rcfs {
// Positions belong to one attribute order. All intervals are half-open.
struct PositionRange {
  Position begin = 0, end = 0;
  bool empty() const { return begin >= end; }
  uint64_t size() const { return empty() ? 0 : uint64_t(end) - begin; }
  bool contains(Position p) const { return p >= begin && p < end; }
};
using RangeBounds = PositionRange;
// Maps between original IDs and one attribute's sorted positions.
// Numeric keys stay float32, while categorical ranks stay integers.
class AttributeOrder {
 public:
  static AttributeOrder build(AttributeId, const std::vector<float>&);
  // The size-only overload treats codes as ranks. The dictionary overload
  // derives canonical ranks from labels before ordering the objects.
  static AttributeOrder build_categorical(AttributeId, const std::vector<ValueId>&, uint32_t dictionary_size);
  static AttributeOrder build_categorical(AttributeId, const std::vector<ValueId>&,
      const std::vector<std::string>& dictionary);
  ValueId rank_of(ValueId code) const { return code==kMissingValue ? kMissingValue : code_to_rank_.at(code); }
  const std::vector<ValueId>& code_to_rank() const { return code_to_rank_; }
  AttributeId attribute() const { return attribute_; }
  AttributeType type() const { return type_; }
  uint32_t size() const { return static_cast<uint32_t>(pos_to_id_.size()); }
  bool empty() const { return pos_to_id_.empty(); }
  // Keep query endpoints in double so rounding cannot change membership.
  PositionRange map_closed(double lower, double upper) const;
  PositionRange map_equal(ValueId code) const;
  NodeId node_at(Position p) const { return pos_to_id_.at(p); }
  Position position_of(NodeId n) const { return id_to_pos_.at(n); }
  const std::vector<float>& sorted_values() const { return std::get<0>(keys_); }
  // With a dictionary, these keys are ranks rather than external codes.
  const std::vector<ValueId>& sorted_codes() const { return std::get<1>(keys_); }
  const std::vector<Position>& category_boundaries() const { return category_boundaries_; }
  const std::vector<NodeId>& pos_to_id() const { return pos_to_id_; }
  const std::vector<Position>& id_to_pos() const { return id_to_pos_; }
  double min_key() const;
  double max_key() const;
  void save(std::ostream&) const;
  static AttributeOrder load(std::istream&, uint64_t max_nodes);
  void validate() const;
 private:
  AttributeId attribute_ = 0;
  AttributeType type_ = AttributeType::kNumeric;
  std::variant<std::vector<float>, std::vector<ValueId>> keys_;
  std::vector<NodeId> pos_to_id_;
  std::vector<Position> id_to_pos_;
  // C category blocks followed by a missing-value block need C+2 boundaries.
  // Missing objects remain in the full domain but match no equality predicate.
  std::vector<Position> category_boundaries_;
  std::vector<ValueId> code_to_rank_;
};
}  // namespace rcfs
