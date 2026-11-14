#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>
#include <memory>

#include "rcfs/attribute_order.h"
#include "rcfs/types.h"

namespace rcfs {

struct EdgeSpan {
  uint64_t offset = 0;
  uint32_t length = 0;
};

// The block is eligible when the query's left boundary is in
// [left_begin, left_end). Its edge span addresses neighbor_positions().
struct ForwardBlock {
  Position left_begin = 0;
  Position left_end = 0;
  EdgeSpan edges;
};

struct RangeNodeRecord {
  uint64_t forward_block_offset = 0;
  uint32_t forward_block_count = 0;
  EdgeSpan reverse_edges;
};

struct PredicateViewHandle {
  AttributeId attribute = 0;
  RangeBounds bounds;

  bool empty() const { return bounds.empty(); }
  uint64_t size() const { return bounds.size(); }
};
using RangeGraphHandle = PredicateViewHandle;

// Stores adjacency in local positions. The parent view owns the order and
// converts those positions to original IDs when vectors or attributes are needed.
class RangeRoutingIndex {
 public:
  RangeRoutingIndex() = default;

  // Inserts shared vector rows in attribute order. The temporary builder is
  // released once its boundary-dependent adjacency batches have been collected.
  static RangeRoutingIndex build(const DenseFloatMatrix& vectors,
                                 const AttributeOrder& order,
                                 const BuildParams& params);
  // Imports directed rows as fixed adjacency for every left boundary.
  static RangeRoutingIndex from_adjacency(AttributeId attribute,
      const std::vector<std::vector<Position>>& rows, std::vector<Position> entries);

  // The caller must establish that both views have the same complete ID order.
  RangeRoutingIndex shared_for(AttributeId attribute) const {
    auto copy=*this; copy.attribute_=attribute; return copy;
  }
  bool shares_storage(const RangeRoutingIndex& other) const { return storage_==other.storage_; }
  void compact_payloads();
  AttributeId attribute() const { return attribute_; }
  uint32_t size() const { return node_count_; }
  uint64_t block_count() const { return storage_->blocks.size(); }
  uint64_t stored_neighbor_count() const { return storage_->neighbor_positions.size(); }

  RangeGraphHandle open(const RangeBounds& bounds) const;
  PredicateViewHandle open_predicate_view(const PositionRange& bounds) const { return open(bounds); }
  // Selects the forward block by the left boundary, then unions it with the
  // cumulative reverse row inside the interval. Materializes the full row.
  void neighbors(const PredicateViewHandle&, Position local_node, std::vector<Position>& output) const;
  std::vector<Position> entries(const PredicateViewHandle&, uint32_t limit) const;
  const std::vector<Position>& full_domain_entries() const { return storage_->entries; }

  // Convenience overload when the caller does not reuse an adjacency buffer.
  std::vector<Position> neighbors(const RangeGraphHandle& handle,
                                  Position local_node) const;

  const std::vector<RangeNodeRecord>& nodes() const { return storage_->nodes; }
  const std::vector<ForwardBlock>& blocks() const { return storage_->blocks; }
  const std::vector<Position>& neighbor_positions() const {
    return storage_->neighbor_positions;
  }

  void save(std::ostream& output) const;
  static RangeRoutingIndex load(std::istream& input, uint64_t max_nodes,
                                uint64_t max_blocks, uint64_t max_edges);
  void validate() const;

 private:
  AttributeId attribute_ = 0;
  uint32_t node_count_ = 0;
  struct Storage {
    std::vector<RangeNodeRecord> nodes;
    std::vector<ForwardBlock> blocks;
    std::vector<Position> neighbor_positions;
    std::vector<Position> entries;
  };
  std::shared_ptr<Storage> storage_=std::make_shared<Storage>();
};

}  // namespace rcfs
