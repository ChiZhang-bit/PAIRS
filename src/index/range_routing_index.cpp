#include "rcfs/range_routing_index.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

// Keep builder-only types out of the public routing interface.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "base_hnsw/space_l2.h"
#include "range_graph/range_segment_graph_backend.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace rcfs {
namespace {

constexpr std::array<char, 8> kGraphMagic = {'R', 'C', 'F', 'S',
                                              'G', 'R', 'F', '4'};
constexpr uint32_t kGraphVersion = 4;

template <typename T>
void write_pod(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!output) {
    throw std::runtime_error("Failed to serialize RangeRoutingIndex");
  }
}

template <typename T>
T read_pod(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("Truncated RangeRoutingIndex serialization");
  }
  return value;
}

template <typename T>
void write_array(std::ostream& output, const std::vector<T>& values) {
  if (values.empty()) return;
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!output) {
    throw std::runtime_error("Failed to serialize RangeRoutingIndex array");
  }
}

void ensure_remaining_bytes(std::istream& input, uint64_t required_bytes,
                            const char* context) {
  if (required_bytes >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(context) + " is too large");
  }
  const std::ios::iostate original_state = input.rdstate();
  const std::istream::pos_type current = input.tellg();
  if (current == std::istream::pos_type(-1)) {
    input.clear(original_state);
    return;
  }
  input.seekg(0, std::ios::end);
  const std::istream::pos_type end = input.tellg();
  input.seekg(current);
  if (!input || end == std::istream::pos_type(-1)) {
    throw std::runtime_error(std::string("Cannot inspect ") + context);
  }
  if (end - current < static_cast<std::streamoff>(required_bytes)) {
    throw std::runtime_error(std::string("Truncated ") + context);
  }
}

template <typename T>
std::vector<T> read_array(std::istream& input, uint64_t count) {
  if (count > static_cast<uint64_t>(
                  std::numeric_limits<size_t>::max() / sizeof(T))) {
    throw std::runtime_error("RangeRoutingIndex array size overflows size_t");
  }
  const uint64_t required_bytes = count * sizeof(T);
  ensure_remaining_bytes(input, required_bytes, "RangeRoutingIndex array");
  std::vector<T> values(static_cast<size_t>(count));
  if (!values.empty()) {
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) {
      throw std::runtime_error("Truncated RangeRoutingIndex array");
    }
  }
  return values;
}

uint32_t checked_u32_size(size_t size, const char* what) {
  if (size > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string(what) + " exceeds uint32");
  }
  return static_cast<uint32_t>(size);
}

}  // namespace

RangeRoutingIndex RangeRoutingIndex::build(const DenseFloatMatrix& vectors,
                                           const AttributeOrder& order,
                                           const BuildParams& params) {
  order.validate();
  if (vectors.rows == 0 || vectors.cols == 0 ||
      vectors.values.size() !=
          static_cast<size_t>(vectors.rows) * vectors.cols) {
    throw std::invalid_argument("Invalid vector matrix for range routing");
  }
  if (vectors.rows != order.size()) {
    throw std::invalid_argument(
        "Vector row count does not match AttributeOrder");
  }
  for (float value : vectors.values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "Range routing cannot be built from non-finite vectors");
    }
  }
  if (vectors.rows > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "Bundled range backend requires node count <= INT_MAX");
  }
  if (params.M < 2 || params.ef_construction < params.M ||
      params.ef_max < params.M) {
    throw std::invalid_argument(
        "Range routing requires M >= 2 and ef values >= M");
  }

  const uint32_t node_count = vectors.rows;
  RangeRoutingIndex result;
  result.attribute_ = order.attribute();
  result.node_count_ = node_count;
  std::vector<rcfs_range_backend::SegmentedForwardNeighbors> raw(node_count);
  {
    base_hnsw::L2Space space(vectors.cols);
    rcfs_range_backend::SegmentedRangeHNSW<float> builder(
        &space, std::max<size_t>(2, node_count), params.M,
        params.ef_construction, params.ef_max, params.random_seed);
    builder.segment_graph = &raw;
    // Batch boundaries refer to insertion positions, so insert in attribute order.
    // Read each row from the shared matrix without making a sorted matrix copy.
    for (Position pos = 0; pos < node_count; ++pos)
      builder.addPoint(vectors.values.data() + size_t(order.node_at(pos)) * vectors.cols, pos);
    result.storage_->entries.push_back(static_cast<Position>(builder.getExternalLabel(builder.enterpoint_node_)));
  }  // Free the temporary HNSW and its vector store before allocating flat adjacency.
  result.storage_->nodes.resize(node_count);
  std::vector<uint64_t> reverse_offsets(size_t(node_count) + 1, 0);
  std::vector<Position> mark(node_count, kInvalidNodeId);
  for (Position src = 0; src < node_count; ++src) {
    auto& record = result.storage_->nodes[src];
    record.forward_block_offset = result.storage_->blocks.size();
    Position previous_end = 0;
    for (auto& rb : raw[src].forward_nns) {
      // Count reverse edges before dropping zero-width forward batches. Their
      // edges still belong to the cumulative reverse row. mark deduplicates
      // each source across all of its batches.
      for (int nb : rb.nns_id) {
        if (nb < 0 || nb >= static_cast<int>(node_count)) throw std::runtime_error("Invalid backend neighbor");
        if (Position(nb) != src && mark[nb] != src) {
          mark[nb] = src; ++reverse_offsets[size_t(nb) + 1];
        }
      }
      Position end = rb.end > 0 ? Position(std::min<int64_t>(rb.end, node_count)) : 0;
      if (end < previous_end) throw std::runtime_error("Nonmonotonic backend blocks");
      if (end == previous_end) continue;
      std::sort(rb.nns_id.begin(), rb.nns_id.end());
      rb.nns_id.erase(std::unique(rb.nns_id.begin(), rb.nns_id.end()), rb.nns_id.end());
      ForwardBlock block;
      block.left_begin = previous_end; block.left_end = end;
      block.edges.offset = result.storage_->neighbor_positions.size();
      for (int nb : rb.nns_id) if (Position(nb) != src) result.storage_->neighbor_positions.push_back(Position(nb));
      block.edges.length = checked_u32_size(result.storage_->neighbor_positions.size() - block.edges.offset, "Forward degree");
      result.storage_->blocks.push_back(block);
      previous_end = end;
    }
    record.forward_block_count = checked_u32_size(result.storage_->blocks.size() - record.forward_block_offset, "Block count");
  }
  for (size_t i = 1; i < reverse_offsets.size(); ++i) reverse_offsets[i] += reverse_offsets[i-1];
  const uint64_t reverse_base = result.storage_->neighbor_positions.size();
  if (reverse_offsets.back() > std::numeric_limits<size_t>::max() - reverse_base)
    throw std::overflow_error("Reverse adjacency too large");
  result.storage_->neighbor_positions.resize(size_t(reverse_base + reverse_offsets.back()));
  auto cursor = reverse_offsets;
  std::fill(mark.begin(), mark.end(), kInvalidNodeId);
  // Increasing source positions fill sorted reverse rows in one allocation.
  // Use the same per-source deduplication as the counting pass above.
  for (Position src = 0; src < node_count; ++src) {
    for (const auto& rb : raw[src].forward_nns)
      for (int nb : rb.nns_id)
        if (Position(nb) != src && mark[nb] != src) {
          mark[nb] = src;
          result.storage_->neighbor_positions[size_t(reverse_base + cursor[nb]++)] = src;
        }
    std::vector<rcfs_range_backend::OneSegmentNeighbors>().swap(raw[src].forward_nns);
  }
  for (Position pos = 0; pos < node_count; ++pos)
    result.storage_->nodes[pos].reverse_edges = {reverse_base + reverse_offsets[pos],
        checked_u32_size(reverse_offsets[size_t(pos)+1] - reverse_offsets[pos], "Reverse degree")};

  result.compact_payloads();
  result.validate();
  return result;
}

void RangeRoutingIndex::compact_payloads() {
  // Detach before rewriting offsets so other views retain their shared graph.
  if (!storage_.unique()) storage_=std::make_shared<Storage>(*storage_);
  const auto& old=storage_->neighbor_positions;
  std::vector<Position> packed; packed.reserve(old.size());
  std::unordered_multimap<uint64_t,EdgeSpan> interned;
  auto intern=[&](EdgeSpan span) {
    if (!span.length) return EdgeSpan{};
    uint64_t hash=14695981039346656037ULL;
    for (uint64_t i=span.offset;i<span.offset+span.length;++i) { hash^=old[i]; hash*=1099511628211ULL; }
    auto range=interned.equal_range(hash);
    for (auto it=range.first;it!=range.second;++it) {
      const auto& candidate=it->second;
      // A hash match is not enough to share a payload.
      if (candidate.length==span.length && std::equal(old.begin()+span.offset,old.begin()+span.offset+span.length,
          packed.begin()+candidate.offset)) return candidate;
    }
    EdgeSpan added{packed.size(),span.length};
    packed.insert(packed.end(),old.begin()+span.offset,old.begin()+span.offset+span.length);
    interned.emplace(hash,added); return added;
  };
  std::vector<ForwardBlock> blocks; blocks.reserve(storage_->blocks.size());
  for (auto& node:storage_->nodes) {
    const auto offset=node.forward_block_offset;
    const auto count=node.forward_block_count;
    node.forward_block_offset=blocks.size();
    for (uint32_t i=0;i<count;++i) {
      auto block=storage_->blocks[offset+i]; block.edges=intern(block.edges);
      // Adjacent boundary intervals can merge only when their full rows agree.
      if (blocks.size()>node.forward_block_offset && blocks.back().left_end==block.left_begin &&
          blocks.back().edges.offset==block.edges.offset && blocks.back().edges.length==block.edges.length)
        blocks.back().left_end=block.left_end;
      else blocks.push_back(block);
    }
    node.forward_block_count=checked_u32_size(blocks.size()-node.forward_block_offset,"Compacted blocks");
    node.reverse_edges=intern(node.reverse_edges);
  }
  packed.shrink_to_fit(); blocks.shrink_to_fit();
  storage_->neighbor_positions=std::move(packed); storage_->blocks=std::move(blocks);
}

RangeRoutingIndex RangeRoutingIndex::from_adjacency(AttributeId attribute,
    const std::vector<std::vector<Position>>& rows, std::vector<Position> entries) {
  if (rows.empty() || rows.size()>static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Invalid fixed graph size");
  RangeRoutingIndex out;
  out.attribute_=attribute; out.node_count_=static_cast<uint32_t>(rows.size());
  out.storage_->entries=std::move(entries);
  out.storage_->nodes.resize(rows.size());
  for (Position i=0;i<rows.size();++i) {
    auto row=rows[i];
    std::sort(row.begin(),row.end());
    row.erase(std::unique(row.begin(),row.end()),row.end());
    row.erase(std::remove(row.begin(),row.end(),i),row.end());
    ForwardBlock block;
    block.left_begin=0; block.left_end=out.node_count_;
    block.edges={out.storage_->neighbor_positions.size(),checked_u32_size(row.size(),"Imported degree")};
    out.storage_->nodes[i].forward_block_offset=out.storage_->blocks.size();
    out.storage_->nodes[i].forward_block_count=1;
    out.storage_->blocks.push_back(block);
    out.storage_->neighbor_positions.insert(out.storage_->neighbor_positions.end(),row.begin(),row.end());
  }
  out.validate(); return out;
}

RangeGraphHandle RangeRoutingIndex::open(const RangeBounds& bounds) const {
  if (bounds.begin > bounds.end || bounds.end > node_count_) {
    throw std::out_of_range("Range bounds exceed routing position space");
  }
  return {attribute_, bounds};
}

std::vector<Position> RangeRoutingIndex::neighbors(const RangeGraphHandle& handle, Position node) const {
  std::vector<Position> result;
  neighbors(handle, node, result);
  return result;
}
void RangeRoutingIndex::neighbors(const PredicateViewHandle& handle, Position node,
                                 std::vector<Position>& out) const {
  out.clear();
  if (handle.attribute != attribute_ || handle.bounds.begin > handle.bounds.end ||
      handle.bounds.end > node_count_ || node >= node_count_)
    throw std::out_of_range("Invalid predicate view handle/node");
  if (!handle.bounds.contains(node)) return;
  const auto& record = storage_->nodes[node];
  auto bb = storage_->blocks.begin() + static_cast<ptrdiff_t>(record.forward_block_offset);
  auto be = bb + record.forward_block_count;
  auto block = std::lower_bound(bb, be, handle.bounds.begin,
      [](const ForwardBlock& b, Position left) { return b.left_end <= left; });
  EdgeSpan f{};
  if (block != be && block->left_begin <= handle.bounds.begin) f = block->edges;
  // Both rows are sorted. Clip them before merging so traversal receives only
  // in-domain positions, with duplicates removed by set_union.
  auto clipped = [&](EdgeSpan span) {
    auto b = storage_->neighbor_positions.begin() + static_cast<ptrdiff_t>(span.offset);
    auto e = b + span.length;
    b = std::lower_bound(b, e, handle.bounds.begin);
    e = std::lower_bound(b, e, handle.bounds.end);
    return std::make_pair(b, e);
  };
  auto a = clipped(f), b = clipped(record.reverse_edges);
  out.reserve(size_t(a.second-a.first) + size_t(b.second-b.first));
  std::set_union(a.first, a.second, b.first, b.second, std::back_inserter(out));
}
std::vector<Position> RangeRoutingIndex::entries(const PredicateViewHandle& handle, uint32_t limit) const {
  if (handle.attribute != attribute_ || handle.bounds.begin > handle.bounds.end || handle.bounds.end > node_count_)
    throw std::out_of_range("Invalid entry interval");
  std::vector<Position> out;
  if (!limit || handle.empty()) return out;
  auto add = [&](Position p) {
    if (out.size() < limit && handle.bounds.contains(p) && std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
  };
  for (Position p : storage_->entries) add(p);
  add(handle.bounds.begin); add(handle.bounds.end-1);
  for (uint32_t i = 0; i < limit && out.size() < limit; ++i)
    add(Position(handle.bounds.begin + (handle.size() * (2ULL*i+1)) / (2ULL*limit)));
  return out;
}

void RangeRoutingIndex::save(std::ostream& output) const {
  validate();
  output.write(kGraphMagic.data(),
               static_cast<std::streamsize>(kGraphMagic.size()));
  if (!output) {
    throw std::runtime_error("Failed to serialize RangeRoutingIndex");
  }
  write_pod(output, kGraphVersion);
  write_pod(output, attribute_);
  write_pod(output, node_count_);
  write_pod(output, static_cast<uint64_t>(storage_->blocks.size()));
  write_pod(output, static_cast<uint64_t>(storage_->neighbor_positions.size()));

  for (const RangeNodeRecord& node : storage_->nodes) {
    write_pod(output, node.forward_block_offset);
    write_pod(output, node.forward_block_count);
    write_pod(output, node.reverse_edges.offset);
    write_pod(output, node.reverse_edges.length);
  }
  for (const ForwardBlock& block : storage_->blocks) {
    write_pod(output, block.left_begin);
    write_pod(output, block.left_end);
    write_pod(output, block.edges.offset);
    write_pod(output, block.edges.length);
  }
  write_array(output, storage_->neighbor_positions);
  write_pod(output, static_cast<uint32_t>(storage_->entries.size()));
  write_array(output, storage_->entries);
}

RangeRoutingIndex RangeRoutingIndex::load(std::istream& input,
                                          uint64_t max_nodes,
                                          uint64_t max_blocks,
                                          uint64_t max_edges) {
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kGraphMagic) {
    throw std::runtime_error("Invalid RangeRoutingIndex serialization magic");
  }
  if (read_pod<uint32_t>(input) != kGraphVersion) {
    throw std::runtime_error(
        "Unsupported RangeRoutingIndex serialization version");
  }

  RangeRoutingIndex result;
  result.attribute_ = read_pod<AttributeId>(input);
  result.node_count_ = read_pod<uint32_t>(input);
  const uint64_t block_count = read_pod<uint64_t>(input);
  const uint64_t edge_count = read_pod<uint64_t>(input);
  // Reject invalid counts before they can drive large allocations.
  if (result.node_count_ == 0 || result.node_count_ > max_nodes) {
    throw std::runtime_error("RangeRoutingIndex node count exceeds load limit");
  }
  if (block_count > max_blocks ||
      block_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("RangeRoutingIndex block count exceeds load limit");
  }
  if (edge_count > max_edges ||
      edge_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("RangeRoutingIndex edge count exceeds load limit");
  }

  constexpr uint64_t kSerializedNodeBytes =
      sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) +
      sizeof(uint32_t);
  constexpr uint64_t kSerializedBlockBytes =
      sizeof(Position) + sizeof(Position) + sizeof(uint64_t) +
      sizeof(uint32_t);
  if (block_count >
          (std::numeric_limits<uint64_t>::max() -
           static_cast<uint64_t>(result.node_count_) * kSerializedNodeBytes) /
              kSerializedBlockBytes) {
    throw std::runtime_error(
        "RangeRoutingIndex serialized block size overflows uint64");
  }
  const uint64_t fixed_bytes =
      static_cast<uint64_t>(result.node_count_) * kSerializedNodeBytes +
      block_count * kSerializedBlockBytes;
  if (edge_count >
      (std::numeric_limits<uint64_t>::max() - fixed_bytes) /
          sizeof(Position)) {
    throw std::runtime_error(
        "RangeRoutingIndex serialized edge size overflows uint64");
  }
  ensure_remaining_bytes(input, fixed_bytes + edge_count * sizeof(Position),
                         "RangeRoutingIndex payload");

  result.storage_->nodes.resize(result.node_count_);
  for (RangeNodeRecord& node : result.storage_->nodes) {
    node.forward_block_offset = read_pod<uint64_t>(input);
    node.forward_block_count = read_pod<uint32_t>(input);
    node.reverse_edges.offset = read_pod<uint64_t>(input);
    node.reverse_edges.length = read_pod<uint32_t>(input);
  }
  result.storage_->blocks.resize(static_cast<size_t>(block_count));
  for (ForwardBlock& block : result.storage_->blocks) {
    block.left_begin = read_pod<Position>(input);
    block.left_end = read_pod<Position>(input);
    block.edges.offset = read_pod<uint64_t>(input);
    block.edges.length = read_pod<uint32_t>(input);
  }
  result.storage_->neighbor_positions = read_array<Position>(input, edge_count);
  const uint32_t entry_count = read_pod<uint32_t>(input);
  if (entry_count > result.node_count_) throw std::runtime_error("Invalid entry count");
  result.storage_->entries = read_array<Position>(input, entry_count);
  result.validate();
  return result;
}

void RangeRoutingIndex::validate() const {
  if (node_count_ == 0 || storage_->nodes.size() != node_count_) {
    throw std::runtime_error("RangeRoutingIndex node array is inconsistent");
  }

  auto span_is_valid = [&](const EdgeSpan& span) {
    return span.offset <= storage_->neighbor_positions.size() &&
           span.length <= storage_->neighbor_positions.size() - span.offset;
  };

  for (Position node_id = 0; node_id < node_count_; ++node_id) {
    const RangeNodeRecord& node = storage_->nodes[node_id];
    if (node.forward_block_offset > storage_->blocks.size() ||
        node.forward_block_count >
            storage_->blocks.size() - node.forward_block_offset ||
        !span_is_valid(node.reverse_edges)) {
      throw std::runtime_error("RangeRoutingIndex node spans are invalid");
    }

    Position previous_end = 0;
    for (uint32_t i = 0; i < node.forward_block_count; ++i) {
      const ForwardBlock& block =
          storage_->blocks[static_cast<size_t>(node.forward_block_offset) + i];
      if (block.left_begin != previous_end ||
          block.left_begin >= block.left_end || block.left_end > node_count_ ||
          !span_is_valid(block.edges)) {
        throw std::runtime_error(
            "RangeRoutingIndex forward block is invalid");
      }
      previous_end = block.left_end;
    }
  }

  auto sorted_span = [&](EdgeSpan span) {
    auto b = storage_->neighbor_positions.begin() + static_cast<ptrdiff_t>(span.offset);
    auto e = b + span.length;
    if (!std::is_sorted(b, e) || std::adjacent_find(b, e) != e)
      throw std::runtime_error("Neighbor span must be sorted and unique");
  };
  for (const auto& n : storage_->nodes) sorted_span(n.reverse_edges);
  for (const auto& b : storage_->blocks) sorted_span(b.edges);
  for (Position p : storage_->entries) if (p >= node_count_) throw std::runtime_error("Invalid graph entry");
  for (Position neighbor : storage_->neighbor_positions) {
    if (neighbor >= node_count_) {
      throw std::runtime_error(
          "RangeRoutingIndex contains an invalid neighbor position");
    }
  }
}

}  // namespace rcfs
