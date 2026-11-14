#pragma once

#include <string>
#include <vector>

#include "rcfs/attribute_order.h"
#include "rcfs/posting_index.h"
#include "rcfs/range_routing_index.h"
#include "rcfs/schema.h"
#include "rcfs/types.h"

namespace rcfs {

struct RoutingStatistics {
  double min_key = 0, max_key = 0;
  uint64_t distinct_keys = 0, forward_blocks = 0, stored_neighbors = 0;
  uint32_t entry_count = 0;
  std::vector<double> numeric_quantiles;
};
// A logical view owns its order and postings. Its graph storage may be shared
// with another view whose full original-ID permutation is identical.
struct UnifiedRoutingView {
  AttributeOrder order;
  PostingIndex postings;
  RangeRoutingIndex routing;
  RoutingStatistics statistics;
};
using AttributeView = UnifiedRoutingView;
RoutingStatistics collect_routing_statistics(const UnifiedRoutingView&);

// All views refer to this single vector matrix and set of attribute columns.
class RcfsIndex {
 public:
  RcfsIndex() = default;

  void save(const std::string& prefix) const;
  static RcfsIndex load(const std::string& prefix);
  void validate() const;

  uint32_t size() const { return dataset_.vectors.rows; }
  uint32_t dim() const { return dataset_.vectors.cols; }
  const Dataset& dataset() const { return dataset_; }
  const Schema& schema() const { return schema_; }
  const BuildParams& build_params() const { return build_params_; }
  const std::vector<AttributeView>& views() const { return views_; }

  const AttributeView& view(AttributeId attribute) const;
  // Accepts an original ID, never a view-local position.
  const float* vector(NodeId id) const;

 private:
  Dataset dataset_;
  Schema schema_;
  BuildParams build_params_;
  std::vector<AttributeView> views_;

  friend class RcfsIndexBuilder;
};

}  // namespace rcfs
