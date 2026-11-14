#pragma once
#include "rcfs/types.h"
namespace rcfs { namespace detail {
// Categorical mismatches contribute to violations only. Gap sums numerical deviations.
struct Bridge { uint32_t violations; float gap,distance; Position position; NodeId node; };
struct BestBridge {
  BridgePriority priority=BridgePriority::kResidual;
  // Reverse the priority tuple so std::priority_queue exposes the best Bridge.
  bool operator()(const Bridge& a,const Bridge& b) const {
    if (priority==BridgePriority::kResidual) {
      if (a.violations!=b.violations) return a.violations>b.violations;
      if (a.gap!=b.gap) return a.gap>b.gap;
    }
    if (a.distance!=b.distance) return a.distance>b.distance;
    return a.node>b.node;
  }
};
}} // namespace rcfs::detail
