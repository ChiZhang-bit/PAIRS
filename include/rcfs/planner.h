#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rcfs/index.h"
#include "rcfs/predicate.h"
#include "rcfs/types.h"
#include "rcfs/query_scratch.h"

namespace rcfs {

enum class CandidateSourceKind : uint8_t {
  kRangeSlice = 0,
  kPostingSlice = 1,
};

// A complete source covers every clause match. Its positions belong to view_attribute.
struct CandidateSource {
  CandidateSourceKind kind = CandidateSourceKind::kRangeSlice;
  AttributeId view_attribute = 0;
  RangeBounds range;
  PostingSlice posting;

  uint64_t size() const {
    return kind == CandidateSourceKind::kRangeSlice ? range.size()
                                                     : posting.size();
  }
  bool empty() const { return size() == 0; }
};

struct ProbeStats {
  uint32_t sample_count = 0;
  uint32_t pass_count = 0;
  double source_pass_rate = 0.0;
  double estimated_joint_count = 0.0;
  double range_match_density = 0.0;
  double joint_count_lower = 0.0, joint_count_upper = 0.0;
  double joint_count_unbiased = 0.0;
  double conservative_count_lower = 0.0, conservative_count_upper = 0.0;
  double driver_match_density = 0.0, density_lower = 0.0;
  bool exact = false;
  std::vector<Position> sampled_positions;  // Positions in the complete source's order.
  std::vector<uint8_t> sampled_matches;
};

struct ClausePlan {
  bool valid = false;
  uint32_t clause_id = 0;
  AttributeId driver_attribute = kInvalidAttribute;
  AttributeType driver_type = AttributeType::kNumeric;
  size_t driver_range_index = static_cast<size_t>(-1);
  size_t driver_equality_index = static_cast<size_t>(-1);
  RangeBounds bounds;
  CandidateSource source;
  ProbeStats probe;
  PlanKind plan = PlanKind::kAuto;
  double exact_cost = 0.0;
  double graph_cost = 0.0;
  uint64_t estimated_graph_visits = 0;  // Required discoveries clipped to the Driver budget.
  uint64_t required_graph_visits = 0, budget_gap = 0, estimated_expansions = 0;
  std::string quality_status;
  uint64_t planning_predicate_checks = 0;
  uint64_t probe_cache_hits = 0;
  double posting_lookup_time_ms = 0.0;
  std::string candidate_source_name;
  std::string driver_atom, empty_reason;
};

class ClausePlanner {
 public:
  static ClausePlan choose(const RcfsIndex& index,
                           const CompiledClause& clause,
                           const SearchParams& params,
                           uint32_t clause_id = 0,
                           ProbeWorkspace* workspace = nullptr);

  // Returns a view position, not an original object ID.
  static Position source_position(const RcfsIndex& index,
                                  const CandidateSource& source,
                                  uint64_t relative_index);
};
// Populate cost coefficients in nanoseconds using warmed local measurements.
void calibrate_costs(const RcfsIndex&, SearchParams&, uint32_t samples = 256);

}  // namespace rcfs
