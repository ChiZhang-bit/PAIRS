#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace rcfs {

using NodeId = uint32_t;
using Position = uint32_t;
using AttributeId = uint32_t;
enum class AttributeType : uint8_t { kNumeric = 0, kCategorical = 1 };
enum class DriverPolicy : uint8_t { kAll = 0, kNumericOnly = 1, kSmallestDomain = 2 };
enum class BridgePriority : uint8_t { kResidual = 0, kDistance = 1 };
constexpr AttributeId kInvalidAttribute = std::numeric_limits<AttributeId>::max();
using NumericAttrId = uint16_t;
using CategoricalAttrId = uint16_t;
using ValueId = uint32_t;
using CategoryCode = uint32_t;

constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();
constexpr ValueId kMissingValue = std::numeric_limits<ValueId>::max();

struct DenseFloatMatrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<float> values;
};

struct NumericColumn {
  std::string name;
  std::vector<float> values;
};

struct CategoricalColumn {
  std::string name;
  std::vector<std::string> value_names;
  std::vector<ValueId> values;
  CategoryCode code_offset = 0;
};

struct Dataset {
  DenseFloatMatrix vectors;
  std::vector<NumericColumn> numeric_columns;
  std::vector<CategoricalColumn> categorical_columns;
};

struct BuildParams {
  uint32_t M = 16;
  uint32_t ef_construction = 400;
  uint32_t ef_max = 2048;
  uint32_t random_seed = 2023;
  AttributeId canonical_full_range_attribute = 0;  // Unified attribute ID used for TRUE clauses.
  bool normalize = false;
};

enum class PlanKind : uint8_t {
  kAuto = 0,
  kExactMeta = 1,
  kGraphBridge = 2,
};

struct QualityRecord {
  uint32_t bucket = 0, samples = 0;
  double recall_lower = 0.0;  // held-out empirical 10th percentile
};
struct SearchParams {
  uint32_t k = 10;
  uint32_t max_clauses = 32;
  uint32_t probe_budget = 64;
  uint32_t total_probe_budget = 256;
  uint32_t probe_seed = 2023;
  uint32_t distance_cache_entries = 65536;
  bool auto_distance_cache = true;
  uint32_t threads = 1;
  uint32_t probe_strata = 8;
  uint32_t exact_scan_threshold = 4096;
  uint32_t entry_probe_budget = 24;
  // Per-clause limits, clipped to the Driver domain before search.
  uint32_t max_visit_budget = 1024;
  uint32_t max_expand_budget = 1024;
  uint32_t ef_match = 64;  // fixed navigation width is max(k, ef_match)
  bool global_exact_guard = true;
  double target_recall = 0.0;  // 0 disables the quality policy; 1 requests EXACT.
  uint32_t quality_min_samples = 20;
  uint64_t quality_config = 0;
  std::vector<QualityRecord> quality_records;
  uint32_t max_bridge_budget = 512;  // Counts expanded Bridge rows, not discoveries.
  float bridge_min_share = 0.10F;
  float bridge_max_share = 0.90F;
  double predicate_cost = 1.0;
  double distance_cost = 8.0;
  double edge_cost = 0.25;  // Cost per enumerated adjacency row, not per edge.
  double open_cost = 1.0;
  std::vector<double> view_open_costs, view_edge_costs;
  double supply_slack = 4.0;
  double density_epsilon = 1.0e-6;
  PlanKind forced_plan = PlanKind::kAuto;
  DriverPolicy driver_policy = DriverPolicy::kAll;
  BridgePriority bridge_priority = BridgePriority::kResidual;
  AttributeId forced_driver = kInvalidAttribute;
};

struct ClauseDiagnostics {
  uint32_t clause_id = 0;
  AttributeId driver_attribute = kInvalidAttribute;
  AttributeType driver_type = AttributeType::kNumeric;
  std::string driver_atom;
  std::string empty_reason;
  PlanKind plan = PlanKind::kAuto;
  uint64_t driver_range_count = 0;
  std::string candidate_source_type;
  uint64_t candidate_source_count = 0;
  uint32_t probe_sample_count = 0;
  uint32_t probe_pass_count = 0;
  double source_pass_rate = 0.0;
  double estimated_joint_count = 0.0;
  double joint_count_lower = 0.0;
  double joint_count_upper = 0.0;
  uint64_t probe_cache_hits = 0;
  double estimated_range_match_density = 0.0;
  double estimated_exact_cost = 0.0;
  double estimated_graph_cost = 0.0;
  uint64_t final_visit_budget = 0;
  uint64_t final_bridge_budget = 0;
  uint64_t visited_nodes = 0;
  uint64_t expanded_match_nodes = 0;
  uint64_t expanded_bridge_nodes = 0;
  uint64_t dormant_bridge_peak = 0;
  uint64_t distance_computations = 0;
  uint64_t predicate_checks = 0;
  uint64_t range_pruned_neighbors = 0;
  uint64_t edge_visits = 0;
  uint64_t required_graph_visits = 0, budget_gap = 0;
  uint64_t estimated_expansions = 0;
  uint64_t final_expand_budget = 0;
  uint32_t final_ef_match = 0;  // effective fixed width, including the k floor
  bool budget_limited = false;
  std::string quality_status;
  uint32_t quality_bucket = 0;
  double empirical_recall_lower = -1.0;
  double joint_count_unbiased = 0.0;
  double conservative_count_lower = 0.0, conservative_count_upper = 0.0;
  double observed_yield = 0.0, result_gain_per_distance = 0.0;
  double posting_lookup_time_ms = 0.0;
  double planning_time_ms = 0.0;
  double execution_time_ms = 0.0;
};

struct SearchResult {
  std::vector<NodeId> ids;
  std::vector<float> distances;
  std::vector<ClauseDiagnostics> clauses;
  uint32_t compiled_clause_count = 0;
  uint64_t distance_computations = 0;
  uint64_t predicate_checks = 0;
  uint64_t visited_nodes = 0;
  double search_time_ms = 0.0;
};

inline const char* plan_kind_name(PlanKind plan) {
  switch (plan) {
    case PlanKind::kAuto:
      return "auto";
    case PlanKind::kExactMeta:
      return "exact_meta";
    case PlanKind::kGraphBridge:
      return "graph_bridge";
  }
  return "unknown";
}

}  // namespace rcfs
