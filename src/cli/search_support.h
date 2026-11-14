#pragma once
// Query CLI parameters, result packing, recall, and per-clause diagnostics.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include "cli/cli_utils.h"
#include "rcfs/dataset_io.h"
#include "rcfs/index.h"
#include "rcfs/predicate.h"
#include "rcfs/searcher.h"
namespace rcfs_cli {
inline rcfs::PlanKind parse_plan(const std::string& value) {
  if (value == "auto") return rcfs::PlanKind::kAuto;
  if (value == "exact_meta") return rcfs::PlanKind::kExactMeta;
  if (value == "graph_bridge") return rcfs::PlanKind::kGraphBridge;
  throw std::invalid_argument("Unknown plan: " + value);
}

inline rcfs::SearchParams make_search_params(const rcfs::Arguments& args, const rcfs::Schema& schema) {
  // Reject old stage controls explicitly so a reused command cannot appear
  // to enable adaptive behavior in this fixed-budget executor.
  for (const char* retired : {"initial-visit-budget", "initial-expand-budget",
                             "initial-bridge-budget", "max-ef-match", "max-restarts", "stall-window"}) {
    if (args.has(retired))
      throw std::invalid_argument(std::string("Removed adaptive-search option --") + retired +
          ". Use fixed --visit-budget, --expand-budget, --max-bridge-budget and --ef-match.");
  }
  rcfs::SearchParams params;
  params.k = args.get_u32("topk", 10);
  params.max_clauses = args.get_u32("max-clauses", 32);
  params.probe_budget = args.get_u32("probe-budget", 64);
  params.total_probe_budget = args.get_u32("total-probe-budget", 256);
  params.probe_seed = args.get_u32("probe-seed", 2023);
  params.threads = args.get_u32("threads", 1);
  params.distance_cache_entries = args.get_u32("distance-cache-entries", 65536);
  const auto cache_mode=args.get("distance-cache-mode","auto");
  if (cache_mode!="auto" && cache_mode!="fixed") throw std::invalid_argument("Unknown distance-cache-mode");
  params.auto_distance_cache=cache_mode=="auto";
  params.open_cost = args.get_double("open-cost", 1.0);
  const auto policy=args.get("driver-policy","all");
  if (policy=="all") params.driver_policy=rcfs::DriverPolicy::kAll;
  else if (policy=="numeric-only") params.driver_policy=rcfs::DriverPolicy::kNumericOnly;
  else if (policy=="smallest-domain") params.driver_policy=rcfs::DriverPolicy::kSmallestDomain;
  else throw std::invalid_argument("Unknown driver policy: "+policy);
  if (args.has("force-driver")) params.forced_driver=schema.attribute_id(args.require("force-driver"));
  params.probe_strata = args.get_u32("probe-strata", 8);
  params.exact_scan_threshold =
      args.get_u32("exact-scan-threshold", 4096);
  params.entry_probe_budget = args.get_u32("entry-probe-budget", 24);
  params.max_visit_budget = args.get_u32("visit-budget", 1024);
  params.max_expand_budget=args.get_u32("expand-budget",1024);
  params.ef_match=args.get_u32("ef-match",64);
  const auto priority=args.get("bridge-priority","residual");
  if (priority=="residual") params.bridge_priority=rcfs::BridgePriority::kResidual;
  else if (priority=="distance") params.bridge_priority=rcfs::BridgePriority::kDistance;
  else throw std::invalid_argument("Unknown bridge priority: "+priority);
  params.global_exact_guard=!args.has("disable-exact-guard");
  params.target_recall=args.get_double("target-recall",0);
  params.quality_min_samples=args.get_u32("quality-min-samples",20);
  params.max_bridge_budget = args.get_u32("max-bridge-budget", 512);
  params.bridge_min_share = args.get_float("bridge-min-share", 0.10F);
  params.bridge_max_share = args.get_float("bridge-max-share", 0.90F);
  params.predicate_cost = args.get_double("predicate-cost", 1.0);
  params.distance_cost = args.get_double("distance-cost", 8.0);
  params.edge_cost = args.get_double("edge-cost", 0.25);
  params.supply_slack = args.get_double("supply-slack", 4.0);
  params.forced_plan = parse_plan(args.get("force-plan", "auto"));
  return params;
}

inline void truncate_workload(rcfs::QueryWorkload* workload, uint32_t max_queries) {
  if (max_queries == 0 || max_queries >= workload->vectors.rows) return;
  workload->vectors.rows = max_queries;
  workload->vectors.values.resize(
      static_cast<size_t>(max_queries) * workload->vectors.cols);
  workload->predicates.resize(max_queries);
}

inline rcfs::ResultMatrix pack_results(
    const std::vector<rcfs::SearchResult>& results, uint32_t k) {
  rcfs::ResultMatrix matrix;
  matrix.rows = static_cast<uint32_t>(results.size());
  matrix.k = k;
  const size_t total = static_cast<size_t>(matrix.rows) * k;
  // Fixed-width rows use matching ID/infinity padding for fewer than k hits.
  matrix.ids.assign(total, rcfs::kInvalidNodeId);
  matrix.distances.assign(total, std::numeric_limits<float>::infinity());
  for (uint32_t q = 0; q < matrix.rows; ++q) {
    const size_t count = std::min<size_t>(k, results[q].ids.size());
    for (size_t i = 0; i < count; ++i) {
      matrix.ids[static_cast<size_t>(q) * k + i] = results[q].ids[i];
      matrix.distances[static_cast<size_t>(q) * k + i] =
          results[q].distances[i];
    }
  }
  return matrix;
}

inline double recall_at_k(const rcfs::ResultMatrix& results,
                   const rcfs::IdMatrix& truth) {
  if (truth.rows < results.rows) {
    throw std::runtime_error("Ground truth has fewer queries than results");
  }
  const uint32_t k = std::min(results.k, truth.cols);
  if (results.rows == 0 || k == 0) return 0.0;
  double total=0.0;
  for (uint32_t q=0;q<results.rows;++q) {
    std::unordered_set<rcfs::NodeId> predicted;
    for (uint32_t i=0;i<results.k;++i) {
      auto id=results.ids[size_t(q)*results.k+i];
      if (id!=rcfs::kInvalidNodeId) predicted.insert(id);
    }
    uint32_t available=0,hits=0;
    for (uint32_t i=0;i<k;++i) {
      auto id=truth.ids[size_t(q)*truth.cols+i];
      if (id!=rcfs::kInvalidNodeId) { ++available; if (predicted.count(id)) ++hits; }
    }
    // Ignore truth padding. An empty answer set scores one only if no object
    // was returned, rather than treating every empty truth row as a success.
    total+=available ? double(hits)/available : (predicted.empty() ? 1.0 : 0.0);
  }
  return results.rows ? total/results.rows : 1.0;

}

inline std::string csv(const std::string& text) {
  std::string out(1,char(34));
  for (char c:text) { if (c==char(34)) out+=c; out+=c; }
  out+=char(34);
  return out;
}

inline void write_stats(const std::string& path,
                 const std::vector<rcfs::SearchResult>& results,
                 const rcfs::Schema& schema) {
  rcfs::ensure_parent_directory(path);
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("Cannot create stats file: " + path);
  output << "query_id,compiled_clause_count,clause_id,driver_attribute,plan,driver_range_count,"
            "candidate_source,candidate_source_count,probe_samples,probe_passes,"
            "source_pass_rate,estimated_joint_count,range_match_density,"
            "estimated_exact_cost,estimated_graph_cost,final_visit_budget,final_bridge_budget,visited_nodes,"
            "expanded_match_nodes,expanded_bridge_nodes,dormant_bridge_peak,"
            "distance_computations,predicate_checks,posting_lookup_ms,planning_ms,execution_ms,total_search_ms,driver_type,driver_atom,joint_count_lower,joint_count_upper,probe_cache_hits,edge_visits,empty_reason,required_graph_visits,budget_gap,estimated_expansions,final_expand_budget,final_ef_match,budget_limited,quality_status,joint_count_unbiased,conservative_count_lower,conservative_count_upper,observed_yield,result_gain_per_distance,quality_bucket,empirical_recall_lower\n";
  output << std::setprecision(10);
  // One row per clause. total_search_ms is repeated for each clause of a query
  // and must not be summed across these rows as though it were clause time.
  for (uint32_t q = 0; q < results.size(); ++q) {
    for (const auto& c : results[q].clauses) {
      output << q << ',' << results[q].compiled_clause_count << ','
             << c.clause_id << ','
             << (c.driver_attribute==rcfs::kInvalidAttribute ? std::string("") : csv(schema.attribute_name(c.driver_attribute))) << ','
             << rcfs::plan_kind_name(c.plan) << ',' << c.driver_range_count
             << ',' << csv(c.candidate_source_type) << ','
             << c.candidate_source_count << ',' << c.probe_sample_count << ','
             << c.probe_pass_count << ',' << c.source_pass_rate << ','
             << c.estimated_joint_count << ','
             << c.estimated_range_match_density << ','
             << c.estimated_exact_cost << ',' << c.estimated_graph_cost << ','
             << c.final_visit_budget << ',' << c.final_bridge_budget << ','
             << c.visited_nodes << ',' << c.expanded_match_nodes << ','
             << c.expanded_bridge_nodes << ',' << c.dormant_bridge_peak << ','
             << c.distance_computations << ',' << c.predicate_checks << ','
             << c.posting_lookup_time_ms << ',' << c.planning_time_ms << ','
             << c.execution_time_ms << ',' << results[q].search_time_ms << ','
             << (c.driver_type==rcfs::AttributeType::kNumeric ? "numeric" : "categorical") << ','
             << csv(c.driver_atom) << ',' << c.joint_count_lower << ',' << c.joint_count_upper << ','
             << c.probe_cache_hits << ',' << c.edge_visits << ',' << csv(c.empty_reason) << ','
             << c.required_graph_visits << ',' << c.budget_gap << ',' << c.estimated_expansions << ','
             << c.final_expand_budget << ',' << c.final_ef_match << ','
             << c.budget_limited << ',' << csv(c.quality_status) << ','
             << c.joint_count_unbiased << ',' << c.conservative_count_lower << ',' << c.conservative_count_upper << ','
             << c.observed_yield << ',' << c.result_gain_per_distance << ',' << c.quality_bucket << ',' << c.empirical_recall_lower << '\n';
    }
  }
}

} // namespace rcfs_cli
