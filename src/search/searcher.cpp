#include "rcfs/searcher.h"
#include "rcfs/planner.h"
#include "rcfs/distance.h"
#include "rcfs/quality.h"
#include "rcfs/bridge_priority.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
namespace rcfs {
namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
struct Hit { float distance; NodeId node; };
bool better(const Hit& a,const Hit& b) { return a.distance<b.distance || (a.distance==b.distance && a.node<b.node); }
struct WorstFirst { bool operator()(const Hit& a,const Hit& b) const { return better(a,b); } };
struct Match { float distance; Position position; NodeId node; };
struct BestMatch { bool operator()(const Match& a,const Match& b) const {
  return better({b.distance,b.node},{a.distance,a.node});
}};
using detail::Bridge;
using detail::BestBridge;
template<class T,class C> struct Heap : std::priority_queue<T,std::vector<T>,C> {
  void reset(size_t capacity=0) { this->c.clear(); if (capacity>this->c.capacity()) this->c.reserve(capacity); }
  void comparator(C value) { this->c.clear(); this->comp=value; }
};
enum class State : uint8_t { MatchQueued=1,Dormant=2,Active=3,MatchExpanded=4,BridgeExpanded=5 };
struct Workspace {
  FlatQueryTable<float> distances;  // Original-ID keys allow reuse across OR clauses.
  FlatQueryTable<State> state;  // Driver positions, reset for each clause.
  ProbeWorkspace probe;
  // Results retain k matches; navigation retains the best w for expansion.
  Heap<Hit,WorstFirst> results,navigation;
  Heap<Match,BestMatch> matches;
  Heap<Bridge,BestBridge> active,dormant;
  std::vector<Position> neighbors,entries;
  std::vector<Match> seeds;
  std::vector<Hit> merged;
  std::vector<float> normalized;
  std::vector<double> inverse_spans;
};
struct Context {
  const RcfsIndex& index;
  const float* query;
  L2Distance distance_function;
  Workspace& work;
  uint64_t distances=0,checks=0;
  Context(const RcfsIndex& i,const float* q,Workspace& w):index(i),query(q),distance_function(i.dim()),work(w){}
  float distance(NodeId id) {
    if (auto found=work.distances.find(id)) return *found;
    float d=distance_function(query,index.vector(id));
    // A full cache skips insertion; the computed distance remains usable.
    work.distances.insert(id,d); ++distances; return d;
  }
};
void topk(Heap<Hit,WorstFirst>& heap,Hit h,uint32_t k) {
  if (heap.size()<k) heap.push(h);
  else if (better(h,heap.top())) { heap.pop(); heap.push(h); }
}
struct Evaluation { uint32_t violations=0; float gap=0; };
Evaluation evaluate(const RcfsIndex& index,const CompiledClause& clause,const ClausePlan& plan,
                    const std::vector<double>& inverse,NodeId id) {
  // The Driver interval already enforces its atom.
  Evaluation out;
  for (size_t i=0;i<clause.categorical_equalities.size();++i) {
    if (i==plan.driver_equality_index) continue;
    const auto& a=clause.categorical_equalities[i];
    if (index.dataset().categorical_columns[a.attribute].values[id]!=a.value) ++out.violations;
  }
  for (size_t i=0;i<clause.numeric_ranges.size();++i) {
    if (i==plan.driver_range_index) continue;
    const auto& a=clause.numeric_ranges[i];
    double v=index.dataset().numeric_columns[a.attribute].values[id];
    if (v<a.lower || v>a.upper) {
      ++out.violations;
      out.gap+=static_cast<float>((v<a.lower ? a.lower-v : v-a.upper)*inverse[i]);
    }
  }
  return out;
}
ClauseDiagnostics diagnostics_for(const ClausePlan& p) {
  ClauseDiagnostics d;
  d.clause_id=p.clause_id; d.driver_attribute=p.driver_attribute; d.driver_type=p.driver_type;
  d.driver_atom=p.driver_atom; d.empty_reason=p.empty_reason; d.plan=p.plan;
  d.driver_range_count=p.bounds.size(); d.candidate_source_type=p.candidate_source_name;
  d.candidate_source_count=p.source.size(); d.probe_sample_count=p.probe.sample_count;
  d.probe_pass_count=p.probe.pass_count; d.source_pass_rate=p.probe.source_pass_rate;
  d.estimated_joint_count=p.probe.estimated_joint_count;
  d.joint_count_lower=p.probe.joint_count_lower; d.joint_count_upper=p.probe.joint_count_upper;
  d.estimated_range_match_density=p.probe.driver_match_density;
  d.estimated_exact_cost=p.exact_cost; d.estimated_graph_cost=p.graph_cost;
  d.predicate_checks=p.planning_predicate_checks; d.probe_cache_hits=p.probe_cache_hits;
  d.posting_lookup_time_ms=p.posting_lookup_time_ms;
  d.required_graph_visits=p.required_graph_visits; d.budget_gap=p.budget_gap;
  d.estimated_expansions=p.estimated_expansions; d.quality_status=p.quality_status;
  d.joint_count_unbiased=p.probe.joint_count_unbiased;
  d.conservative_count_lower=p.probe.conservative_count_lower;
  d.conservative_count_upper=p.probe.conservative_count_upper;
  return d;
}
void exact_search(const CompiledClause& clause,const ClausePlan& plan,const SearchParams& params,
                  Context& ctx,ClauseDiagnostics& d) {
  const auto& view=ctx.index.view(plan.source.view_attribute);
  for (uint64_t i=0;i<plan.source.size();++i) {
    NodeId id=view.order.node_at(ClausePlanner::source_position(ctx.index,plan.source,i));
    auto known=ctx.work.probe.matches.find(id);
    bool match;
    if (known) { match=*known!=0; ++d.probe_cache_hits; }
    else { match=matches_clause(ctx.index.dataset(),clause,id); ++ctx.checks; }
    if (match) topk(ctx.work.results,{ctx.distance(id),id},params.k);
  }
  d.visited_nodes=plan.source.size();
}
void graph_search(const CompiledClause& clause,const ClausePlan& plan,const SearchParams& params,
                  Context& ctx,ClauseDiagnostics& d) {
  auto& w=ctx.work;
  const auto& index=ctx.index;
  const auto& view=index.view(plan.driver_attribute);
  auto handle=view.routing.open_predicate_view(plan.bounds);
  if (handle.empty()) return;
  const uint64_t visit_max=std::min<uint64_t>(params.max_visit_budget,plan.bounds.size());
  const uint64_t expand_max=std::min<uint64_t>(params.max_expand_budget,visit_max);
  const uint64_t bridge_max=std::min<uint64_t>(params.max_bridge_budget,expand_max);
  const uint32_t width=std::max(params.k,params.ef_match);
  w.state.reset(size_t(visit_max));
  w.active.comparator(BestBridge{params.bridge_priority});
  w.dormant.comparator(BestBridge{params.bridge_priority});
  w.matches.reset(size_t(visit_max)); w.active.reset(size_t(bridge_max)); w.dormant.reset(size_t(visit_max));
  w.navigation.reset(std::min<uint64_t>(width,visit_max));
  w.inverse_spans.resize(clause.numeric_ranges.size());
  for (size_t i=0;i<clause.numeric_ranges.size();++i) {
    const auto& o=index.view(clause.numeric_ranges[i].attribute).order;
    w.inverse_spans[i]=1.0/std::max(o.max_key()-o.min_key(),1e-12);
  }
  uint64_t matches_seen=0,expanded=0,bridges=0,gains=0;
  const uint64_t distance_begin=ctx.distances;
  auto discover=[&](Position pos) -> bool {
    if (!plan.bounds.contains(pos)) { ++d.range_pruned_neighbors; return true; }
    // Revisiting a neighbor does not consume the discovery budget.
    if (w.state.find(pos)) return true;
    if (w.state.size()>=visit_max) { d.budget_limited=true; return false; }
    NodeId id=view.order.node_at(pos);
    Evaluation e;
    auto known=w.probe.matches.find(id);
    if (!known || !*known) { e=evaluate(index,clause,plan,w.inverse_spans,id); ++ctx.checks; }
    else ++d.probe_cache_hits;
    float distance=ctx.distance(id);
    if (!e.violations) {
      w.state.insert(pos,State::MatchQueued); w.matches.push({distance,pos,id});
      Hit hit{distance,id};
      if (w.results.size()<params.k || better(hit,w.results.top())) ++gains;
      topk(w.results,hit,params.k); topk(w.navigation,hit,width);
      ++matches_seen;
    } else {
      w.state.insert(pos,State::Dormant); w.dormant.push({e.violations,e.gap,distance,pos,id});
      d.dormant_bridge_peak=std::max<uint64_t>(d.dormant_bridge_peak,w.dormant.size());
    }
    return true;
  };
  const uint32_t entry_limit=static_cast<uint32_t>(std::min<uint64_t>(visit_max,std::max(1U,params.entry_probe_budget)));
  w.entries.clear(); w.seeds.clear();
  auto add_entry=[&](Position pos) {
    if (w.entries.size()<entry_limit && plan.bounds.contains(pos) &&
        std::find(w.entries.begin(),w.entries.end(),pos)==w.entries.end()) w.entries.push_back(pos);
  };
  // Convert probe positions through original IDs before using the Driver order.
  const auto& source_order=index.view(plan.source.view_attribute).order;
  for (size_t i=0;i<plan.probe.sampled_positions.size();++i) if (plan.probe.sampled_matches[i]) {
    NodeId id=source_order.node_at(plan.probe.sampled_positions[i]);
    Position pos=view.order.position_of(id);
    if (plan.bounds.contains(pos)) w.seeds.push_back({ctx.distance(id),pos,id});
  }
  std::sort(w.seeds.begin(),w.seeds.end(),[](const Match& a,const Match& b) { return better({a.distance,a.node},{b.distance,b.node}); });
  for (const auto& seed:w.seeds) {
    if (w.entries.size()>=std::max(1U,entry_limit/2)) break;
    add_entry(seed.position);
  }
  const size_t partial_limit=w.entries.size()+std::max(1U,entry_limit/4);
  for (size_t i=0;i<clause.categorical_equalities.size() && w.entries.size()<partial_limit;++i) {
    if (i==plan.driver_equality_index) continue;
    const auto& a=clause.categorical_equalities[i];
    auto part=view.postings.slice(index.schema().category_code(a.attribute,a.value),plan.bounds);
    if (!part.empty()) add_entry(view.postings.at(part,part.size()/2));
  }
  for (Position pos:view.routing.entries(handle,entry_limit)) add_entry(pos);
  for (Position pos:plan.probe.sampled_positions) add_entry(view.order.position_of(source_order.node_at(pos)));
  for (Position pos:w.entries) if (!discover(pos)) break;
  const double share=std::clamp(1-plan.probe.driver_match_density,double(params.bridge_min_share),double(params.bridge_max_share));
  // Active Bridges reserve row allowance; only expansion spends it.
  auto promote=[&]() {
    while (!w.dormant.empty() && bridges+w.active.size()<bridge_max) {
      Bridge b=w.dormant.top(); w.dormant.pop();
      auto state=w.state.find(b.position);
      if (state && *state==State::Dormant) { *state=State::Active; w.active.push(b); }
    }
  };
  while (!d.budget_limited) {
    promote();
    const bool can_match=!w.matches.empty() &&
        (w.navigation.size()<width || !better(w.navigation.top(),{w.matches.top().distance,w.matches.top().node}));
    // With no Bridge slots left, eligible Matches can still expand.
    if (!can_match && w.active.empty()) {
      if (!w.dormant.empty()) d.budget_limited=true;
      break;
    }
    // Charge an expansion when a row starts, including an empty row.
    if (expanded>=expand_max) { d.budget_limited=true; break; }
    bool choose_bridge=!w.active.empty() && (!can_match || expanded==0 || double(bridges)/expanded<share);
    Position pos;
    if (choose_bridge) {
      Bridge b=w.active.top(); w.active.pop(); pos=b.position;
      auto st=w.state.find(pos); if (!st || *st!=State::Active) continue;
      *st=State::BridgeExpanded; ++bridges; ++d.expanded_bridge_nodes;
    } else {
      Match m=w.matches.top(); w.matches.pop(); pos=m.position;
      auto st=w.state.find(pos); if (!st || *st!=State::MatchQueued) continue;
      *st=State::MatchExpanded; ++d.expanded_match_nodes;
    }
    ++expanded;
    view.routing.neighbors(handle,pos,w.neighbors);
    for (Position neighbor:w.neighbors) {
      ++d.edge_visits;
      if (!discover(neighbor)) break;
    }
  }
  d.visited_nodes=w.state.size(); d.final_bridge_budget=bridge_max; d.final_visit_budget=visit_max;
  d.final_expand_budget=expand_max; d.final_ef_match=width;
  d.observed_yield=w.state.size() ? double(matches_seen)/w.state.size() : 0;
  d.result_gain_per_distance=ctx.distances>distance_begin ? double(gains)/(ctx.distances-distance_begin) : 0;
}
SearchResult search_impl(const RcfsIndex& index,const float* query,const CompiledPredicate& predicate,
                         const SearchParams& requested,Workspace& work,bool quality_config_valid) {
  auto start=Clock::now();
  SearchParams policy;
  const SearchParams* executing=&requested;
  const uint32_t bucket=quality_bucket(predicate);
  double quality_lower=-1;
  bool calibrated=false;
  if (requested.target_recall>0 && requested.target_recall<1) {
    // Use EXACT unless this query bucket has sufficient calibration evidence.
    policy=requested; policy.target_recall=1;
    if (quality_config_valid)
      for (const auto& r:requested.quality_records)
        if (r.bucket==bucket && r.samples>=requested.quality_min_samples && r.recall_lower>=requested.target_recall) {
          quality_lower=r.recall_lower; calibrated=true; policy.target_recall=0; break;
        }
    executing=&policy;
  }
  const SearchParams& params=*executing;
  const double policy_ms=milliseconds(start);
  if (!params.k || !params.threads || !params.max_clauses || !params.max_visit_budget ||
      !params.probe_budget || !params.total_probe_budget || !params.probe_strata ||
      !params.quality_min_samples || !params.max_expand_budget ||
      !params.ef_match || !std::isfinite(params.target_recall) ||
      params.target_recall<0 || params.target_recall>1)
    throw std::invalid_argument("Invalid search parameters");
  if (predicate.clauses.size()>params.max_clauses) throw std::invalid_argument("Compiled predicate exceeds max_clauses");
  for (uint32_t i=0;i<index.dim();++i) if (!std::isfinite(query[i])) throw std::invalid_argument("Non-finite query vector");
  if (index.build_params().normalize) {
    work.normalized.assign(query,query+index.dim());
    double norm=0; for (float v:work.normalized) norm+=double(v)*v;
    if (norm>0) { float inverse=static_cast<float>(1/std::sqrt(norm)); for (float& v:work.normalized) v*=inverse; }
    query=work.normalized.data();
  }
  const uint64_t cache_limit=params.auto_distance_cache && predicate.clauses.size()==1 ?
      std::min<uint64_t>(params.distance_cache_entries,uint64_t(params.max_visit_budget)+params.probe_budget) :
      params.distance_cache_entries;
  work.distances.reset(static_cast<size_t>(cache_limit)); work.merged.clear();
  Context ctx(index,query,work);
  SearchResult result; result.compiled_clause_count=static_cast<uint32_t>(predicate.clauses.size());
  if (predicate.clauses.empty()) {
    ClauseDiagnostics empty; empty.empty_reason="compiled_false";
    result.clauses.push_back(std::move(empty));
  }
  for (uint32_t i=0;i<predicate.clauses.size();++i) {
    auto planning=Clock::now();
    ClausePlan plan=ClausePlanner::choose(index,predicate.clauses[i],params,i,&work.probe);
    ClauseDiagnostics d=diagnostics_for(plan); d.planning_time_ms=milliseconds(planning)+(i==0 ? policy_ms : 0);
    d.quality_bucket=bucket; d.empirical_recall_lower=quality_lower;
    if (requested.target_recall>0 && requested.target_recall<1)
      d.quality_status=calibrated ? "empirical_bucket_covered" : "quality_unknown_exact";
    ctx.checks+=plan.planning_predicate_checks;
    if (plan.valid) {
      work.results.reset(std::min<uint64_t>(params.k,index.size()));
      auto execution=Clock::now(); uint64_t before_dist=ctx.distances,before_checks=ctx.checks;
      if (plan.plan==PlanKind::kExactMeta) {
        // A single EXACT pass cannot reuse distances between clauses.
        if (params.auto_distance_cache && predicate.clauses.size()==1) work.distances.reset(0);
        exact_search(predicate.clauses[i],plan,params,ctx,d);
      }
      else graph_search(predicate.clauses[i],plan,params,ctx,d);
      d.distance_computations=ctx.distances-before_dist; d.predicate_checks+=ctx.checks-before_checks;
      d.execution_time_ms=milliseconds(execution);
      while (!work.results.empty()) { work.merged.push_back(work.results.top()); work.results.pop(); }
    }
    result.visited_nodes+=d.visited_nodes; result.clauses.push_back(std::move(d));
  }
  // Clause results share original IDs, even when their Drivers use different orders.
  std::sort(work.merged.begin(),work.merged.end(),[](const Hit& a,const Hit& b) {
    return a.node<b.node || (a.node==b.node && a.distance<b.distance);
  });
  work.results.reset(std::min<uint64_t>(params.k,index.size()));
  NodeId last=kInvalidNodeId;
  for (const Hit& hit:work.merged) {
    if (hit.node==last) continue;
    last=hit.node;
    ++ctx.checks;
    if (matches_predicate(index.dataset(),predicate,hit.node)) topk(work.results,hit,params.k);
  }
  result.ids.resize(work.results.size()); result.distances.resize(work.results.size());
  for (size_t i=work.results.size();i>0;--i) {
    result.ids[i-1]=work.results.top().node; result.distances[i-1]=work.results.top().distance; work.results.pop();
  }
  result.distance_computations=ctx.distances; result.predicate_checks=ctx.checks;
  result.search_time_ms=milliseconds(start); return result;
}
}  // namespace
SearchResult RcfsSearcher::search(const std::vector<float>& query,const CompiledPredicate& predicate,const SearchParams& params) const {
  if (query.size()!=index_.dim()) throw std::invalid_argument("Query dimension mismatch");
  Workspace workspace;
  return search_impl(index_,query.data(),predicate,params,workspace,
      params.quality_config && params.quality_config==quality_config_fingerprint(params));
}
std::vector<SearchResult> RcfsSearcher::batch_search(const DenseFloatMatrix& queries,
    const std::vector<CompiledPredicate>& predicates,const SearchParams& params) const {
  if (queries.cols!=index_.dim() || queries.values.size()!=size_t(queries.rows)*queries.cols || predicates.size()!=queries.rows)
    throw std::invalid_argument("Query/predicate matrix mismatch");
  if (!params.threads) throw std::invalid_argument("threads must be positive");
  std::vector<SearchResult> out(queries.rows);
  if (!queries.rows) return out;
  (void)l2_kernel(index_.dim()); // finish SIMD dispatch before launching workers
  std::atomic<uint32_t> next{0}; std::atomic<bool> stop{false};
  std::exception_ptr error; std::mutex error_mutex;
  const bool quality_valid=params.quality_config && params.quality_config==quality_config_fingerprint(params);
  auto worker=[&]() {
    try {
      Workspace work;
      while (!stop.load(std::memory_order_relaxed)) {
        uint32_t row=next.fetch_add(1,std::memory_order_relaxed);
        if (row>=queries.rows) break;
        out[row]=search_impl(index_,queries.values.data()+size_t(row)*queries.cols,predicates[row],params,work,quality_valid);
      }
    } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error=std::current_exception(); stop=true; }
  };
  uint32_t count=std::min(params.threads,queries.rows);
  std::vector<std::thread> threads;
  try { for (uint32_t i=1;i<count;++i) threads.emplace_back(worker); }
  catch (...) { stop=true; for (auto& t:threads) t.join(); throw; }
  worker(); for (auto& t:threads) t.join();
  if (error) std::rethrow_exception(error);
  return out;
}
}  // namespace rcfs
