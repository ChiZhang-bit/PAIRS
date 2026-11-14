#include "rcfs/planner.h"
#include "rcfs/distance.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
namespace rcfs {
namespace {
constexpr size_t none = static_cast<size_t>(-1);
double elapsed(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
}
uint64_t random_next(uint64_t& s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z^(z>>30))*0xbf58476d1ce4e5b9ULL; z = (z^(z>>27))*0x94d049bb133111ebULL;
  return z^(z>>31);
}
void probe(const RcfsIndex& index, const CompiledClause& clause, ClausePlan& plan,
           uint32_t budget, const SearchParams& params, ProbeWorkspace& work,
           uint64_t& checks, uint64_t& hits) {
  auto& s = plan.probe;
  const uint64_t n=plan.source.size();
  const uint32_t samples=static_cast<uint32_t>(std::min<uint64_t>(n,budget));
  s.sample_count=samples; s.exact=(samples==n);
  uint64_t rng=params.probe_seed ^ (uint64_t(plan.source.view_attribute)<<32) ^ plan.clause_id;
  const uint32_t strata=std::min(params.probe_strata,samples);
  s.sampled_positions.reserve(samples); s.sampled_matches.reserve(samples);
  std::vector<uint64_t> offsets;
  // Weight each stratum by its population, including unequal and fully sampled strata.
  for (uint32_t h=0;h<strata;++h) {
    const uint64_t begin=n*h/strata, length=n*(h+1)/strata-begin;
    const uint32_t quota=std::min<uint64_t>(length,samples/strata+(h>=strata-samples%strata));
    offsets.clear(); uint32_t passed=0;
    for (uint32_t j=0;j<quota;++j) {
      // Floyd's sampling avoids retries even when nearly the whole stratum is sampled.
      const uint64_t end=length-quota+j;
      uint64_t offset=random_next(rng)%(end+1);
      if (std::find(offsets.begin(),offsets.end(),offset)!=offsets.end()) offset=end;
      offsets.push_back(offset);
      Position pos=ClausePlanner::source_position(index,plan.source,begin+offset);
      NodeId id=index.view(plan.source.view_attribute).order.node_at(pos);
      auto cached=work.matches.find(id);
      bool match;
      if (cached) { match=*cached!=0; ++hits; }
      else { match=matches_clause(index.dataset(),clause,id); work.matches.insert(id,match); ++checks; }
      s.sampled_positions.push_back(pos); s.sampled_matches.push_back(match);
      passed+=match; s.pass_count+=match;
    }
    const double rate=double(passed)/quota;
    s.joint_count_unbiased+=length*rate;
    if (quota==length) {
      s.estimated_joint_count+=passed;
      s.joint_count_lower+=passed; s.joint_count_upper+=passed;
      s.conservative_count_lower+=passed; s.conservative_count_upper+=passed;
    } else {
      s.estimated_joint_count+=length*(double(passed)+1)/(quota+2.0);
      constexpr double z=1.959963984540054, z2=z*z;
      double den=1+z2/quota, center=(rate+z2/(2*quota))/den;
      double half=z*std::sqrt(rate*(1-rate)/quota+z2/(4.0*quota*quota))/den;
      s.joint_count_lower+=length*std::max(0.0,center-half);
      s.joint_count_upper+=length*std::min(1.0,center+half);
      // The union bound covers sampling error across strata, not graph-search recall.
      double eps=std::sqrt(std::log(2.0*strata/0.05)/(2*quota));
      s.conservative_count_lower+=length*std::max(0.0,rate-eps);
      s.conservative_count_upper+=length*std::min(1.0,rate+eps);
    }
  }
  if (!samples) {
    s.estimated_joint_count=0.5*n;
    s.joint_count_upper=s.conservative_count_upper=double(n);
  }
  // For one atom or TRUE, the selected source is the full clause answer set.
  if (clause.numeric_ranges.size()+clause.categorical_equalities.size()<=1) {
    s.exact=true; s.estimated_joint_count=s.joint_count_unbiased=double(n);
    s.joint_count_lower=s.joint_count_upper=double(n);
    s.conservative_count_lower=s.conservative_count_upper=double(n);
  }
  s.source_pass_rate=n ? s.estimated_joint_count/n : 0;
}
void cost_plan(ClausePlan& plan, const SearchParams& p) {
  plan.probe.driver_match_density=plan.probe.range_match_density=
      std::min(1.0,plan.probe.estimated_joint_count/plan.bounds.size());
  plan.probe.density_lower=std::min(1.0,plan.probe.joint_count_lower/plan.bounds.size());
  // Preserve required work before clipping: clipping first would hide an infeasible plan.
  const double required=std::ceil((double(p.k)+p.supply_slack)/
      std::max(plan.probe.density_lower,p.density_epsilon));
  plan.required_graph_visits=static_cast<uint64_t>(std::min(required,double(UINT64_MAX/2)));
  const uint64_t capacity=std::min<uint64_t>(p.max_visit_budget,plan.bounds.size());
  plan.budget_gap=plan.required_graph_visits>capacity ? plan.required_graph_visits-capacity : 0;
  plan.estimated_graph_visits=std::min(capacity,plan.required_graph_visits);
  // Estimate row expansions using eight discoveries per row, with a navigation-width floor.
  plan.estimated_expansions=std::min<uint64_t>(plan.bounds.size(),
      std::max<uint64_t>(std::max(p.k,p.ef_match),plan.required_graph_visits/8+1));
  plan.exact_cost=plan.source.size()*p.predicate_cost+plan.probe.estimated_joint_count*p.distance_cost;
  const double open=p.view_open_costs.empty() ? p.open_cost : p.view_open_costs[plan.driver_attribute];
  const double expand=p.view_edge_costs.empty() ? p.edge_cost : p.view_edge_costs[plan.driver_attribute];
  plan.graph_cost=open+plan.required_graph_visits*(p.predicate_cost+p.distance_cost)+
      plan.estimated_expansions*expand;
  plan.quality_status=plan.budget_gap ? "predicted_budget_limited" : "uncalibrated_conservative";
  if (p.target_recall==1.0) { plan.plan=PlanKind::kExactMeta; plan.quality_status="deterministic_exact"; }
  else if (p.forced_plan!=PlanKind::kAuto) plan.plan=p.forced_plan;
  else if ((p.global_exact_guard && plan.source.size()<=p.exact_scan_threshold) || plan.budget_gap ||
           (p.target_recall>0)) {
    plan.plan=PlanKind::kExactMeta;
    plan.quality_status=p.target_recall>0 ? "quality_unknown_exact" : "global_exact_guard";
  } else plan.plan=plan.exact_cost<=plan.graph_cost ? PlanKind::kExactMeta : PlanKind::kGraphBridge;
}
}  // namespace
Position ClausePlanner::source_position(const RcfsIndex& index, const CandidateSource& s, uint64_t i) {
  if (i >= s.size()) throw std::out_of_range("Candidate source index");
  return s.kind == CandidateSourceKind::kRangeSlice ? Position(s.range.begin+i)
      : index.view(s.view_attribute).postings.at(s.posting,i);
}
ClausePlan ClausePlanner::choose(const RcfsIndex& index, const CompiledClause& clause,
    const SearchParams& p, uint32_t clause_id, ProbeWorkspace* workspace) {
  if (!p.k || !p.probe_budget || !p.total_probe_budget || !p.probe_strata || !p.max_clauses || !p.quality_min_samples ||
      !p.max_visit_budget || !p.threads || !p.max_expand_budget || !p.ef_match)
    throw std::invalid_argument("Search/probe/visit budgets, k, clauses and threads must be positive");
  if (!std::isfinite(p.target_recall) || p.target_recall<0 || p.target_recall>1)
    throw std::invalid_argument("target_recall must be in [0,1]");
  for (double x : {p.predicate_cost,p.distance_cost,p.edge_cost,p.open_cost,p.supply_slack})
    if (!std::isfinite(x) || x<0) throw std::invalid_argument("Invalid planner cost");
  if (!std::isfinite(p.density_epsilon) || p.density_epsilon<=0 ||
      !std::isfinite(p.bridge_min_share) || !std::isfinite(p.bridge_max_share) ||
      p.bridge_min_share<0 || p.bridge_max_share>1 || p.bridge_min_share>p.bridge_max_share)
    throw std::invalid_argument("Invalid density/bridge parameters");
  if (p.forced_driver != kInvalidAttribute && p.forced_driver >= index.views().size())
    throw std::out_of_range("Forced driver does not exist");
  for (const auto* values : {&p.view_open_costs,&p.view_edge_costs}) {
    if (!values->empty() && values->size()!=index.views().size()) throw std::invalid_argument("Calibration/view count mismatch");
    for (double value:*values) if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid calibrated cost");
  }
  ProbeWorkspace local;
  ProbeWorkspace& work = workspace ? *workspace : local;
  // Predicate outcomes belong to this clause, even when OR branches share object IDs.
  work.matches.reset(p.total_probe_budget);
  std::vector<ClausePlan> candidates;
  auto posting_start = std::chrono::steady_clock::now();
  bool proven_empty=false;
  std::string empty_reason;
  auto add = [&](AttributeId attr, size_t r, size_t c, PositionRange bounds) {
    ClausePlan plan;
    plan.clause_id=clause_id; plan.driver_attribute=attr;
    plan.driver_type=index.schema().attribute_type(attr);
    plan.driver_range_index=r; plan.driver_equality_index=c; plan.bounds=bounds;
    plan.source.view_attribute=attr; plan.source.range=bounds;
    plan.candidate_source_name="interval:"+index.schema().attribute_name(attr);
    plan.driver_atom=plan.candidate_source_name;
    if (r != none) {
      const auto& a=clause.numeric_ranges[r];
      std::ostringstream text; text << index.schema().attribute_name(attr) << " in [" << a.lower << ";" << a.upper << "]";
      plan.driver_atom=text.str();
    } else if (c != none) {
      const auto& a=clause.categorical_equalities[c];
      plan.driver_atom=index.schema().attribute_name(attr)+"="+index.schema().categorical_value_names[a.attribute][a.value];
    } else plan.driver_atom="true:"+index.schema().attribute_name(attr);
    if (bounds.empty()) { proven_empty=true; empty_reason="empty_driver_interval"; return; }
    const auto& view=index.view(attr);
    // Each required category intersection still contains every match of the clause.
    for (size_t i=0;i<clause.categorical_equalities.size();++i) {
      if (i==c) continue;
      const auto& a=clause.categorical_equalities[i];
      auto part=view.postings.slice(index.schema().category_code(a.attribute,a.value),bounds);
      if (part.empty()) { proven_empty=true; empty_reason="empty_posting_intersection"; return; }
      if (part.size()<plan.source.size()) {
        plan.source.kind=CandidateSourceKind::kPostingSlice; plan.source.posting=part;
        plan.candidate_source_name="posting:"+index.schema().categorical_names[a.attribute]+"="+index.schema().categorical_value_names[a.attribute][a.value];
      }
    }
    plan.valid=true; candidates.push_back(std::move(plan));
  };
  // A necessary atom can prove emptiness even when it is ineligible as a Driver.
  for (size_t i=0;i<clause.numeric_ranges.size();++i) {
    const auto& a=clause.numeric_ranges[i];
    AttributeId attr=index.schema().unified_numeric(a.attribute);
    if (std::isnan(a.lower) || std::isnan(a.upper)) throw std::invalid_argument("NaN predicate bounds");
    add(attr,i,none,index.view(attr).order.map_closed(a.lower,a.upper));
  }
  for (size_t i=0;i<clause.categorical_equalities.size();++i) {
    const auto& a=clause.categorical_equalities[i];
    (void)index.schema().category_code(a.attribute,a.value);
    AttributeId attr=index.schema().unified_categorical(a.attribute);
    add(attr,none,i,index.view(attr).order.map_equal(a.value));
  }
  if (proven_empty) {
    ClausePlan empty; empty.clause_id=clause_id; empty.empty_reason=empty_reason;
    empty.posting_lookup_time_ms=elapsed(posting_start); return empty;
  }
  auto eligible=[&](const ClausePlan& plan) {
    return p.forced_driver!=kInvalidAttribute ? plan.driver_attribute==p.forced_driver :
        p.driver_policy!=DriverPolicy::kNumericOnly || plan.driver_type==AttributeType::kNumeric;
  };
  if (std::none_of(candidates.begin(),candidates.end(),eligible)) {
    bool true_clause=clause.numeric_ranges.empty() && clause.categorical_equalities.empty();
    if (p.forced_driver!=kInvalidAttribute && !true_clause)
      throw std::invalid_argument("Forced driver is not an atom of this clause");
    AttributeId attr=p.forced_driver!=kInvalidAttribute ? p.forced_driver : index.build_params().canonical_full_range_attribute;
    if (p.driver_policy==DriverPolicy::kNumericOnly && p.forced_driver==kInvalidAttribute) {
      if (index.schema().numeric_names.empty()) throw std::invalid_argument("numeric-only needs a numeric view");
      attr=0;
    }
    // Tag-only queries still need a navigation domain when categorical Drivers are disabled.
    add(attr,none,none,{0,index.size()});
  }
  if (proven_empty) {
    ClausePlan empty; empty.clause_id=clause_id; empty.empty_reason=empty_reason; return empty;
  }
  const double posting_ms=elapsed(posting_start);
  // Driver restrictions do not restrict complete-source access. Keep the source
  // in its own view so categorical postings remain usable by every Driver.
  auto primary=std::min_element(candidates.begin(),candidates.end(),[](const ClausePlan& a,const ClausePlan& b) {
    return a.source.size()<b.source.size() ||
        (a.source.size()==b.source.size() && a.driver_attribute<b.driver_attribute);
  });
  uint64_t checks=0,hits=0;
  // One clause-level probe supplies the joint count used by all Driver densities.
  probe(index,clause,*primary,std::min(p.probe_budget,p.total_probe_budget),p,work,checks,hits);
  if (primary->probe.exact && primary->probe.estimated_joint_count==0) {
    primary->valid=false; primary->empty_reason="exact_probe_empty";
    primary->planning_predicate_checks=checks; primary->probe_cache_hits=hits;
    primary->posting_lookup_time_ms=posting_ms; return *primary;
  }
  const auto source=primary->source;
  auto stats=primary->probe;
  auto sampled_positions=std::move(stats.sampled_positions);
  auto sampled_matches=std::move(stats.sampled_matches);
  const auto source_name=primary->candidate_source_name;
  const bool rank_by_domain=p.driver_policy==DriverPolicy::kSmallestDomain && p.forced_driver==kInvalidAttribute;
  const bool rank_feasible_domains=rank_by_domain && p.forced_plan==PlanKind::kAuto && p.target_recall==0 &&
      !(p.global_exact_guard && source.size()<=p.exact_scan_threshold);
  ClausePlan best; double best_cost=std::numeric_limits<double>::infinity();
  for (auto& candidate:candidates) {
    if (!eligible(candidate)) continue;
    candidate.source=source; candidate.probe=stats; candidate.candidate_source_name=source_name;
    cost_plan(candidate,p);
    if (rank_feasible_domains) {
      // Rank feasible Drivers by domain size, then use the selected Driver's
      // comparison with EXACT. If none is feasible, cost_plan has selected EXACT.
      const bool feasible=candidate.budget_gap==0;
      const bool best_feasible=best.budget_gap==0;
      const bool smaller=candidate.bounds.size()<best.bounds.size() ||
          (candidate.bounds.size()==best.bounds.size() && candidate.driver_attribute<best.driver_attribute);
      if (!best.valid || (feasible && !best_feasible) || (feasible==best_feasible && smaller))
        best=std::move(candidate);
      continue;
    }
    double cost=rank_by_domain ? double(candidate.bounds.size()) :
        candidate.plan==PlanKind::kExactMeta ? candidate.exact_cost : candidate.graph_cost;
    if (!best.valid || cost<best_cost || (cost==best_cost && candidate.driver_attribute<best.driver_attribute)) {
      best_cost=cost; best=std::move(candidate);
    }
  }
  best.probe.sampled_positions=std::move(sampled_positions);
  best.probe.sampled_matches=std::move(sampled_matches);
  best.planning_predicate_checks=checks; best.probe_cache_hits=hits; best.posting_lookup_time_ms=posting_ms;
  return best;
}

void calibrate_costs(const RcfsIndex& index, SearchParams& params, uint32_t samples) {
  if (!samples) throw std::invalid_argument("Calibration sample count must be positive");
  const uint32_t n=std::min(samples,index.size());
  L2Distance distance(index.dim());
  volatile double sink=0;
  auto median=[](std::vector<double> v) { std::sort(v.begin(),v.end()); return std::max(0.001,v[v.size()/2]); };
  std::vector<double> distance_times,predicate_times;
  CompiledClause clause;
  if (!index.schema().numeric_names.empty())
    clause.numeric_ranges.push_back({0,index.view(0).order.min_key(),index.view(0).order.max_key()});
  else clause.categorical_equalities.push_back({0,0});
  for (unsigned trial=0;trial<6;++trial) {
    auto begin=std::chrono::steady_clock::now();
    for (uint32_t i=0;i<n;++i) sink+=distance(index.vector(i),index.vector((uint64_t(i)*97+1)%index.size()));
    double dt=elapsed(begin)*1e6/n;
    begin=std::chrono::steady_clock::now();
    for (uint32_t i=0;i<n;++i) sink+=matches_clause(index.dataset(),clause,i) ? 1.0 : 0.0;
    double pt=elapsed(begin)*1e6/n;
    if (trial) { distance_times.push_back(dt); predicate_times.push_back(pt); }
  }
  params.distance_cost=median(distance_times); params.predicate_cost=median(predicate_times);
  params.view_open_costs.clear(); params.view_edge_costs.clear();
  std::vector<Position> scratch;
  for (const auto& view:index.views()) {
    std::vector<double> opens,edges;
    for (unsigned trial=0;trial<6;++trial) {
      auto begin=std::chrono::steady_clock::now();
      for (uint32_t i=0;i<n;++i) sink+=view.routing.open({i,index.size()}).size();
      double open=elapsed(begin)*1e6/n;
      auto handle=view.routing.open({0,index.size()});
      begin=std::chrono::steady_clock::now();
      for (uint32_t i=0;i<n;++i) {
        view.routing.neighbors(handle,Position(uint64_t(i)*index.size()/n),scratch);
        if (!scratch.empty()) sink+=scratch.front();
      }
      // Charge the complete row operation once per expansion, regardless of degree.
      double edge=elapsed(begin)*1e6/n;
      if (trial) { opens.push_back(open); edges.push_back(edge); }
    }
    params.view_open_costs.push_back(median(opens)); params.view_edge_costs.push_back(median(edges));
  }
  params.open_cost=median(params.view_open_costs); params.edge_cost=median(params.view_edge_costs);
  (void)sink;
}
}  // namespace rcfs
