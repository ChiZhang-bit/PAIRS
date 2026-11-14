#include "rcfs/quality.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
namespace rcfs {
namespace {
uint64_t digest(const std::string& text) {
  uint64_t h=14695981039346656037ULL;
  for (unsigned char c:text) { h^=c; h*=1099511628211ULL; }
  return h;
}
}
uint32_t quality_bucket(const CompiledPredicate& predicate) {
  if (predicate.clauses.empty()) return 0;
  size_t atoms=0;
  for (const auto& c:predicate.clauses)
    atoms=std::max(atoms,c.numeric_ranges.size()+c.categorical_equalities.size());
  return (predicate.clauses.size()>1 ? 3U : 1U)+(atoms>1 ? 1U : 0U);
}
uint64_t quality_config_fingerprint(const SearchParams& p) {
  std::ostringstream s; s.imbue(std::locale::classic()); s<<std::hexfloat;
  // Bind quality records to the search policy as well as its parameter values.
  s<<"PAIRS-search-1.0 L2 "<<p.k<<' '<<p.max_clauses<<' '<<p.probe_budget<<' '<<p.total_probe_budget<<' '
   <<p.probe_seed<<' '<<p.probe_strata<<' '<<p.exact_scan_threshold<<' '<<p.global_exact_guard<<' '
   <<p.entry_probe_budget<<' '<<p.max_visit_budget<<' '<<p.max_expand_budget<<' '
   <<p.max_bridge_budget<<' '<<p.ef_match<<' '
   <<p.bridge_min_share<<' '<<p.bridge_max_share<<' '<<int(p.forced_plan)<<' '<<int(p.driver_policy)<<' '
   <<p.forced_driver<<' '<<p.predicate_cost<<' '<<p.distance_cost<<' '<<p.edge_cost<<' '<<p.open_cost<<' '
   <<p.supply_slack<<' '<<p.density_epsilon;
  for (double v:p.view_open_costs) s<<" o"<<v;
  for (double v:p.view_edge_costs) s<<" e"<<v;
  // Different Bridge orderings need separate calibration.
  if (p.bridge_priority!=BridgePriority::kResidual)
    s<<" bridge-priority "<<int(p.bridge_priority);
  return digest(s.str());
}
uint64_t quality_index_fingerprint(const std::string& prefix) {
  std::ifstream file(prefix+".meta",std::ios::binary);
  if (!file) throw std::runtime_error("Cannot fingerprint index meta");
  std::ostringstream s; s<<file.rdbuf(); return digest(s.str());
}
bool load_quality_table(const std::string& path,uint64_t fingerprint,SearchParams& p) {
  // Failed or mismatched loads must not leave previous records active.
  p.quality_records.clear(); p.quality_config=0;
  std::ifstream file(path); file.imbue(std::locale::classic());
  std::string magic,workload; uint32_t version=0,count=0; uint64_t index=0,config=0;
  if (!(file>>magic>>version>>index>>config>>workload>>count) || magic!="UDRCFS_QUALITY" || version!=1 ||
      count>5 || workload.empty()) throw std::runtime_error("Invalid quality sidecar header");
  std::vector<QualityRecord> records; bool seen[5]={};
  for (uint32_t i=0;i<count;++i) {
    QualityRecord r;
    if (!(file>>r.bucket>>r.samples>>r.recall_lower) || r.bucket>4 || seen[r.bucket] || !r.samples ||
        !std::isfinite(r.recall_lower) || r.recall_lower<0 || r.recall_lower>1)
      throw std::runtime_error("Invalid quality sidecar record");
    seen[r.bucket]=true; records.push_back(r);
  }
  file>>std::ws;
  if (!file.eof()) throw std::runtime_error("Trailing quality sidecar data");
  if (index!=fingerprint || config!=quality_config_fingerprint(p)) return false;
  p.quality_records=std::move(records); p.quality_config=config; return true;
}
}  // namespace rcfs
