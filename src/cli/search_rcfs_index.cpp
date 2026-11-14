#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "cli/cli_utils.h"
#include "rcfs/dataset_io.h"
#include "rcfs/index.h"
#include "rcfs/predicate.h"
#include "rcfs/searcher.h"
#include "rcfs/planner.h"
#include "rcfs/quality.h"
#include "cli/search_support.h"
using namespace rcfs_cli;

namespace {

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  search_pairs_index --index-prefix PREFIX "
         "--queries sift_query.fvecs --filters filters.jsonl "
         "--output results.bin [options]\n\n"
      << "Options:\n"
      << "  --vector-format fvecs|bin       default: fvecs\n"
      << "  --topk N                        default: 10\n"
      << "  --max-clauses N                 default: 32\n"
      << "  --probe-budget N                default: 64\n"
      << "  --probe-strata N                default: 8\n"
      << "  --exact-scan-threshold N        default: 4096\n"
      << "  --entry-probe-budget N          default: 24\n"
      << "  --visit-budget N                default: 1024\n"
      << "  --ef-match N                    fixed navigation width max(topk,N), default: 64\n"
      << "  --expand-budget N               fixed row expansion limit, default: 1024\n"
      << "  --bridge-priority residual|distance  default: residual\n"
      << "  --quality-table FILE            versioned held-out recall evidence\n"
      << "  --quality-min-samples N         minimum independent queries per bucket, default: 20\n"
      << "  --disable-exact-guard           explicit best-effort ablation\n"
      << "  --target-recall X               0: best effort; 1: exact; other: quality table\n"
      << "  --max-bridge-budget N           default: 512\n"
      << "  --bridge-min-share X            default: 0.10\n"
      << "  --bridge-max-share X            default: 0.90\n"
      << "  --force-plan auto|exact_meta|graph_bridge\n"
      << "  --max-queries N                 optional prefix of workload\n"
      << "  --driver-policy all|numeric-only|smallest-domain\n"
      << "  --force-driver FIELD            require FIELD as driver in each clause\n"
      << "  --threads N                     default: 1\n"
      << "  --total-probe-budget N          per-clause total, default: 256\n"
      << "  --probe-seed N                  default: 2023\n"
      << "  --distance-cache-entries N      default: 65536; 0 disables caching\n"
      << "  --distance-cache-mode auto|fixed  auto shrinks single-clause caches\n"
      << "  --calibrate-costs               warmed, per-view hardware calibration\n"
      << "  --warmup N                      unmeasured query prefix, default: 0\n"
      << "  --stats FILE.csv                optional per-clause diagnostics\n"
      << "  --gt FILE --gt-format ivecs|rcfs_bin  optional Recall report\n";
}


}  // namespace

int main(int argc, char** argv) {
  try {
    rcfs::Arguments args(argc, argv);
    if (args.flag("help")) {
      print_usage();
      return 0;
    }

    rcfs::RcfsIndex index =
        rcfs::RcfsIndex::load(args.require("index-prefix"));
    rcfs::QueryFiles query_files;
    query_files.vectors = args.require("queries");
    query_files.predicates = args.require("filters");
    query_files.vector_format = rcfs::parse_vector_file_format(
        args.get("vector-format", "fvecs"));
    rcfs::QueryWorkload workload =
        rcfs::read_query_workload(query_files, index.schema());
    if (workload.vectors.cols != index.dim()) {
      throw std::runtime_error("Query dimension does not match index");
    }
    truncate_workload(&workload, args.get_u32("max-queries", 0));

    rcfs::SearchParams params = make_search_params(args, index.schema());
    if (args.flag("calibrate-costs")) {
      rcfs::calibrate_costs(index,params,args.get_u32("calibration-samples",256));
      if (args.has("predicate-cost")) params.predicate_cost=args.get_double("predicate-cost",1.0);
      if (args.has("distance-cost")) params.distance_cost=args.get_double("distance-cost",8.0);
      if (args.has("open-cost")) { params.open_cost=args.get_double("open-cost",1.0); params.view_open_costs.clear(); }
      if (args.has("edge-cost")) { params.edge_cost=args.get_double("edge-cost",0.25); params.view_edge_costs.clear(); }
      std::cout << "[pairs-calibration] units=ns predicate=" << params.predicate_cost
                << " distance=" << params.distance_cost << " open=" << params.open_cost << " edge=" << params.edge_cost << std::endl;
    }
    const auto fingerprint=rcfs::quality_index_fingerprint(args.require("index-prefix"));
    std::cout << "[pairs-quality] index_fingerprint=" << fingerprint
              << " config_fingerprint=" << rcfs::quality_config_fingerprint(params) << std::endl;
    if (args.has("quality-table")) {
      bool enabled=rcfs::load_quality_table(args.require("quality-table"),fingerprint,params);
      std::cout << "[pairs-quality] table=" << (enabled ? "enabled" : "disabled_fingerprint_mismatch") << std::endl;
    }
    const std::vector<rcfs::CompiledPredicate> predicates =
        rcfs::compile_predicate_records(workload.predicates,
                                        params.max_clauses);
    rcfs::RcfsSearcher searcher(index);
    // Clip warm-up to the evaluated prefix. These queries run again in the
    // measured batch, so warming up does not remove rows from the workload.
    const uint32_t warmup=std::min(args.get_u32("warmup",0),workload.vectors.rows);
    if (warmup) {
      rcfs::DenseFloatMatrix warm{warmup,workload.vectors.cols,{}};
      warm.values.assign(workload.vectors.values.begin(),workload.vectors.values.begin()+size_t(warmup)*warm.cols);
      std::vector<rcfs::CompiledPredicate> wp(predicates.begin(),predicates.begin()+warmup);
      (void)searcher.batch_search(warm,wp,params);
    }
    // Measure the query batch, excluding load, compilation, warm-up, and I/O.
    const auto start = std::chrono::steady_clock::now();
    const std::vector<rcfs::SearchResult> results =
        searcher.batch_search(workload.vectors, predicates, params);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();

    const rcfs::ResultMatrix packed = pack_results(results, params.k);
    const std::string output_path = args.require("output");
    rcfs::ensure_parent_directory(output_path);
    rcfs::write_rcfs_result(output_path, packed);
    if (args.has("stats")) {
      write_stats(args.require("stats"), results, index.schema());
    }

    uint64_t distances = 0;
    uint64_t predicates_checked = 0;
    uint64_t visited = 0;
    for (const auto& result : results) {
      distances += result.distance_computations;
      predicates_checked += result.predicate_checks;
      visited += result.visited_nodes;
    }
    if (results.empty()) throw std::invalid_argument("Empty query workload");
    const double nq = static_cast<double>(results.size());
    // Percentiles use per-query timers. avg_latency_ms below is batch wall
    // time divided by query count, which differs from mean latency in parallel.
    std::vector<double> latencies;
    for (const auto& r:results) latencies.push_back(r.search_time_ms);
    std::sort(latencies.begin(),latencies.end());
    auto percentile=[&](double f) { return latencies[std::min(latencies.size()-1,size_t(std::ceil(f*latencies.size()))-1)]; };
    std::cout << std::fixed << std::setprecision(4)
              << "[pairs-search] queries=" << results.size()
              << " QPS=" << (seconds > 0.0 ? nq / seconds : 0.0)
              << " avg_latency_ms=" << (seconds * 1000.0 / nq)
              << " P50_ms=" << percentile(0.50) << " P95_ms=" << percentile(0.95) << " P99_ms=" << percentile(0.99)
              << " avg_distances=" << (distances / nq)
              << " avg_predicate_checks=" << (predicates_checked / nq)
              << " avg_visited=" << (visited / nq) << std::endl;

    if (args.has("gt")) {
      rcfs::IdMatrix truth;
      const std::string format = args.get("gt-format", "ivecs");
      if (format == "ivecs") {
        truth = rcfs::read_ivecs(args.require("gt"));
      } else if (format == "rcfs_bin") {
        const rcfs::ResultMatrix stored =
            rcfs::read_rcfs_result(args.require("gt"));
        truth.rows = stored.rows;
        truth.cols = stored.k;
        truth.ids = stored.ids;
      } else {
        throw std::invalid_argument("Unknown --gt-format: " + format);
      }
      std::cout << "[pairs-search] Recall@"
                << std::min(packed.k, truth.cols) << '='
                << recall_at_k(packed, truth) << std::endl;
    }
    std::cout << "[pairs-search] results: " << output_path << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "search_pairs_index: " << error.what() << std::endl;
    print_usage();
    return 1;
  }
}
