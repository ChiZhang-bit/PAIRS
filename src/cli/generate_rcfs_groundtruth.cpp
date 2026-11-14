#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/cli_utils.h"
#include "rcfs/dataset_io.h"
#include "rcfs/index.h"
#include "rcfs/predicate.h"
#include "rcfs/searcher.h"

namespace {

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  generate_pairs_groundtruth --index-prefix PREFIX "
         "--queries QUERY.fvecs --filters filters.jsonl "
         "--output groundtruth.bin [--topk 100] [--max-queries N]\n";
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
    rcfs::QueryFiles files;
    files.vectors = args.require("queries");
    files.predicates = args.require("filters");
    files.vector_format = rcfs::parse_vector_file_format(
        args.get("vector-format", "fvecs"));
    rcfs::QueryWorkload workload =
        rcfs::read_query_workload(files, index.schema());
    const uint32_t maximum = args.get_u32("max-queries", 0);
    if (maximum != 0 && maximum < workload.vectors.rows) {
      workload.vectors.rows = maximum;
      workload.vectors.values.resize(
          static_cast<size_t>(maximum) * workload.vectors.cols);
      workload.predicates.resize(maximum);
    }

    rcfs::SearchParams params;
    params.k = args.get_u32("topk", 100);
    params.threads = args.get_u32("threads", 1);
    params.max_clauses = args.get_u32("max-clauses", 32);
    params.probe_budget = args.get_u32("probe-budget", 64);
    params.forced_plan = rcfs::PlanKind::kExactMeta;
    const auto predicates = rcfs::compile_predicate_records(
        workload.predicates, params.max_clauses);
    rcfs::RcfsSearcher searcher(index);
    const auto start = std::chrono::steady_clock::now();
    const auto results =
        searcher.batch_search(workload.vectors, predicates, params);

    rcfs::ResultMatrix output;
    output.rows = static_cast<uint32_t>(results.size());
    output.k = params.k;
    const size_t total = static_cast<size_t>(output.rows) * output.k;
    output.ids.assign(total, rcfs::kInvalidNodeId);
    output.distances.assign(total, std::numeric_limits<float>::infinity());
    for (uint32_t q = 0; q < output.rows; ++q) {
      for (size_t i = 0;
           i < std::min<size_t>(output.k, results[q].ids.size()); ++i) {
        output.ids[static_cast<size_t>(q) * output.k + i] = results[q].ids[i];
        output.distances[static_cast<size_t>(q) * output.k + i] =
            results[q].distances[i];
      }
    }
    const std::string output_path = args.require("output");
    rcfs::ensure_parent_directory(output_path);
    rcfs::write_rcfs_result(output_path, output);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    std::cout << "[pairs-groundtruth] queries=" << output.rows
              << " topk=" << output.k << " seconds=" << seconds
              << " output=" << output_path << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "generate_pairs_groundtruth: " << error.what() << std::endl;
    print_usage();
    return 1;
  }
}
