#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "cli/cli_utils.h"
#include "rcfs/dataset_io.h"
#include "rcfs/index_builder.h"
#include "rcfs/schema.h"

namespace {

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  build_pairs_index --data sift_base.fvecs "
         "--base-attrs base_attrs.bin --index-prefix PREFIX [options]\n\n"
      << "Options:\n"
      << "  --vector-format fvecs|bin       default: fvecs\n"
      << "  --tag-mapping FILE              legacy v2/v3 dictionary (default: beside attrs); omit for v4\n"
      << "  --color-mapping FILE            legacy v1 mapping alias\n"
      << "  --canonical-attribute NAME      default: first filter field\n"
      << "  --canonical-range NAME          legacy numeric-only spelling\n"
      << "  --M N                           default: 16\n"
      << "  --ef-construction N             default: 400\n"
      << "  --ef-max N                      default: 2048\n"
      << "  --category-code-seed N          optional dictionary/code permutation test\n"
      << "  --random-seed N                 default: 2023\n"
      << "  --build-threads N               maximum concurrent view builds (default: 1)\n"
      << "  --normalize                     normalize vectors before indexing\n";
}

uint64_t index_storage_bytes(const std::string& prefix, uint32_t views) {
  uint64_t total = 0;
  const std::vector<std::string> shared = {
      prefix + ".meta", prefix + ".vectors.bin",
      prefix + ".attributes.bin"};
  for (const auto& path : shared) {
    total += static_cast<uint64_t>(std::filesystem::file_size(path));
  }
  for (uint32_t j = 0; j < views; ++j) {
    const std::string stem = prefix + ".attr_" + std::to_string(j);
    total += static_cast<uint64_t>(
        std::filesystem::file_size(stem + ".order.bin"));
    total += static_cast<uint64_t>(
        std::filesystem::file_size(stem + ".graph.bin"));
    total += static_cast<uint64_t>(
        std::filesystem::file_size(stem + ".postings.bin"));
  }
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    rcfs::Arguments args(argc, argv);
    if (args.flag("help")) {
      print_usage();
      return 0;
    }

    rcfs::DatasetFiles files;
    files.vectors = args.require("data");
    files.base_attributes = args.require("base-attrs");
    files.color_mapping = args.get("color-mapping", "");
    files.tag_mapping = args.get("tag-mapping", "");
    if (args.has("tag-mapping") && args.has("color-mapping"))
      throw std::invalid_argument("Specify only one mapping option");
    files.vector_format = rcfs::parse_vector_file_format(
        args.get("vector-format", "fvecs"));

    std::cout << "[pairs-build] reading vectors and attributes" << std::endl;
    rcfs::Dataset dataset = rcfs::read_sift_gist_dataset(files);
    if (args.has("category-code-seed")) {
      const uint32_t seed=args.get_u32("category-code-seed",0);
      for (size_t c=0;c<dataset.categorical_columns.size();++c) {
        auto& column=dataset.categorical_columns[c];
        std::vector<rcfs::ValueId> permutation(column.value_names.size());
        std::iota(permutation.begin(),permutation.end(),rcfs::ValueId{0});
        std::mt19937 rng(seed+static_cast<uint32_t>(c)*104729U);
        std::shuffle(permutation.begin(),permutation.end(),rng);
        auto old_names=column.value_names;
        for (size_t v=0;v<permutation.size();++v) column.value_names[permutation[v]]=old_names[v];
        for (auto& v:column.values) if (v!=rcfs::kMissingValue) v=permutation[v];
      }
    }
    const rcfs::Schema schema = rcfs::make_schema(dataset);

    rcfs::BuildParams params;
    params.M = args.get_u32("M", 16);
    params.ef_construction = args.get_u32("ef-construction", 400);
    params.ef_max = args.get_u32("ef-max", 2048);
    params.random_seed = args.get_u32("random-seed", 2023);
    params.normalize = args.flag("normalize");
    if (args.has("canonical-attribute") && args.has("canonical-range"))
      throw std::invalid_argument("Use only one canonical attribute option");
    params.canonical_full_range_attribute = args.has("canonical-attribute")
        ? schema.attribute_id(args.require("canonical-attribute"))
        : args.has("canonical-range") ? schema.unified_numeric(schema.numeric_id(args.require("canonical-range"))) : 0;

    const std::string prefix = args.require("index-prefix");
    const uint32_t build_threads = args.get_u32("build-threads", 1);
    if (!build_threads) throw std::invalid_argument("--build-threads must be positive");
    rcfs::ensure_prefix_parent(prefix);
    const auto start = std::chrono::steady_clock::now();
    rcfs::RcfsIndex index =
        rcfs::RcfsIndexBuilder::build(std::move(dataset), params, build_threads);
    const auto built = std::chrono::steady_clock::now();
    index.save(prefix);
    const auto saved = std::chrono::steady_clock::now();

    const double build_seconds =
        std::chrono::duration<double>(built - start).count();
    const double save_seconds =
        std::chrono::duration<double>(saved - built).count();
    const uint64_t bytes = index_storage_bytes(
        prefix, static_cast<uint32_t>(index.views().size()));
    std::cout << std::fixed << std::setprecision(3)
              << "[pairs-build] N=" << index.size() << " dim=" << index.dim()
              << " unified_views=" << index.views().size()
              << " build_threads_requested=" << build_threads
              << " view_workers=" << std::min<size_t>(build_threads, index.views().size())
              << " build_seconds=" << build_seconds
              << " save_seconds=" << save_seconds
              << " total_seconds=" << (build_seconds + save_seconds)
              << " storage_gb="
              << (static_cast<double>(bytes) / 1000000000.0) << '\n'
              << "[pairs-build] saved prefix: " << prefix << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "build_pairs_index: " << error.what() << std::endl;
    print_usage();
    return 1;
  }
}
