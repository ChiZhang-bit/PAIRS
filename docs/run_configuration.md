# Running PAIRS

The benchmark CLI loads dataset filenames and base parameters from `configs/paper.json`. Run it from the package root:

```sh
python scripts/run_benchmark.py --dataset sift1m --data-root ../dataset --stage build
python scripts/run_benchmark.py --dataset sift1m --data-root ../dataset --stage search --topk 10 --threads 1 --warmup 100 --repeats 3
python scripts/run_benchmark.py --data-root ../dataset --help
```

Command-line options override the profile. For example, `--variant no-bridge --visit-budget 1024 --expand-budget 1024` selects the Bridge ablation, while `--max-queries 100 --repeats 1 --warmup 20` selects a shorter measurement. `--config` selects another profile. The CLI checks required inputs before calling the native programs and records each run's options, logs, and metrics.

The prepared datasets are stored beside the code directory, under `../dataset`. Pass `--data-root ../dataset` from the package root. The script's unchanged defaults are `build`, `dataset`, `artifacts`, and `results` under the package root. Override them with `--build-dir`, `--data-root`, `--index-root`, and `--output-dir`. Explicit relative command-line paths are resolved from the current working directory. Absolute paths are accepted.

## Dataset and stage

`--dataset` accepts `sift1m`, `gist1m`, `arxiv2m`, `paper`, or `demo`. Dataset filenames and directory names come from the selected profile, normally `configs/paper.json`. With the data root above, the first three use `../dataset/SIFT1M`, `../dataset/GIST1M`, and `../dataset/Arxiv2M`. The `paper` dataset is an additional paper benchmark whose files must be supplied separately.

| `--stage` | Work performed | Required inputs |
| --- | --- | --- |
| `build` | Construct and save an index | Base vectors and self-contained v4 attributes |
| `groundtruth` | Generate exact filtered answers | Existing index, query vectors, and filters |
| `search` | Run and measure retrieval | Existing index, query vectors, filters, and supplied or validated cached truth |
| `all` | Build, generate truth, and search | Base vectors, self-contained v4 attributes, query vectors, and filters |

Each prepared bundle contains five files: base and query vectors, `base_attrs.bin`, `filters.jsonl`, and `filtered_groundtruth.ivecs`. Its URFATTR v4 attribute file embeds the category dictionaries. No mapping file or mapping option is needed for these bundles. The V4 index format is unchanged.

The default stage is `search`, which requires an existing index. Use `--stage build` for the first construction and `--stage groundtruth` when the supplied `filtered_groundtruth.ivecs` is absent. Generated truth is stored beside the index and bound to the input hashes. A supplied truth file takes precedence in `search` and must match the workload and cover the requested k. To use newly generated truth after changing the workload, move the old supplied file out of the configured location before running `groundtruth` and then `search`.

For a small example:

```sh
python scripts/make_demo.py --output ../dataset/demo
python scripts/run_benchmark.py --dataset demo --data-root ../dataset --stage all --warmup 3 --repeats 1
```

This creates a five-file example with 192 objects and 12 queries. The prepared SIFT1M, GIST1M, and Arxiv2M bundles use the same input layout. arXiv contains 2,000,000 base objects and 10,000 queries, distinct from the paper's 2,138,591-object version. Indexes still need to be built before the `search` stage.

## Direct native programs

The following commands build an index, generate native Top-100 truth, and run one search with the paper's EXACT-safeguard setting. They do not load the benchmark profile or produce the runner's aggregated summary.

```sh
./build/build_pairs_index --data ../dataset/SIFT1M/sift_base.fvecs --base-attrs ../dataset/SIFT1M/base_attrs.bin --index-prefix artifacts/sift1m/index --M 16 --ef-construction 400 --ef-max 2048
./build/generate_pairs_groundtruth --index-prefix artifacts/sift1m/index --queries ../dataset/SIFT1M/sift_query.fvecs --filters ../dataset/SIFT1M/filters.jsonl --output artifacts/sift1m/native_truth.bin --topk 100 --threads 1
./build/search_pairs_index --index-prefix artifacts/sift1m/index --queries ../dataset/SIFT1M/sift_query.fvecs --filters ../dataset/SIFT1M/filters.jsonl --output results/sift1m/native_results.bin --stats results/sift1m/native_clauses.csv --gt artifacts/sift1m/native_truth.bin --gt-format rcfs_bin --topk 10 --threads 1 --warmup 100 --disable-exact-guard
```

For supplied truth, replace the search truth options with `--gt ../dataset/SIFT1M/filtered_groundtruth.ivecs --gt-format ivecs`. The directly generated binary above has no runner cache sidecar, so pass it explicitly to native search. All three programs accept `--help`. On Windows, use the corresponding `.exe` files. Installed programs are under `<prefix>/bin`.

Older URFATTR v2/v3 attribute files still require their external dictionary. The native builder accepts `--tag-mapping FILE` for those versions, or reads `tag_mapping.json` beside the attributes. Version 4 reads the embedded dictionary and rejects an explicit external mapping. See [data formats](data_format.md) for the layout and conversion command.

## Optional configuration launcher

For a workflow without per-run arguments, edit `configs/local.json` and run:

```sh
python scripts/run.py
```

This launcher accepts no arguments and calls the same benchmark workflow. The supplied `configs/local.json` sets `paths.data` to `"../dataset"` for the current layout. It validates the local configuration's keys and values before passing its settings to the runner. `dataset`, `stage`, and `variant` select the same options as their command-line equivalents.

All relative paths in `configs/local.json` are resolved against the package root containing `scripts/` and `configs/`, regardless of the current working directory. Absolute paths are also accepted.

| Setting | Shipped value | Meaning |
| --- | --- | --- |
| `profile` | `configs/paper.json` | Dataset layout and base build/search settings |
| `paths.build` | `build` | Directory containing compiled programs, or its parent with a `bin`, `Release`, or `Debug` subdirectory |
| `paths.data` | `../dataset` | Parent of the configured dataset directories |
| `paths.index` | `artifacts` | Indexes and generated truth under `<dataset>/` |
| `paths.output` | `results` | Logs, per-clause CSV, results, and run summaries |
| `build.threads` | 1 | Index construction threads |
| `search.topk` / `search.threads` | 10 / 1 | Result count and query worker count |
| `search.ef_match` | 64 | Match navigation width, with effective width at least k |
| `search.visit_budget` / `search.expand_budget` | 1024 / 1024 | Fixed discovery and row-expansion caps |
| `search.bridge_budget` | 512 | Bridge row cap, which may be zero |
| `search.force_plan` | `auto` | `auto`, `exact_meta`, or `graph_bridge` |
| `search.force_driver` | `null` | Optional field name or unified field ID as a string |
| `search.calibrate_costs` | `false` | Run the native cost calibration before search |
| `evaluation.warmup` / `evaluation.repeats` | 100 / 3 | Warm-up queries per run and measured repetitions |
| `evaluation.max_queries` | `null` | Optional positive query prefix length |

Unknown keys are rejected, including keys inside each section. Numeric settings require integers. `warmup` and `bridge_budget` may be zero. The other numeric overrides must be positive. Omitted parameter overrides inherit the profile values.

After `cmake --install`, the package root is `<prefix>/share/pairs`. Set `paths.build` to `../../bin` to use the installed programs, and set the data, index, and result paths to suitable locations.

## Ablations and output

`--variant` accepts `full`, `numeric-only`, `smallest-domain`, `no-bridge`, or `match-width-k`. The optional local configuration uses the `variant` key. Variant settings are applied after the search overrides. For example, `no-bridge` always sets the Bridge cap to zero, and `match-width-k` sets the Match width to k. Their paper definitions are listed in [paper alignment](paper_alignment.md).

Each benchmark invocation records effective native options and metrics in a separate `summary.json` under the result directory. Missing-input and local-configuration errors are reported before result directories are created. Native execution failures retain their log and a summary with `complete=false`.
