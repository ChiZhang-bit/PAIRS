# PAIRS: Predicate-Agnostic Indexing and Retrieval System for Multi-Attribute Filtered ANNS

C++17 implementation of PAIRS for numerical ranges, categorical equalities, and their AND/OR combinations.

## Overview

Each attribute view maps a range or equality to a position interval. Segmented graph adjacency supports navigation inside that interval. Categorical postings support exact counting and candidate enumeration. Views share vectors and original attributes.

The query planner compiles the filter into bounded DNF, estimates joint cardinality, and chooses complete-source scanning or graph search for each clause. Graph search distinguishes qualifying Match nodes from Bridge nodes used for navigation. Match navigation width and the discovery, row-expansion, and Bridge-expansion limits remain fixed during each search.

Distances are squared L2, with original ID as the tie-breaker. EXACT returns the exact filtered Top-k. GRAPH returns the best qualifying objects discovered within its budget. The final merge deduplicates object IDs and checks the complete predicate.

See [paper alignment](docs/paper_alignment.md) for the section-to-code mapping.

## Benchmark datasets

The dataset download contains three prepared bundles. Paths below are relative to the code repository:

| Directory | Base objects | Queries | Dimension | Attributes | Local availability |
| --- | ---: | ---: | ---: | --- | --- |
| `../dataset/SIFT1M` | 1,000,000 | 10,000 | 128 | 3 numerical, 3 categorical | Complete |
| `../dataset/GIST1M` | 1,000,000 | 1,000 | 960 | 3 numerical, 3 categorical | Complete |
| `../dataset/Arxiv2M` | 2,000,000 | 10,000 | 384 | 2 numerical, 100 binary categorical | Complete |

The arXiv count above describes the prepared local snapshot. It differs from the 2,138,591 objects listed in the paper. Do not treat those versions as interchangeable. The separate Paper benchmark in the configuration has no local bundle.

SIFT and GIST use `popularity`, `price`, `quality_score`, `bool`, `color`, and `size`. Their fixed workloads contain two, three, four, and five atoms in proportions of 35%, 35%, 20%, and 10%.

arXiv uses submission-time ranks and 100 category-presence fields. Category presence is represented by `present` or `absent` in each single-valued categorical column. Its supplied filters use these stored time ranks.

## Data

**Dataset download: [Google Drive](https://drive.google.com/drive/folders/1IKqtCKaxAff45GVn5ttETMNme58e8KHH?usp=sharing).**

The data is distributed separately from this code repository. Place the downloaded `dataset` directory beside the checkout:

```text
parent/
├── github_submit_code/     # This repository
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   └── ...
└── dataset/                # Separate Google Drive download
    ├── README.md
    ├── SIFT1M/
    │   ├── sift_base.fvecs
    │   ├── sift_query.fvecs
    │   ├── base_attrs.bin
    │   ├── filters.jsonl
    │   └── filtered_groundtruth.ivecs
    ├── GIST1M/             # gist_base.fvecs and gist_query.fvecs
    └── Arxiv2M/            # arxiv_base.fvecs and arxiv_query.fvecs
```

Run commands from `github_submit_code` and pass `--data-root ../dataset`. For a different download location, replace `../dataset` with that directory. Each dataset contains exactly five files: base vectors, query vectors, `base_attrs.bin`, `filters.jsonl`, and `filtered_groundtruth.ivecs`. The URFATTR v4 attribute file embeds its categorical dictionaries, so no separate mapping file is needed.

Base-vector row numbers are original IDs. Attribute rows use the same IDs. Query vectors, filter records, and filtered-ground-truth rows must align. Numerical intervals include both endpoints. Equality predicates use dictionary labels.

```json
{"query_id":0,"predicate":{"AND":[{"price":[100,300]},{"OR":[{"color":"red"},{"bool":true}]}]}}
```

Use the supplied `filtered_groundtruth.ivecs` for evaluation of the fixed vectors and filters. It contains filtered Top-100 answers. Replacing vectors, changing row order, or editing predicates requires matching ground truth.

The download contains a short `dataset/README.md` and the three five-file dataset directories. [Data formats](docs/data_format.md) documents the embedded dictionary layout, predicate syntax, and compatibility with older attribute files. URFATTR input version 4 and the V4 index format are separate formats.

## Build

Requirements are CMake 3.20+, GCC or Clang with C++17 support, and Python 3.9+. The launch scripts use only the Python standard library. MinGW is supported on Windows.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Release builds use `-march=native`. Set `-DRCFS_ARCH=<architecture>` for a different target CPU. On Windows, use an ASCII source junction if the MinGW toolchain cannot handle the checkout path.

```powershell
New-Item -ItemType Junction -Path "$env:TEMP\pairs-source" -Target (Get-Location).Path
cmake -S "$env:TEMP\pairs-source" -B "$env:TEMP\pairs-build" -G "MinGW Makefiles"
cmake --build "$env:TEMP\pairs-build" --parallel 4
```

Pass that location as `--build-dir` when running experiments. The junction points to this checkout and does not duplicate its files.

To install libraries, headers, programs, and supporting files:

```bash
cmake --install build --prefix <installation-directory>
```

## Run from the command line

Use the benchmark CLI from the package root. For example, build a SIFT index and then evaluate it:

```bash
python scripts/run_benchmark.py --dataset sift1m --data-root ../dataset --stage build
python scripts/run_benchmark.py --dataset sift1m --data-root ../dataset --stage search --topk 10 --threads 1 --warmup 100 --repeats 3
```

Use `--build-dir`, `--data-root`, `--index-root`, and `--output-dir` to select other locations. The CLI's default data root is inside the repository, so keep `--data-root ../dataset` for the sibling layout above. [configs/paper.json](configs/paper.json) supplies the base parameters and dataset filenames. Command-line options override its settings. Run `python scripts/run_benchmark.py --data-root ../dataset --help` for the supported options, or see the [running reference](docs/run_configuration.md) for native-program examples and path rules.

The local dataset keys are `sift1m`, `gist1m`, and `arxiv2m`, mapped to the directories above. The default stage is `search`, which requires an existing index.

| Stage | Work performed |
| --- | --- |
| `build` | Build and save the V4 index |
| `groundtruth` | Scan the indexed data to generate exact filtered ground truth |
| `search` | Load the index and evaluate the configured workload |
| `all` | Build, generate ground truth, then search |

For a new dataset, run `build` first, then switch to `search`. The supplied filtered Top-100 files support evaluation at k up to 100. `search` gives the supplied truth file priority over generated caches. For a changed workload or k above 100, provide matching truth, or point the profile's `groundtruth` filename to an absent file and run `groundtruth` followed by `search` to use the validated cache. Missing inputs are reported before work starts.

The defaults are M=16, ef-construction=400, ef-max=2048, Match width 64, discovery/expansion/Bridge limits 1024/1024/512, one query thread, 100 warm-up queries, and three measured runs. Effective Match width is at least k. Exhausting the Bridge limit still permits eligible Match expansions.

The paper runner disables the optional small-source EXACT safeguard. Native search retains its compatibility default of 4096. These settings are distinguished in [paper alignment](docs/paper_alignment.md).

The three native programs can also be called directly:

```bash
./build/build_pairs_index --help
./build/search_pairs_index --help
./build/generate_pairs_groundtruth --help
```

On Windows, the executable filenames end in `.exe`. Direct native search performs one run and does not read `configs/paper.json`. Use the benchmark CLI for profile defaults, repeated measurements, and summary files.

For an optional configuration-only workflow, edit [configs/local.json](configs/local.json) and run `python scripts/run.py` without arguments. It calls the same benchmark workflow. The included configuration sets `paths.data` to `../dataset`. Relative paths in this local configuration are anchored to the package root.

## Variants and measurements

| Variant | Change |
| --- | --- |
| `full` | All attribute Drivers with residual-aware navigation |
| `numeric-only` | Numerical graph Drivers only |
| `smallest-domain` | Smallest feasible graph Driver, compared with EXACT |
| `no-bridge` | Bridge expansion limit set to zero |
| `match-width-k` | Match navigation width set to k |

Runs write logs, results, per-clause CSV diagnostics, and `summary.json` under `results/`. Summaries include Recall@k, QPS, P50/P95/P99, distance and predicate counts, planning time, and medians across complete runs. Build summaries record construction time and serialized component sizes.

`avg_latency_ms` is batch wall time divided by query count. With multiple workers, it describes throughput rather than individual latency. Percentiles are computed from individual queries within each run. Peak process memory must be measured separately.

Use identical data, filters, ground truth, k, and thread counts when comparing variants. This package does not contain a new full-dataset performance run.

## Project structure

```text
configs/                  Benchmark profile and optional local configuration
docs/                     Formats and paper alignment
include/rcfs/             Public C++ APIs
include/base_hnsw/        Temporary HNSW construction dependency
include/range_graph/      Segmented graph construction backend
src/index/                Attribute orders, postings, and routing graphs
src/query/                Predicate compilation and planning
src/search/               Exact and graph execution
src/io/                   Data and index I/O
src/cli/                  Native executable entry points
scripts/                  Benchmark CLI and optional configuration launcher
```

The internal namespace and V4 file identifiers retain the name `rcfs`. A V4 index contains shared metadata, vectors, and attributes plus per-view order, graph, and posting components. Keep all components with the same prefix together.

## Small-data check

```bash
python scripts/smoke_test.py --build-dir build --work-dir artifacts/smoke
```

This creates synthetic data and checks index construction, loading, EXACT, and GRAPH. It does not modify the prepared datasets.

## Baseline implementations

FAVOR, ACORN, UNG, SeRF + PostFilter, HNSW + PostFilter, and Result Set Filtering are maintained separately. The programs here run PAIRS and the variants listed above.
