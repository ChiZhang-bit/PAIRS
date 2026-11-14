# Paper-to-code alignment

This document maps the PAIRS algorithms to their implementation and lists the settings used by the benchmark runner. The C++ namespace and V4 file identifiers use the original `rcfs` name.

## Method mapping

| Paper section or algorithm | Implementation | Responsibility |
| --- | --- | --- |
| III.B.1, Attribute Ordering and Interval Representation | `src/index/attribute_order.cpp`, `include/rcfs/attribute_order.h` | Stable numerical order, categorical canonical ranks, bidirectional original-ID mappings, and atom-to-interval access |
| III.B.2, Segmented Routing Graph Construction | `include/range_graph/range_segment_graph_backend.h`, `src/index/range_routing_index.cpp` | Temporary HNSW candidate discovery, segmented forward blocks, cumulative reverse adjacency, and interval neighbor access |
| III.B.3, Categorical Postings Across Attribute Orders | `src/index/posting_index.cpp` | Sorted cross-view postings, implicit own-category intervals, exact slicing, and enumeration |
| Algorithm 1, Offline Stage | `src/index/index_builder.cpp`, `src/io/index_io.cpp`, `src/cli/build_rcfs_index.cpp` | View construction, shared storage, validation, and V4 persistence |
| III.C.1, Candidate Source Construction | `src/query/predicate_compiler.cpp`, `src/query/planner.cpp` | Bounded DNF, atomic domains, and complete sources carrying their own order handles |
| III.C.2, Joint Cardinality Estimation | `src/query/planner.cpp` | One shared clause-level stratified probe, smoothed joint counts, and density estimates |
| III.C.3, Execution Plan Selection | `src/query/planner.cpp` | Uncapped estimated discovery requirement, per-Driver budget gap, and EXACT/GRAPH cost comparison |
| III.D.1, Residual-Aware Navigation | `src/search/searcher.cpp`, `include/rcfs/bridge_priority.h` | Match/Bridge states, residual violation and numerical-gap priority, and initial entries |
| III.D.2, Budgeted Search Management | `src/search/searcher.cpp`, `include/rcfs/types.h` | Separate result and Match navigation heaps, fixed limits, and density-guided Bridge scheduling |
| Algorithm 2, Online Query Processing | `src/search/searcher.cpp`, `src/cli/search_rcfs_index.cpp` | Clause execution, shared bounded distance cache, original-ID deduplication, and full-predicate verification |
| V.A, Experimental Setup | `configs/paper.json`, `configs/local.json`, `scripts/run.py`, `scripts/run_benchmark.py` | Dataset specifications, parameter configuration, repeated runs, and measured outputs |

The graph backend uses segmented construction and temporary HNSW candidate discovery. Upstream attribution remains in the graph and HNSW source headers. Postings provide exact set access alongside the graph.

## Search behavior

The implementation uses squared L2 and a common `(distance, original ID)` result order. A complete source contains every object satisfying its clause, but may contain additional objects. EXACT scans the source and checks the complete clause. GRAPH fixes one Driver and checks the other predicates as residual conditions.

The numerical gap uses the field's full stored range. Categorical mismatches contribute a violation count, without a distance between category ranks. Bridge objects cannot enter the result heap.

The Match navigation width is `max(k, ef_match)`. For a Driver domain D, the fixed effective limits are:

```text
H     = min(max_visit_budget, |D|)
E_cap = min(max_expand_budget, H)
B_cap = min(max_bridge_budget, E_cap)
```

These limits correspond to Eq. (15). A previously unseen object blocked by H terminates graph discovery. E_cap prevents opening another adjacency row. B_cap prevents only further Bridge expansion, so eligible Match nodes can still expand. A filled result heap does not by itself end navigation.

Search follows the fixed width and budgets in Section III.D and Eqs. (14)–(16). The paper also uses “resumable search” in its overview and Figure 2. Here, all probe, posting, and coverage entries are chosen at initialization, and one Driver is used until termination. Budgets stay fixed, with no paused-row recovery or runtime switch to EXACT.

## Parameters

| Parameter | Release value | Source |
| --- | ---: | --- |
| M | 16 | V.A.4 |
| ef_construction | 400 | V.A.4, paper efc |
| ef_max | 2048 | V.A.4, paper efmax |
| ef_match | 64 | V.A.4, paper ef0 |
| max_visit_budget / max_expand_budget | 1024 / 1024 | V.A.4, Hmax / Emax |
| max_bridge_budget | 512 | V.A.4, Bmax |
| Warm-up queries / measured repetitions | 100 / 3 | V.A.4 |
| Result count | 10 by default | Primary Recall@10 experiments, with k = 1, 50, 100 also evaluated in V.B.2 |
| Small-source EXACT safeguard | Disabled by the paper runner | Algorithm 2 uses feasible-plan cost comparison, native CLI retains an optional threshold of 4096 |
| Main probe / total probe limit / strata | 64 / 256 / 8 | Implementation defaults |
| Maximum compiled DNF clauses | 32 | Implementation default |
| Entry budget / bounded distance cache | 24 / 65536 | Implementation defaults |
| Construction and probe seed | 2023 | Implementation defaults |
| Query threads | 1 by default | Launcher default |

The main probe precedes plan selection. A positive budget gap excludes that GRAPH candidate from feasibility, while other Drivers may still compete. An EXACT/GRAPH cost tie selects EXACT. The paper runner passes `--disable-exact-guard` to follow Algorithm 2's feasible-plan cost comparison. Direct native search retains the optional 4096-object safeguard unless this flag is supplied. When enabled, that safeguard applies after probing and covers the entire clause. Cost coefficients can be calibrated on the target machine through the native `--calibrate-costs` option.

The discovery estimate uses supply slack xi = 4 and density floor epsilon = 1e-6. Row expansions are estimated as `min(|D|, max(max(k, ef_match), U_req / 8 + 1))`, using eight new discoveries per row as a heuristic. These implementation defaults fill in values left unspecified by the paper's cost equations. The row-expansion cost includes block lookup, interval clipping, and merging.

The native CLI also supports an empirical quality table. The paper runner leaves it disabled (`target_recall=0`). Use `--target-recall 1` for EXACT on every clause. Intermediate targets use calibration statistics and provide no per-query Recall guarantee.

## Ablations

| Runner variant | Relationship to paper V.D |
| --- | --- |
| `full` | Full PAIRS configuration |
| `numeric-only` | w/o Cat. driver, categorical postings remain available for complete-source access |
| `smallest-domain` | w/o Cost ranking, choose the smallest feasible graph Driver and compare its cost with EXACT |
| `no-bridge` | w/o Bridge, retain discovery but prohibit Bridge row expansion |
| `match-width-k` | w/o Extra width, effective navigation width equals k |

The native CLI has an additional distance-only Bridge-priority option for experiments outside these four ablations. To compare against FAVOR, ACORN, UNG, SeRF, HNSW, or RSF, use their separate implementations with the same vectors, filters, and ground truth.

## Running experiments

Use `python scripts/run_benchmark.py --dataset sift1m --data-root ../dataset --stage search` to evaluate an existing index. The CLI loads `configs/paper.json` and accepts overrides for paths, budgets, repetitions, and ablations. An optional `scripts/run.py` launcher reads `configs/local.json` instead of per-run arguments. Its supplied configuration sets `paths.data` to `"../dataset"`. See [running PAIRS](run_configuration.md) for both interfaces and direct native-program examples. Use `demo` to check file formats and retrieval correctness before running the prepared benchmark bundles for performance measurements.

The prepared data root is `../dataset`, beside the code directory, with bundles under `SIFT1M`, `GIST1M`, and `Arxiv2M`. The CLI examples pass this location explicitly because the script still defaults to a `dataset` directory inside the code root. The arXiv bundle contains 2,000,000 base objects, while Table II reports 2,138,591 objects. These counts are separate fields in `configs/paper.json`.

Each bundle contains only base vectors, query vectors, `base_attrs.bin`, `filters.jsonl`, and `filtered_groundtruth.ivecs`. URFATTR v4 embeds categorical dictionaries in the attribute input. This packaging change preserves the attribute columns, labels, vectors, filters, and supplied ground truth. It does not change the graph construction or search algorithms, and the V4 index serialization remains unchanged. Older v2/v3 attribute inputs with external dictionaries remain supported.

Dataset download instructions are in the [repository README](../README.md). Keep the base ordering, filters, and dictionary labels unchanged when reusing ground truth.

The default protocol is three complete runs, each after 100 warm-up queries. Results are saved under `results/<dataset>/<variant>/<configuration-signature>/<UTC-timestamp>/`. Record the hardware, compiler, thread count, workload, and parameter overrides alongside comparisons. Build logs report construction and serialization times separately. Index bytes measure serialized storage, while peak process memory requires a separate measurement.

Generated ground truth is cached under `artifacts/<dataset>/groundtruth_k<K>_n<all-or-limit>.bin`. Its JSON sidecar records k, the query limit, and hashes of the inputs and result file. Search prefers a supplied `filtered_groundtruth.ivecs`, then falls back to this cache after validation. See [data formats](data_format.md) for the layouts and validation rules.
