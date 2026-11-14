# PAIRS data and artifact formats

The benchmark launcher uses one directory per dataset. Binary I/O requires a little-endian host. Row numbers identify objects and queries, so row order must agree across files.

The directories and vector filenames are configured in `configs/paper.json`. The prepared bundles now sit beside the code directory. From the code root, pass `--data-root ../dataset` to the benchmark CLI:

| Dataset key | Directory | Base vectors | Query vectors |
| --- | --- | --- | --- |
| `sift1m` | `../dataset/SIFT1M/` | `sift_base.fvecs` | `sift_query.fvecs` |
| `gist1m` | `../dataset/GIST1M/` | `gist_base.fvecs` | `gist_query.fvecs` |
| `arxiv2m` | `../dataset/Arxiv2M/` | `arxiv_base.fvecs` | `arxiv_query.fvecs` |
| `demo` | `../dataset/demo/` | `base.fvecs` | `query.fvecs` |

Alongside the two vector files, each directory contains these three files:

```text
  base_attrs.bin
  filters.jsonl
  filtered_groundtruth.ivecs
```

These five files are self-contained. Categorical dictionaries are embedded in `base_attrs.bin`. `python scripts/make_demo.py --output ../dataset/demo` creates a synthetic example using the same five-file layout. The formats are implemented in `src/io/dataset_io.cpp` and `src/io/predicate_io.cpp`. See [running PAIRS](run_configuration.md) for the benchmark CLI, native programs, and optional configuration launcher.

The arXiv bundle contains 2,000,000 base objects, 10,000 queries, and filtered Top-100 ground truth. The paper's 2,138,591-object dataset is a different size, recorded separately as `paper_base_count`. The native loaders validate the files supplied at runtime.

## Vector files

Each `.fvecs` record contains an `int32` dimension followed by that many `float32` values. Every record has the same positive dimension. The total file length must be a multiple of `4 * (dimension + 1)`. Base and query dimensions must agree.

Base vector row `i` is original ID `i`. Attribute row `i` refers to that same vector. Query-vector row `j` is paired with predicate record `j`. The index stores a shared vector matrix and converts between original IDs and attribute-local positions internally.

The native programs also accept `--vector-format bin`. That format contains `uint32 rows`, `uint32 dimensions`, and a row-major `float32[rows * dimensions]` payload. The portable benchmark layout uses `.fvecs`.

The distance implementation uses squared L2. Optional `--normalize` on the native builder normalizes base vectors and causes queries to receive the same transformation. This option changes the distance setting and must be held fixed across index construction, truth generation, and evaluation.

## Attribute files

The prepared bundles use **URFATTR version 4**, which stores attribute columns and categorical dictionaries in one file. Versions 2 and 3 remain readable with an external dictionary. The input-container version is independent of the V4 index serialization version.

### Header

| Offset | Type | Value |
| ---: | --- | --- |
| 0 | 8 bytes | ASCII `URFATTR` followed by a zero byte |
| 8 | `uint32` | Input version, 4 for the prepared bundles |
| 12 | `uint32` | Header size, `32 + 24 * (numeric_count + categorical_count)` |
| 16 | `uint64` | Number of base rows |
| 24 | `uint32` | Number of numerical fields |
| 28 | `uint32` | Number of categorical fields |
| 32 | Field descriptors | Numerical descriptors first, then categorical descriptors |

The file loader requires at least one numerical and one categorical field, a positive row count fitting `uint32`, and at most 4096 fields. Purely categorical datasets can be constructed through the C++ index API.

### Field descriptors

Each descriptor occupies 24 bytes:

| Bytes | Type | Meaning |
| --- | --- | --- |
| 0–15 | `char[16]` | Unique nonempty ASCII field name, null-terminated and zero-padded |
| 16 | `uint8` | Kind, 1 for numerical and 2 for categorical |
| 17 | `uint8` | Storage type from the table below |
| 18–19 | `uint16` | Reserved, must be zero |
| 20–23 | `uint32` | Dictionary size for categorical fields, zero for numerical fields |

| Type code | Storage | Applicable kind | Constraints |
| ---: | --- | --- | --- |
| 1 | `float32` | Numerical | All values finite |
| 2 | `int32` | Numerical | Values within [-16777216, 16777216], converted exactly to float32 |
| 3 | `uint8` | Categorical | Dictionary size from 1 to 256 |
| 4 | `uint16` | Categorical | Versions 3 and 4, dictionary size from 1 to 65536 |

The column payload is **column-major** in descriptor order. Write all rows of the first field, then all rows of the second field, and so on. Its end offset is `P = header_bytes + rows * sum(field_storage_bytes)`.

Categorical codes must lie in `[0, dictionary_size)`. The file format has no missing-category code, although the in-memory C++ model supports missing values.

### Embedded categorical dictionaries

Version 4 appends the dictionary immediately after the column payload:

| Offset | Type | Contents |
| --- | --- | --- |
| `P` | `uint32`, little-endian | JSON length in bytes, `J` |
| `P + 4` | `J` bytes | UTF-8 JSON dictionary |

The complete file length must be exactly `P + 4 + J`. Missing dictionary bytes and trailing bytes are rejected. The JSON contains exactly one dictionary for every categorical field. Codes are contiguous decimal keys starting at zero. Labels are nonempty and unique within a field, and dictionary sizes must match the descriptors.

```json
{
  "bool": {"0": "false", "1": "true"},
  "color": {"0": "red", "1": "blue", "2": "green"}
}
```

Equality queries use labels. The index assigns canonical ranks by label byte order for storage and interval lookup. For multiple categories per object, represent each membership as a Boolean categorical field, as in arXiv-2M. Each field stores one value per object.

The native builder reads this embedded dictionary automatically. Do not pass `--tag-mapping` with a version-4 file, since an explicit external dictionary is rejected.

### Older input formats

Version 2 supports uint8 categorical columns. Version 3 also supports uint16 columns. Both use the same header, descriptors, and column layout above, but end at offset `P` without an embedded dictionary. They require a matching external `tag_mapping.json`, either beside the attribute file or supplied through `--tag-mapping`. A legacy version-1 loader also remains available for the fixed `range1`, `range2`, `bool`, and `color` schema.

Convert an existing version-2 or version-3 file from the code root with:

```sh
python scripts/embed_attribute_mapping.py --attributes ../dataset/custom/base_attrs.bin --mapping ../dataset/custom/tag_mapping.json
```

The converter preserves the descriptors and column payload, changes the input version to 4, and appends the dictionary. It replaces the attribute file atomically by default. Use `--output NEWFILE` to write a separate converted file. The converter leaves the original mapping file in place. The prepared SIFT1M, GIST1M, and Arxiv2M bundles already use version 4.

## Predicate JSONL

Each nonempty line is one JSON object containing `query_id` and `predicate`. `query_id` is zero-based and must match the row of `query.fvecs`. `local_query_id` may be used instead. If both identifiers are present, both must match the row.

```json
{"query_id":0,"predicate":{"price":[100,300]}}
{"query_id":1,"predicate":{"color":"red"}}
{"query_id":2,"predicate":{"AND":[{"price":[100,300]},{"color":"red"}]}}
{"query_id":3,"predicate":{"OR":[{"color":"blue"},{"bool":true}]}}
```

Each predicate node is an object with exactly one key:

- A numerical field maps to two finite numbers `[lower, upper]` with `lower <= upper`. Both endpoints are included. Boundaries are retained as double precision for comparison with stored float32 keys.
- A categorical field maps to a dictionary label string or Boolean. Booleans normally resolve to the labels `"true"` and `"false"`.
- `AND` or `OR` maps to an array containing at least two child nodes. Nested forms support conjunctions, disjunctions, DNF, and bounded CNF-to-DNF conversion.

`NOT`, arbitrary category-code numbers, and set-membership arrays are not accepted. TRUE and FALSE are internal compiled states, not standalone JSONL literals. Repeated or contradictory conditions can be simplified by the compiler. The default maximum is 32 compiled DNF clauses.

Optional record metadata includes `source_query_id` and a string `template`. The parser checks the number of predicate records against the number of query vectors before a query-prefix limit is applied. Preserve the same base IDs, query order, dictionaries, filters, and distance configuration when reusing ground truth.

## Ground truth and search results

### Standard `.ivecs` ground truth

Each record contains an `int32 k` followed by `k` signed `int32` original IDs. All rows must use the same positive k. The sentinel `-1` pads queries with fewer than k qualifying objects. Other negative IDs are rejected.

Ground truth must come from exhaustive evaluation of the **complete filter** and distance ranking, with original ID as the tie-breaker. The number of rows must cover all evaluated query rows. For Recall@k, provide at least k ground-truth columns.

### Native result binary

`search_pairs_index` and `generate_pairs_groundtruth` use this format:

| Order | Type | Contents |
| --- | --- | --- |
| 1 | `uint32` | Number of query rows |
| 2 | `uint32` | Requested result count k |
| 3 | `uint32[rows * k]` | Row-major original IDs |
| 4 | `float32[rows * k]` | Row-major squared distances |

The complete IDs array comes before the distances array. Missing entries use ID `4294967295` and positive infinity. Read this format with `--gt-format rcfs_bin`, including files produced by the native ground-truth generator.

The native search executable accepts either format through `--gt FILE --gt-format ivecs|rcfs_bin`. Recall divides the number of retrieved ground-truth IDs by the number of valid ground-truth entries up to k. A query with no qualifying truth entries receives recall 1 only when no valid result is returned.

### Generated ground-truth identity

The runner's `groundtruth` and `all` stages write native truth beside the index:

```text
artifacts/<dataset>/
  groundtruth_k10_nall.bin
  groundtruth_k10_nall.json
```

The name contains the requested k and either `all` or the explicit `--max-queries` limit. For example, k = 50 with `--max-queries 100` uses `groundtruth_k50_n100.bin`. The JSON sidecar contains an `identity` object with `topk`, `max_queries`, and SHA-256 digests named `index_meta_sha256`, `queries_sha256`, and `filters_sha256`. Its top-level `sha256` field is the digest of the generated truth binary. Query and filter digests cover their complete input files, even when a prefix is evaluated.

A standalone `search` stage prefers `filtered_groundtruth.ivecs` in the configured dataset directory. If that file is absent, the runner looks for the generated binary and its sidecar for the requested k and limit. The identity must match the current inputs and the binary checksum must match the sidecar. A mismatch rejects reuse. Run the `groundtruth` stage again to replace stale or corrupt generated truth. The `all` stage generates fresh native truth and uses it in the same invocation.

Supplied `.ivecs` files have no input-identity sidecar. Prepare them using the matching base IDs, filters, and distance setting. The run summary records their checksum.

## Run outputs

Each invocation writes a separate directory:

```text
results/<dataset>/<variant>/<configuration-signature>/<UTC-timestamp>/
  summary.json
  build.log                 # build or all stage
  groundtruth.log           # groundtruth or all stage
  search_0.log              # search or all stage
  clauses_0.csv
  results_0.bin
  ...                       # further measured repetitions
```

The signature hashes the search settings, requested warm-up count, repetitions, and query limit. Timestamps separate runs with the same configuration. `summary.json` records commands, exit codes, configuration, input hashes, ground truth, individual measurements, and medians. `complete` is set only after all requested stages succeed. Latency percentiles are computed within each run, then summarized by their median across runs.

`warmup` records the requested count, and `effective_warmup_queries` records the count after clipping to the query workload. `avg_latency_ms` is batch wall time divided by query count. P50/P95/P99 use individual query timings. With multiple workers, `avg_latency_ms` reflects throughput while the percentiles describe response latency.

## V4 index components

For an index prefix `artifacts/sift1m/index`, the builder writes:

```text
index.meta
index.vectors.bin
index.attributes.bin
index.attr_0.order.bin
index.attr_0.graph.bin
index.attr_0.postings.bin
...
```

The attribute suffix uses a unified field ID, with numerical fields before categorical fields. Metadata records the schema, construction settings, statistics, component sizes, and checksums. Attribute components store typed keys and mappings, segmented routing storage or a shared graph owner, and categorical postings.

Keep all files for a prefix together. The loader checks component consistency and accepts V4 indexes only. Older indexes must be rebuilt. Serialization identifiers still use the `rcfs` name.
