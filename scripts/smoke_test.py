#!/usr/bin/env python3
"""Build and query a small PAIRS index, checking results with a Python oracle."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from make_demo import DEFAULT_K, INVALID_ID, matches, oracle, write_demo
from embed_attribute_mapping import convert


def executable(build_dir: Path, name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    for folder in (build_dir, build_dir / "bin", build_dir / "Release", build_dir / "Debug"):
        path = folder / (name + suffix)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Cannot find {} in {}. Build the project first.".format(name + suffix, build_dir))


def read_result(path: Path):
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated result header: {}".format(path))
    rows, k = struct.unpack_from("<II", raw)
    count = rows * k
    if not rows or not k or len(raw) != 8 + count * 8:
        raise ValueError("Invalid result shape or payload length: {}".format(path))
    # Native results store the entire ID array before the distance array.
    ids = struct.unpack_from("<" + "I" * count, raw, 8)
    distances = struct.unpack_from("<" + "f" * count, raw, 8 + count * 4)
    return [(list(ids[i * k:(i + 1) * k]), list(distances[i * k:(i + 1) * k])) for i in range(rows)]


def validate_result(path: Path, demo, exact: bool) -> None:
    """Check legality and ordering for GRAPH, and complete Top-k IDs for EXACT."""
    actual = read_result(path)
    truth = oracle(demo, DEFAULT_K)
    if len(actual) != len(truth):
        raise AssertionError("Result query count mismatch")
    for q, ((ids, distances), expected) in enumerate(zip(actual, truth)):
        if len(ids) != DEFAULT_K:
            raise AssertionError("Result topk mismatch")
        seen = set()
        previous = (-math.inf, -1)
        padding = False
        valid_ids = []
        for node, distance in zip(ids, distances):
            if node == INVALID_ID:
                padding = True
                if distance != math.inf:
                    raise AssertionError("Invalid-ID padding must have positive infinite distance")
                continue
            if padding or node in seen or node >= len(demo.vectors):
                raise AssertionError("Malformed or duplicate result ID at query {}".format(q))
            if not matches(demo.records[q]["predicate"], demo.attributes, node):
                raise AssertionError("Predicate false positive at query {}".format(q))
            expected_distance = math.fsum((a - b) ** 2 for a, b in zip(demo.queries[q], demo.vectors[node]))
            # The demo's integer vectors make exact float equality intentional.
            if not math.isfinite(distance) or distance != expected_distance:
                raise AssertionError("Incorrect squared L2 at query {}".format(q))
            if (distance, node) < previous:
                raise AssertionError("Results are not ordered by distance then original ID")
            previous = (distance, node)
            seen.add(node)
            valid_ids.append(node)
        if exact and valid_ids != [node for _, node in expected]:
            raise AssertionError("Exact search differs from independent scan at query {}".format(q))


def validate_stats(path: Path, visit: int, expand: int, bridge: int, width: int) -> int:
    count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["empty_reason"]:
                continue
            if row["plan"] != "graph_bridge":
                raise AssertionError("Forced GRAPH query unexpectedly used another plan")
            count += 1
            discovered = int(row["visited_nodes"])
            bridges = int(row["expanded_bridge_nodes"])
            expanded = bridges + int(row["expanded_match_nodes"])
            domain = int(row["driver_range_count"])
            # Limits apply to each clause's Driver domain. Discovering a Bridge
            # consumes a visit, but only opening its row consumes Bridge budget.
            h = min(visit, domain)
            e = min(expand, h)
            b = min(bridge, e)
            if not (discovered <= h and expanded <= min(e, discovered) and bridges <= b):
                raise AssertionError("Fixed search cap exceeded")
            if [int(row[key]) for key in ("final_visit_budget", "final_expand_budget", "final_bridge_budget")] != [h, e, b]:
                raise AssertionError("Fixed cap diagnostics mismatch")
            if int(row["final_ef_match"]) != max(DEFAULT_K, width):
                raise AssertionError("Navigation width changed during search")
    if not count:
        raise AssertionError("No nonempty GRAPH clause was tested")
    return count


def validate_attribute_formats(run, builder, search, work, demo_dir, demo):
    raw = (demo_dir / 'base_attrs.bin').read_bytes()
    _, version, header_bytes, rows, numeric, categorical = struct.unpack_from('<8sIIQII', raw)
    if (version, rows, numeric, categorical) != (4, len(demo.vectors), 2, 2):
        raise AssertionError('Demo must use self-contained v4 attributes')
    payload_end = header_bytes + rows * 10  # Two float32 and two uint8 columns.
    mapping_bytes = raw[payload_end + 4:]
    if struct.unpack_from('<I', raw, payload_end)[0] != len(mapping_bytes):
        raise AssertionError('Demo dictionary length mismatch')
    reference = (work / 'exact_1.bin').read_bytes()

    def check(name, attributes, mapping=None):
        prefix = work / 'attribute_formats' / name / 'index'
        options = [builder, '--data', demo_dir / 'base.fvecs', '--base-attrs', attributes,
                   '--index-prefix', prefix, '--M', 4, '--ef-construction', 32, '--ef-max', 64]
        if mapping is not None:
            options.extend(['--tag-mapping', mapping])
        run(name + '_build', options)
        output = prefix.parent / 'exact.bin'
        run(name + '_search', [search, '--index-prefix', prefix, '--queries', demo_dir / 'query.fvecs',
                              '--filters', demo_dir / 'filters.jsonl', '--topk', DEFAULT_K,
                              '--force-plan', 'exact_meta', '--output', output])
        if output.read_bytes() != reference:
            raise AssertionError('Attribute dictionary semantics changed: ' + name)

    legacy_dir = work / 'attribute_formats' / 'legacy_v2'
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / 'base_attrs.bin'
    mapping_path = legacy_dir / 'tag_mapping.json'
    legacy = bytearray(raw[:payload_end])
    struct.pack_into('<I', legacy, 8, 2)
    legacy_path.write_bytes(legacy)
    mapping_path.write_bytes(mapping_bytes)
    check('legacy_v2', legacy_path, mapping_path)
    convert(legacy_path, mapping_path)
    converted = legacy_path.read_bytes()
    if converted != raw or not mapping_path.is_file():
        raise AssertionError('In-place conversion changed payload/dictionary bytes or removed mapping')
    check('converted_v2', legacy_path)

    # Exercise uint16 codes above 255 while retaining the same label semantics.
    wide_dir = work / 'attribute_formats' / 'legacy_v3'
    wide_dir.mkdir(parents=True, exist_ok=True)
    wide_path, wide_mapping = wide_dir / 'base_attrs.bin', wide_dir / 'tag_mapping.json'
    wide_header = bytearray(raw[:header_bytes])
    struct.pack_into('<I', wide_header, 8, 3)
    color_descriptor = 32 + 2 * 24
    wide_header[color_descriptor + 17] = 4
    struct.pack_into('<I', wide_header, color_descriptor + 20, 257)
    color_values = raw[header_bytes + rows * 8:header_bytes + rows * 9]
    wide_codes = struct.pack('<' + 'H' * rows, *(256 if code == 0 else code for code in color_values))
    wide_raw = bytes(wide_header) + raw[header_bytes:header_bytes + rows * 8] + wide_codes + raw[header_bytes + rows * 9:payload_end]
    wide_path.write_bytes(wide_raw)
    mapping = json.loads(mapping_bytes)
    mapping['color']['0'] = 'unobserved_0'
    mapping['color'].update({str(code): 'unobserved_' + str(code) for code in range(4, 256)})
    mapping['color']['256'] = 'red'
    wide_mapping_bytes = (json.dumps(mapping) + '\n').encode('utf8')
    wide_mapping.write_bytes(wide_mapping_bytes)
    check('legacy_v3', wide_path)
    converted_path = wide_dir / 'embedded.bin'
    if converted_path.exists():
        converted_path.unlink()
    convert(wide_path, wide_mapping, converted_path)
    expected = bytearray(wide_raw)
    struct.pack_into('<I', expected, 8, 4)
    expected.extend(struct.pack('<I', len(wide_mapping_bytes)) + wide_mapping_bytes)
    if converted_path.read_bytes() != expected or wide_path.read_bytes() != wide_raw:
        raise AssertionError('Out-of-place conversion changed source or column payload')
    check('converted_v3', converted_path)

    wrong_length = bytearray(raw)
    struct.pack_into('<I', wrong_length, payload_end, len(mapping_bytes) + 1)
    invalid_utf8 = raw[:payload_end + 4] + mapping_bytes.replace(b'red', b'r\xffd')
    bad_schema = json.loads(mapping_bytes)
    bad_schema['other'] = bad_schema.pop('color')
    schema_bytes = json.dumps(bad_schema).encode('utf8')
    corruptions = {
        'truncated_tail': (raw[:-1], 'mapping length mismatch'),
        'extra_tail': (raw + b'x', 'mapping length mismatch'),
        'missing_length': (raw[:payload_end], 'Truncated embedded'),
        'wrong_length': (wrong_length, 'mapping length mismatch'),
        'invalid_utf8': (invalid_utf8, 'Invalid UTF-8'),
        'schema_mismatch': (raw[:payload_end] + struct.pack('<I', len(schema_bytes)) + schema_bytes, 'dictionary/schema mismatch'),
    }
    for name, (content, message) in corruptions.items():
        path = work / 'attribute_formats' / (name + '.bin')
        path.write_bytes(content)
        run(name, [builder, '--data', demo_dir / 'base.fvecs', '--base-attrs', path,
                   '--index-prefix', work / 'attribute_formats' / 'invalid'], expect_error=message)
    run('embedded_external_override', [builder, '--data', demo_dir / 'base.fvecs',
        '--base-attrs', demo_dir / 'base_attrs.bin', '--tag-mapping', mapping_path,
        '--index-prefix', work / 'attribute_formats' / 'invalid'], expect_error='omit external mapping')
    return len(corruptions) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    demo_dir = work / "demo_v4"
    demo = write_demo(demo_dir)
    builder = executable(args.build_dir, "build_pairs_index")
    search = executable(args.build_dir, "search_pairs_index")
    groundtruth = executable(args.build_dir, "generate_pairs_groundtruth")
    commands = []

    def run(name, argv, expect_error=None):
        command = [str(item) for item in argv]
        commands.append({"name": name, "argv": command})
        (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=90, check=False)
        log = work / (name + ".log")
        log.write_bytes(completed.stdout)
        if expect_error is not None:
            if completed.returncode == 0 or expect_error not in completed.stdout.decode('utf8', errors='replace'):
                raise AssertionError('Malformed input was not rejected as expected: ' + name)
        elif completed.returncode:
            raise RuntimeError("{} failed ({}). {}\n{}".format(name, completed.returncode, log,
                               completed.stdout.decode("utf-8", errors="replace")[-5000:]))

    prefix = work / "index" / "demo"
    run("build", [builder, "--data", demo_dir / "base.fvecs", "--base-attrs", demo_dir / "base_attrs.bin",
                  "--index-prefix", prefix,
                  "--M", 4, "--ef-construction", 32, "--ef-max", 64, "--build-threads", 2])
    common = ["--index-prefix", prefix, "--queries", demo_dir / "query.fvecs",
              "--filters", demo_dir / "filters.jsonl", "--topk", DEFAULT_K]
    for threads in (1, 4):
        output = work / ("exact_{}.bin".format(threads))
        run("exact_{}".format(threads), [search, *common, "--force-plan", "exact_meta",
                                       "--threads", threads, "--output", output])
        validate_result(output, demo, exact=True)
    if (work / "exact_1.bin").read_bytes() != (work / "exact_4.bin").read_bytes():
        raise AssertionError("EXACT output changed with thread count")
    malformed_attributes_checked = validate_attribute_formats(run, builder, search, work, demo_dir, demo)
    run("groundtruth", [groundtruth, *common, "--threads", 4, "--output", work / "groundtruth.bin"])
    validate_result(work / "groundtruth.bin", demo, exact=True)
    if (work / "groundtruth.bin").read_bytes() != (work / "exact_1.bin").read_bytes():
        raise AssertionError("Ground-truth CLI and exact CLI disagree")
    graph_clauses = 0
    for bridge in (0, 8):
        for threads in (1, 4):
            name = "graph_b{}_t{}".format(bridge, threads)
            output, stats = work / (name + ".bin"), work / (name + ".csv")
            run(name, [search, *common, "--force-plan", "graph_bridge", "--threads", threads,
                       "--visit-budget", 48, "--expand-budget", 32, "--max-bridge-budget", bridge,
                       "--ef-match", 16, "--entry-probe-budget", 8, "--probe-budget", 8,
                       "--output", output, "--stats", stats])
            validate_result(output, demo, exact=False)
            graph_clauses += validate_stats(stats, 48, 32, bridge, 16)
        if (work / "graph_b{}_t1.bin".format(bridge)).read_bytes() != (work / "graph_b{}_t4.bin".format(bridge)).read_bytes():
            raise AssertionError("GRAPH output changed with thread count")
    summary = {"status": "passed", "base_rows": len(demo.vectors), "queries": len(demo.queries),
               "graph_clauses_checked": graph_clauses, "commands": len(commands),
               "malformed_attributes_checked": malformed_attributes_checked,
               "checks": ["independent exact IDs and distances", "ground-truth CLI", "GRAPH predicate legality",
                          "fixed limits and width", "zero bridge cap", "thread determinism", "V4 CLI reload",
                          "embedded v4 dictionaries", "legacy v2/v3 compatibility", "uint16 categories",
                          "byte-preserving attribute conversion", "malformed attribute tails rejected"]}
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PAIRS release smoke passed: {} queries, {} GRAPH clauses. Evidence: {}".format(
        len(demo.queries), graph_clauses, work))


if __name__ == "__main__":
    main()
