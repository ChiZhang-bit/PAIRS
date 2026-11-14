#!/usr/bin/env python3
"""Generate a small PAIRS dataset with repeatable inputs and exact answers.

The predicate file grammar has no TRUE/FALSE literals. The demo uses a full
numeric range and contradictory ranges for those cases. URFATTR v4 has no
missing-category encoding, so the demo contains no missing categories.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Tuple


INVALID_ID = 0xFFFFFFFF
DIMENSION = 8
DEFAULT_ROWS = 192
DEFAULT_K = 10


@dataclass
class Demo:
    vectors: List[List[float]]
    queries: List[List[float]]
    attributes: Dict[str, List[Any]]
    records: List[dict]


def matches(predicate: dict, attributes: dict, node: int) -> bool:
    """Evaluate the original AST directly, without DNF or index structures."""
    if len(predicate) != 1:
        raise ValueError("Each predicate node must have exactly one key")
    field, value = next(iter(predicate.items()))
    if field == "AND":
        return all(matches(child, attributes, node) for child in value)
    if field == "OR":
        return any(matches(child, attributes, node) for child in value)
    actual = attributes[field][node]
    if isinstance(value, list):
        return value[0] <= actual <= value[1]
    return actual is not None and actual == value


def oracle(demo: Demo, k: int = DEFAULT_K) -> List[List[Tuple[float, int]]]:
    """Scan the original predicates and sort matches by squared L2, then ID."""
    if k < 1:
        raise ValueError("topk must be positive")
    answers = []
    for query, record in zip(demo.queries, demo.records):
        hits = []
        for node, vector in enumerate(demo.vectors):
            if matches(record["predicate"], demo.attributes, node):
                distance = math.fsum((a - b) ** 2 for a, b in zip(query, vector))
                hits.append((distance, node))
        answers.append(sorted(hits)[:k])
    return answers


def create_demo(rows: int = DEFAULT_ROWS) -> Demo:
    if rows < 32:
        raise ValueError("The demonstration needs at least 32 rows")
    # These small integer components and their squared-distance sums fit
    # exactly in float32. Python's oracle and the native SIMD kernel therefore
    # rank the same stored values without a rounding-dependent tie order.
    vectors = [[float(((node * 13 + axis * 7) % 61) - 30)
                for axis in range(DIMENSION)] for node in range(rows)]
    vectors[0] = vectors[1] = [0.0] * DIMENSION
    attributes = {
        "price": [float(node % 24) for node in range(rows)],
        "quality": [float((node * 7) % 17) for node in range(rows)],
        "color": [("red", "blue", "green")[node % 3] for node in range(rows)],
        "bool": [bool(node % 2) for node in range(rows)],
    }
    attributes["price"][2] = 1.0
    attributes["price"][3] = 1.0 + 2.0 ** -23
    # The double_boundary query starts just above 1.0 but below the next
    # float32 value. Rounding that bound to float would wrongly admit row 2.
    filters = [
        ("range", {"price": [3, 8]}),
        ("and", {"AND": [{"price": [2, 17]}, {"quality": [5, 12]}, {"color": "red"}]}),
        ("overlapping_or", {"OR": [{"price": [4, 12]},
                                  {"AND": [{"quality": [2, 14]}, {"bool": True}]}]}),
        ("cnf", {"AND": [{"OR": [{"price": [0, 4]}, {"color": "blue"}]},
                         {"OR": [{"quality": [9, 16]}, {"bool": True}]}]}),
        ("category_only", {"color": "red"}),
        ("category_and", {"AND": [{"color": "blue"}, {"bool": False}]}),
        ("true_domain", {"price": [-1, 30]}),
        ("false_contradiction", {"AND": [{"price": [0, 2]}, {"price": [20, 23]}]}),
        ("empty_category", {"color": "unused"}),
        ("double_boundary", {"price": [math.nextafter(1.0, 2.0), 1.0 + 2.0 ** -23]}),
        ("duplicate_value", {"price": [5, 5]}),
        ("distance_ties", {"price": [0, 1]}),
    ]
    records = [{"query_id": i, "template": name, "predicate": predicate}
               for i, (name, predicate) in enumerate(filters)]
    queries = [list(vectors[(17 * i + 5) % rows]) for i in range(len(records))]
    queries[9] = list(vectors[3])
    queries[11] = [0.0] * DIMENSION
    return Demo(vectors, queries, attributes, records)


def write_fvecs(path: Path, rows: List[List[float]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<i", len(row)))
            stream.write(struct.pack("<" + "f" * len(row), *row))


def write_demo(output: Path, rows: int = DEFAULT_ROWS, k: int = DEFAULT_K) -> Demo:
    demo = create_demo(rows)
    answers = oracle(demo, k)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_fvecs(output / "base.fvecs", demo.vectors)
    write_fvecs(output / "query.fvecs", demo.queries)
    mapping = {"color": {"0": "red", "1": "blue", "2": "green", "3": "unused"},
               "bool": {"0": "false", "1": "true"}}
    mapping_bytes = (json.dumps(mapping, ensure_ascii=False) + "\n").encode("utf-8")
    # URFATTR v4 adds a length-prefixed dictionary after the column payload.
    descriptors = [("price", 1, 1, 0), ("quality", 1, 1, 0),
                   ("color", 2, 3, 4), ("bool", 2, 3, 2)]
    with (output / "base_attrs.bin").open("wb") as stream:
        stream.write(struct.pack("<8sIIQII", b"URFATTR\0", 4, 32 + 24 * len(descriptors), rows, 2, 2))
        for name, kind, dtype, cardinality in descriptors:
            stream.write(struct.pack("<16sBBHI", name.encode("ascii"), kind, dtype, 0, cardinality))
        for name in ("price", "quality"):
            stream.write(struct.pack("<" + "f" * rows, *demo.attributes[name]))
        stream.write(bytes(("red", "blue", "green").index(value) for value in demo.attributes["color"]))
        stream.write(bytes(demo.attributes["bool"]))
        stream.write(struct.pack("<I", len(mapping_bytes)))
        stream.write(mapping_bytes)
    with (output / "filters.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in demo.records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    with (output / "filtered_groundtruth.ivecs").open("wb") as stream:
        for hits in answers:
            ids = [node for _, node in hits] + [-1] * (k - len(hits))
            stream.write(struct.pack("<i" + "i" * k, k, *ids))
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dataset/demo")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--topk", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    demo = write_demo(args.output, args.rows, args.topk)
    print("Wrote {} vectors, {} queries, and independent Top-{} truth to {}".format(
        len(demo.vectors), len(demo.queries), args.topk, args.output))


if __name__ == "__main__":
    main()
