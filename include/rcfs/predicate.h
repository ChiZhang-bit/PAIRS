#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rcfs/schema.h"
#include "rcfs/types.h"

namespace rcfs {

enum class PredicateNodeKind : uint8_t {
  kAnd = 0,
  kOr = 1,
  kNumericRange = 2,
  kCategoricalEquality = 3,
};

struct NumericRangeAtom {
  NumericAttrId attribute = 0;
  double lower = 0.0;
  double upper = 0.0;
};

struct CategoricalEqualityAtom {
  CategoricalAttrId attribute = 0;
  ValueId value = 0;
};

// A deliberately small predicate tree.  Leaves are already resolved against
// Schema, so query execution never compares attribute names or value strings.
struct PredicateNode {
  PredicateNodeKind kind = PredicateNodeKind::kAnd;
  std::vector<PredicateNode> children;
  NumericRangeAtom numeric_range;
  CategoricalEqualityAtom categorical_equality;
};

struct PredicateRecord {
  // physical_query_id is the row in the query-vector file.  source_query_id
  // preserves an optional ID from a derived/subset workload.
  uint32_t physical_query_id = 0;
  uint32_t source_query_id = 0;
  bool has_source_query_id = false;
  std::string template_name;
  PredicateNode predicate;
};

struct CompiledClause {
  std::vector<NumericRangeAtom> numeric_ranges;
  std::vector<CategoricalEqualityAtom> categorical_equalities;
};

// Disjunction of normalized conjunctions.  An empty clauses vector denotes
// an unsatisfiable predicate (for example color=blue AND color=red).
struct CompiledPredicate {
  std::vector<CompiledClause> clauses;
};

// Read one JSON object per non-empty line.  Each record must contain a
// predicate and either query_id or local_query_id matching its physical row.
// Other workload metadata is intentionally ignored.
std::vector<PredicateRecord> read_predicate_jsonl(
    const std::string& path, const Schema& schema,
    size_t expected_query_count = 0, uint32_t max_ast_depth = 64);

// Convert an AND/OR tree into bounded DNF.  Equalities on the same
// single-valued field and ranges on the same numeric field are normalized.
// NOT is rejected by the JSON parser; exceeding max_clauses is an error.
CompiledPredicate compile_bounded_dnf(const PredicateNode& predicate,
                                      uint32_t max_clauses = 32);

std::vector<CompiledPredicate> compile_predicate_records(
    const std::vector<PredicateRecord>& records,
    uint32_t max_clauses = 32);

// Exact predicate checks used by the planner and graph search.  These
// functions do not update counters; callers own all diagnostic accounting.
bool matches_clause(const Dataset& dataset, const CompiledClause& clause,
                    NodeId node_id);
bool matches_predicate(const Dataset& dataset,
                       const CompiledPredicate& predicate, NodeId node_id);

}  // namespace rcfs
