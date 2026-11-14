#include "rcfs/predicate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rcfs {
namespace {

// Merge constraints by field so each surviving clause has one atom per field.
bool canonicalize_clause(CompiledClause* clause) {
  std::sort(clause->numeric_ranges.begin(), clause->numeric_ranges.end(),
            [](const NumericRangeAtom& left, const NumericRangeAtom& right) {
              if (left.attribute != right.attribute) {
                return left.attribute < right.attribute;
              }
              if (left.lower != right.lower) return left.lower < right.lower;
              return left.upper < right.upper;
            });
  std::vector<NumericRangeAtom> normalized_ranges;
  normalized_ranges.reserve(clause->numeric_ranges.size());
  for (const NumericRangeAtom& atom : clause->numeric_ranges) {
    if (!std::isfinite(atom.lower) || !std::isfinite(atom.upper) ||
        atom.lower > atom.upper) {
      throw std::invalid_argument("Invalid numeric range atom");
    }
    if (normalized_ranges.empty() ||
        normalized_ranges.back().attribute != atom.attribute) {
      normalized_ranges.push_back(atom);
      continue;
    }
    NumericRangeAtom& merged = normalized_ranges.back();
    merged.lower = std::max(merged.lower, atom.lower);
    merged.upper = std::min(merged.upper, atom.upper);
    if (merged.lower > merged.upper) return false;
  }
  clause->numeric_ranges = std::move(normalized_ranges);

  std::sort(
      clause->categorical_equalities.begin(),
      clause->categorical_equalities.end(),
      [](const CategoricalEqualityAtom& left,
         const CategoricalEqualityAtom& right) {
        if (left.attribute != right.attribute) {
          return left.attribute < right.attribute;
        }
        return left.value < right.value;
      });
  std::vector<CategoricalEqualityAtom> normalized_equalities;
  normalized_equalities.reserve(clause->categorical_equalities.size());
  for (const CategoricalEqualityAtom& atom :
       clause->categorical_equalities) {
    if (normalized_equalities.empty() ||
        normalized_equalities.back().attribute != atom.attribute) {
      normalized_equalities.push_back(atom);
      continue;
    }
    if (normalized_equalities.back().value != atom.value) return false;
  }
  clause->categorical_equalities = std::move(normalized_equalities);
  return true;
}

bool clauses_equal(const CompiledClause& left, const CompiledClause& right) {
  if (left.numeric_ranges.size() != right.numeric_ranges.size() ||
      left.categorical_equalities.size() !=
          right.categorical_equalities.size()) {
    return false;
  }
  for (size_t i = 0; i < left.numeric_ranges.size(); ++i) {
    const NumericRangeAtom& a = left.numeric_ranges[i];
    const NumericRangeAtom& b = right.numeric_ranges[i];
    if (a.attribute != b.attribute || a.lower != b.lower ||
        a.upper != b.upper) {
      return false;
    }
  }
  for (size_t i = 0; i < left.categorical_equalities.size(); ++i) {
    const CategoricalEqualityAtom& a = left.categorical_equalities[i];
    const CategoricalEqualityAtom& b = right.categorical_equalities[i];
    if (a.attribute != b.attribute || a.value != b.value) return false;
  }
  return true;
}

// Returns true when every object satisfying `specific` also satisfies
// `covering`. In a disjunction, the specific clause is then redundant.
bool clause_covers(const CompiledClause& covering,
                   const CompiledClause& specific) {
  for (const NumericRangeAtom& required : covering.numeric_ranges) {
    const auto found = std::lower_bound(
        specific.numeric_ranges.begin(), specific.numeric_ranges.end(),
        required.attribute,
        [](const NumericRangeAtom& atom, NumericAttrId attribute) {
          return atom.attribute < attribute;
        });
    if (found == specific.numeric_ranges.end() ||
        found->attribute != required.attribute ||
        found->lower < required.lower || found->upper > required.upper) {
      return false;
    }
  }
  for (const CategoricalEqualityAtom& required :
       covering.categorical_equalities) {
    const auto found = std::lower_bound(
        specific.categorical_equalities.begin(),
        specific.categorical_equalities.end(), required.attribute,
        [](const CategoricalEqualityAtom& atom,
           CategoricalAttrId attribute) {
          return atom.attribute < attribute;
        });
    if (found == specific.categorical_equalities.end() ||
        found->attribute != required.attribute ||
        found->value != required.value) {
      return false;
    }
  }
  return true;
}

void append_unique_clause(CompiledClause clause, uint32_t max_clauses,
                          std::vector<CompiledClause>* clauses) {
  if (!canonicalize_clause(&clause)) return;
  const auto duplicate = std::find_if(
      clauses->begin(), clauses->end(),
      [&clause](const CompiledClause& existing) {
        return clauses_equal(existing, clause);
      });
  if (duplicate != clauses->end()) return;
  for (const CompiledClause& existing : *clauses) {
    if (clause_covers(existing, clause)) return;
  }
  clauses->erase(
      std::remove_if(clauses->begin(), clauses->end(),
                     [&clause](const CompiledClause& existing) {
                       return clause_covers(clause, existing);
                     }),
      clauses->end());
  // Count only clauses retained after local simplification.
  if (clauses->size() >= max_clauses) {
    throw std::runtime_error("Predicate DNF exceeds max_clauses=" +
                             std::to_string(max_clauses));
  }
  clauses->push_back(std::move(clause));
}

bool merge_clauses(const CompiledClause& left, const CompiledClause& right,
                   CompiledClause* result) {
  result->numeric_ranges = left.numeric_ranges;
  result->numeric_ranges.insert(result->numeric_ranges.end(),
                                right.numeric_ranges.begin(),
                                right.numeric_ranges.end());
  result->categorical_equalities = left.categorical_equalities;
  result->categorical_equalities.insert(
      result->categorical_equalities.end(),
      right.categorical_equalities.begin(),
      right.categorical_equalities.end());
  return canonicalize_clause(result);
}

std::vector<CompiledClause> compile_node(const PredicateNode& node,
                                         uint32_t max_clauses) {
  switch (node.kind) {
    case PredicateNodeKind::kNumericRange: {
      if (!node.children.empty()) {
        throw std::invalid_argument("Numeric predicate leaf has children");
      }
      CompiledClause clause;
      clause.numeric_ranges.push_back(node.numeric_range);
      if (!canonicalize_clause(&clause)) return {};
      return {std::move(clause)};
    }
    case PredicateNodeKind::kCategoricalEquality: {
      if (!node.children.empty()) {
        throw std::invalid_argument("Categorical predicate leaf has children");
      }
      CompiledClause clause;
      clause.categorical_equalities.push_back(node.categorical_equality);
      if (!canonicalize_clause(&clause)) return {};
      return {std::move(clause)};
    }
    case PredicateNodeKind::kOr: {
      if (node.children.size() < 2) {
        throw std::invalid_argument("OR requires at least two children");
      }
      std::vector<CompiledClause> result;
      for (const PredicateNode& child : node.children) {
        std::vector<CompiledClause> child_clauses =
            compile_node(child, max_clauses);
        for (CompiledClause& clause : child_clauses) {
          append_unique_clause(std::move(clause), max_clauses, &result);
        }
      }
      return result;
    }
    case PredicateNodeKind::kAnd: {
      if (node.children.size() < 2) {
        throw std::invalid_argument("AND requires at least two children");
      }
      std::vector<CompiledClause> accumulated(1);
      for (const PredicateNode& child : node.children) {
        const std::vector<CompiledClause> child_clauses =
            compile_node(child, max_clauses);
        if (child_clauses.empty()) return {};
        std::vector<CompiledClause> next;
        // Simplify each product immediately to bound the intermediate DNF.
        for (const CompiledClause& left : accumulated) {
          for (const CompiledClause& right : child_clauses) {
            CompiledClause merged;
            if (merge_clauses(left, right, &merged)) {
              append_unique_clause(std::move(merged), max_clauses, &next);
            }
          }
        }
        accumulated = std::move(next);
        if (accumulated.empty()) return {};
      }
      return accumulated;
    }
  }
  throw std::invalid_argument("Unknown PredicateNodeKind");
}

}  // namespace

CompiledPredicate compile_bounded_dnf(const PredicateNode& predicate,
                                      uint32_t max_clauses) {
  if (max_clauses == 0) {
    throw std::invalid_argument("max_clauses must be positive");
  }
  CompiledPredicate result;
  result.clauses = compile_node(predicate, max_clauses);
  return result;
}

std::vector<CompiledPredicate> compile_predicate_records(
    const std::vector<PredicateRecord>& records, uint32_t max_clauses) {
  std::vector<CompiledPredicate> compiled;
  compiled.reserve(records.size());
  for (const PredicateRecord& record : records) {
    compiled.push_back(compile_bounded_dnf(record.predicate, max_clauses));
  }
  return compiled;
}

bool matches_clause(const Dataset& dataset, const CompiledClause& clause,
                    NodeId node_id) {
  if (node_id >= dataset.vectors.rows) {
    throw std::out_of_range("Predicate check node ID is out of range");
  }
  for (const NumericRangeAtom& atom : clause.numeric_ranges) {
    if (atom.attribute >= dataset.numeric_columns.size()) {
      throw std::out_of_range(
          "Predicate numeric attribute ID is out of range");
    }
    const NumericColumn& column = dataset.numeric_columns[atom.attribute];
    if (node_id >= column.values.size()) {
      throw std::runtime_error("Numeric column is shorter than the dataset");
    }
    const double value = static_cast<double>(column.values[node_id]);
    if (!(value >= atom.lower && value <= atom.upper)) return false;
  }
  for (const CategoricalEqualityAtom& atom :
       clause.categorical_equalities) {
    if (atom.attribute >= dataset.categorical_columns.size()) {
      throw std::out_of_range(
          "Predicate categorical attribute ID is out of range");
    }
    const CategoricalColumn& column =
        dataset.categorical_columns[atom.attribute];
    if (atom.value >= column.value_names.size()) {
      throw std::out_of_range(
          "Predicate categorical value ID is out of range");
    }
    if (node_id >= column.values.size()) {
      throw std::runtime_error(
          "Categorical column is shorter than the dataset");
    }
    if (column.values[node_id] == kMissingValue ||
        column.values[node_id] != atom.value) {
      return false;
    }
  }
  return true;
}

bool matches_predicate(const Dataset& dataset,
                       const CompiledPredicate& predicate, NodeId node_id) {
  for (const CompiledClause& clause : predicate.clauses) {
    if (matches_clause(dataset, clause, node_id)) return true;
  }
  return false;
}

}  // namespace rcfs
