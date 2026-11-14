#pragma once

#include <vector>

#include "rcfs/index.h"
#include "rcfs/predicate.h"
#include "rcfs/types.h"

namespace rcfs {

class RcfsSearcher {
 public:
  explicit RcfsSearcher(const RcfsIndex& index) : index_(index) {}

  SearchResult search(const std::vector<float>& query,
                      const CompiledPredicate& predicate,
                      const SearchParams& params = SearchParams()) const;

  // Workers reuse private workspaces; result rows retain the input order.
  std::vector<SearchResult> batch_search(
      const DenseFloatMatrix& queries,
      const std::vector<CompiledPredicate>& predicates,
      const SearchParams& params = SearchParams()) const;

 private:
  const RcfsIndex& index_;  // Must outlive the searcher and remain immutable during queries.
};

}  // namespace rcfs
