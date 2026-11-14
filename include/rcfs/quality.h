#pragma once
#include "rcfs/predicate.h"
#include "rcfs/types.h"
#include <string>
namespace rcfs {
uint32_t quality_bucket(const CompiledPredicate&);
uint64_t quality_config_fingerprint(const SearchParams&);
uint64_t quality_index_fingerprint(const std::string& prefix);
// Invalid files throw; a valid but stale sidecar is disabled and returns false.
bool load_quality_table(const std::string& path,uint64_t index_fingerprint,SearchParams&);
}  // namespace rcfs
