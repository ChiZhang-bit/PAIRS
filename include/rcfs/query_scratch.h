#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "rcfs/types.h"
namespace rcfs {
// Bounded open-address table. Insertions fail at the limit without allocating.
// Resets reuse storage by advancing its epoch.
template<class T> class FlatQueryTable {
  std::vector<NodeId> keys_;
  std::vector<T> values_;
  std::vector<uint32_t> epochs_;
  uint32_t epoch_ = 0;
  size_t size_ = 0, limit_ = 0;
  size_t slot(NodeId key) const { return (uint64_t(key) * 11400714819323198485ULL) & (keys_.size()-1); }
 public:
  void reset(size_t limit) {
    limit_ = limit; size_ = 0;
    if (limit > std::numeric_limits<size_t>::max()/2) throw std::length_error("Query table limit");
    size_t capacity = 2;
    while (capacity < limit*2) {
      if (capacity > std::numeric_limits<size_t>::max()/2) throw std::length_error("Query table capacity");
      capacity *= 2;
    }
    if (capacity > keys_.size()) {
      keys_.resize(capacity); values_.resize(capacity); epochs_.assign(capacity,0); epoch_ = 0;
    }
    if (++epoch_ == 0) { std::fill(epochs_.begin(),epochs_.end(),0); epoch_ = 1; }
  }
  T* find(NodeId key) {
    if (keys_.empty()) return nullptr;
    size_t p = slot(key);
    while (epochs_[p] == epoch_) {
      if (keys_[p] == key) return &values_[p];
      p = (p+1) & (keys_.size()-1);
    }
    return nullptr;
  }
  bool insert(NodeId key, T value) {
    if (size_ >= limit_) return false;
    size_t p = slot(key);
    while (epochs_[p] == epoch_) {
      if (keys_[p] == key) { values_[p] = value; return true; }
      p = (p+1) & (keys_.size()-1);
    }
    epochs_[p] = epoch_; keys_[p] = key; values_[p] = value; ++size_; return true;
  }
  size_t size() const { return size_; }
};
// Predicate results use original IDs and are valid for one clause only.
struct ProbeWorkspace { FlatQueryTable<uint8_t> matches; };
}  // namespace rcfs
