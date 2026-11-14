#pragma once
#include <cstddef>
namespace rcfs {
using DistanceFunction = float (*)(const void*, const void*, const void*);
DistanceFunction l2_kernel(size_t dim);
class L2Distance {
  size_t dim_;
  DistanceFunction function_;
 public:
  explicit L2Distance(size_t dim) : dim_(dim), function_(l2_kernel(dim)) {}
  float operator()(const float* a, const float* b) const { return function_(a,b,&dim_); }
};
}  // namespace rcfs
