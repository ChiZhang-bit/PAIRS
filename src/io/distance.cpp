#include "rcfs/distance.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "base_hnsw/space_l2.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
namespace rcfs {
DistanceFunction l2_kernel(size_t dim) {
  // Thread-safe static initialization finishes all HNSW SIMD dispatch writes once.
  struct Kernels {
    DistanceFunction by16, by4, tail16, tail4, scalar;
    Kernels() {
      base_hnsw::L2Space a(16), b(4), c(17), d(5), e(1);
      by16=a.get_dist_func(); by4=b.get_dist_func(); tail16=c.get_dist_func();
      tail4=d.get_dist_func(); scalar=e.get_dist_func();
    }
  };
  static const Kernels kernels;
  if (dim % 16 == 0) return kernels.by16;
  if (dim % 4 == 0) return kernels.by4;
  if (dim > 16) return kernels.tail16;
  if (dim > 4) return kernels.tail4;
  return kernels.scalar;
}
}  // namespace rcfs
