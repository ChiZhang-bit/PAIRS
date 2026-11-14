#pragma once

#include "rcfs/index.h"

namespace rcfs {

class RcfsIndexBuilder {
 public:
  // Dataset is accepted by value so the builder can normalize and move the
  // shared columns into the finished index without retaining duplicate data.
  static RcfsIndex build(Dataset dataset,
                         const BuildParams& params = BuildParams(),
                         uint32_t build_threads = 1);
};

}  // namespace rcfs
