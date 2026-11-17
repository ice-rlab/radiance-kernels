// bench_<name>: <one line: which gcstack bucket this drives, and how>
//
// Copy to bench_<name>.cpp; `make` picks it up automatically.

// Per-bench defaults (a `make NUM_WARPS=.. ITERS=..` override still wins):
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 8
#endif
// #define UBENCH_SETUP   // then define `void ubench_setup() { ... }` below

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t acc = u.tid;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // TODO: the instruction pattern under test
    ubench_keep(acc);
  }

  ubench_sink(u, acc);
}
