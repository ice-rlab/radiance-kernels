// bench_idle -- bucket 5 "idle": ibuf empty with no icache miss in flight.
//
// The fetch-side bubble: after a control hazard is fetched, fetch for that warp
// stalls until the instruction resolves in the backend, and the ibuf drains.
// Taken branches / jumps every few instructions, with NUM_WARPS=1 so no other
// warp fills the fetch slots. Note the discard window just after the hazard
// lands in control (higher priority), so expect control + idle together.

#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t acc = u.tid;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // Smallest possible body: the loop is just add, counter add, taken branch.
    // After every branch fetch waits for it to resolve, and with one warp
    // nothing else fills the gap.
    acc += 1;
    ubench_keep(acc);
  }

  ubench_sink(u, acc);
}
