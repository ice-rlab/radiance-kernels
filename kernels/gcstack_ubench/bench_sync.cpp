// bench_sync -- bucket 3 "sync": the warp is parked at a vx_bar barrier.
//
// Imbalance is what creates sync time: give warps different amounts of work
// (e.g. scale the loop by u.warp -- that loop bound is divergent, see
// the trap in ubench.hpp), then mu_barrier(id, UBENCH_NUM_WARPS).
// Use ids >= UBENCH_FIRST_FREE_BARRIER (id 0 belongs to mu_schedule), and
// never reuse an id with a different participant count -- it hangs.
// Every lane of the warp must reach the barrier: do not put it under a
// lane-divergent `if` (see the barrier note in mu_intrinsics.h).
// vx_bar is itself a control hazard, so a tight barrier loop also adds control.

// How this bench works, in simple terms:
//   8 warps run. Each round, ONLY warp 0 does a long block of work (1024 adds);
//   warps 1-7 skip it and go straight to the barrier. They are stuck there ("sync")
//   until warp 0 finishes its work and also reaches the barrier. Then the next round
//   starts. So almost all the time: 1 warp working (base), 7 warps waiting (sync).
//
//   The `if (u.warp == 0)` is the same for every lane of a warp, so no warp is split
//   in two, and the barrier is outside the `if`, so every lane of every warp reaches it.
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 256
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t a = u.tid, b = a + 1, c = a + 2, d = a + 3;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    if (u.warp == 0) {
      // 1024 independent adds = 8 KiB, fits in the icache.
      asm volatile(".rept 256\n\t"
                   "addi %0, %0, 1\n\t"
                   "addi %1, %1, 1\n\t"
                   "addi %2, %2, 1\n\t"
                   "addi %3, %3, 1\n\t"
                   ".endr"
                   : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
    }
    mu_barrier(UBENCH_FIRST_FREE_BARRIER, UBENCH_NUM_WARPS);
  }

  ubench_sink(u, a ^ b ^ c ^ d);
}
