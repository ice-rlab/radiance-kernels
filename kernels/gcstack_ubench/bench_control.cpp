// bench_control -- bucket 2 "control": fetch is discarding wrong-path fetches
// after a control-hazard instruction (WarpScheduler discardValid).
//
// Triggers (Decode.scala HasControlHazard): BRANCH, JAL, JALR, SYSTEM (incl.
// every csrr!), MISC_MEM (fence), NU_INVOKE, and vx_tmc/split/pred/wspawn/bar.
// A discard only happens if fetch had already run ahead past the hazard, so
// dense hazards with straight-line code between them (e.g. `#pragma unroll`
// of a never-taken compare-and-branch) maximize it. The time the warp waits
// for the hazard to RESOLVE afterwards shows up as idle, not control.
// Highest-priority stall bucket: it masks everything below it.

// How "control" is counted (WarpScheduler.scala, discardValid):
//   When a hazard instruction (branch, jump, csrr, fence, ...) comes back from the
//   icache, and fetch has ALREADY asked for instructions after it, the warp goes into
//   "discard" mode and throws those extra instructions away as they arrive. Only that
//   throw-away time is "control". Waiting for the hazard to finish afterwards is "idle",
//   and fetching again after it is "ifetch".
//
// So to get as much control as possible:
//   - Make EVERY instruction a hazard, so every one of them has extra fetches to throw away.
//   - Use a hazard that finishes fast (a branch on zero == zero that is never taken:
//     its inputs are always ready), so little time goes to idle.
//   - Use one warp, so fetch runs as far ahead as it can and there is more to throw away.
//   - Keep the loop under the 16 KiB icache, so there are no real cache misses.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 256
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 1024 back-to-back never-taken branches x 8 B = 8 KiB.
    // .+8 is the next instruction (Muon instructions are 8 bytes), so even if one
    // were taken it would land on the next branch.
    asm volatile(".rept 1024\n\tbne zero, zero, .+8\n\t.endr" ::: "memory");
  }

  ubench_sink(u, u.tid);
}
