// bench_base -- bucket 1 "base": the warp issued an instruction this cycle.
//
// Goal: maximize base / minimize everything else. Long, branch-free run of
// independent, short-latency ALU ops (no RAW chains), unrolled so the loop
// back-edge is rare. Only ONE instruction issues per core per cycle, so the
// sum of base over all warps is at most 1.0 per cycle; with W warps the
// ceiling per warp is 1/W and the rest of each warp shows up as struct/comData.

// Why only ONE warp:
//   The Muon core can only issue one instruction per cycle, no matter how many warps
//   are running. So with 8 warps, only 1 of them can be "base" (issuing) on any cycle,
//   and the other 7 must be counted as waiting (they showed up as ifetch). Even a
//   perfect run would then read at most 1/8 = 12.5% base per warp.
//   With a single warp there is nobody to share the issue slot with, so if that warp
//   never has to wait, it can be "base" on (nearly) every cycle -- which is exactly
//   what this bench is meant to show.
//
// Why the loop body is so long:
//   After every branch the warp has to wait for the branch to finish before it can
//   fetch more instructions. 1024 adds per loop means only one branch every 1024
//   instructions, so that waiting is rare.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 256
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t a = u.tid, b = a + 1, c = a + 2, d = a + 3;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 1024 independent adds x 8 B = 8 KiB: fits in the 16 KiB L0 icache, so after the
    // first pass there are no misses, and only one branch per 1024 instructions.
    // Rotating over 4 registers: each add depends on its own register 4 ops back.
    asm volatile(".rept 256\n\t"
                 "addi %0, %0, 1\n\t"
                 "addi %1, %1, 1\n\t"
                 "addi %2, %2, 1\n\t"
                 "addi %3, %3, 1\n\t"
                 ".endr"
                 : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
  }

  ubench_sink(u, a ^ b ^ c ^ d);
}
