// bench_gg_launch -- ggstack "launch": a work command being committed to Gemmini.
//
// Thread 0 writes ONLY the instruction word of a FLUSH command (funct 7, a work command
// that does almost nothing). rs1/rs2 are set to 0 once and left alone, so there is no
// configwrite traffic in the loop. Each instruction-word write commits one command
// (regValid with funct != CONFIG): that cycle is "launch" -- but ONLY if Gemmini is not
// busy that cycle, because "compute" (io.busy) is checked first.
//
// The timing problem, measured:
//   * Flushes back to back (~1 every 2 cycles): each arrived while the previous still had
//     Gemmini busy -> launch 0.2%, compute 94.6%.
//   * One flush, then gemmini_fence() (poll the busy register until 0): every launch landed
//     on an idle cycle, but polling takes ~100 cycles per command -> launch 1.0%, idle 95%.
//   A flush keeps Gemmini busy ~4 cycles, so the best possible is 1 launch cycle per
//   1 + 4 cycles, ~20%, reached only if the next flush arrives right as the last finishes.
//
// So: a FIXED delay instead of polling -- each flush followed by 4 nops, in one long
// unrolled block. One instruction issues per cycle, so the next flush comes ~5 cycles later.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 16
#endif
// nops after each flush. Measured with 4: 10.6 cycles between commits (each instruction
// takes ~2 cycles to issue with one warp) -> launch 9.45%, 1 launch + 4 busy + 5.6 idle per
// flush. A flush is busy ~4 cycles, so the target gap is ~5 cycles: 1-2 nops.
#ifndef GG_LAUNCH_NOPS
#define GG_LAUNCH_NOPS 4
#endif
#define GG_STR2(x) #x
#define GG_STR(x) GG_STR2(x)

#include "ubench.hpp"
#include "gg_common.hpp"

UBENCH_BODY(u) {
  if (u.tid == 0) {
    store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET, 0);  // flush: skip = 0
    store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET, 0);
    const uint32_t flush = gg_inst(k_FLUSH);
    const uint32_t inst_addr = GEMMINI_INST_ADDR;

    #pragma unroll 1
    for (uint32_t i = 0; i < u.iters; ++i) {
      // 256 flushes per pass, each followed by GG_LAUNCH_NOPS nops.
      // At most 256 x 5 instructions x 8 B = 10 KiB, inside the 16 KiB icache.
      asm volatile(".rept 256\n\t"
                   "sw.shared %0, 0(%1)\n\t"
                   ".rept " GG_STR(GG_LAUNCH_NOPS) "\n\tnop\n\t.endr\n\t"
                   ".endr"
                   :: "r"(flush), "r"(inst_addr) : "memory");
    }
    gemmini_fence();
  }
  ubench_sink(u, u.tid);
}
