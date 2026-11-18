// bench_gg_configwrite -- ggstack "configwrite": Muon writing Gemmini's setup registers.
//
// Thread 0 writes the rs1 and rs2 staging registers over and over WITHOUT ever writing
// the instruction word, so no command is ever sent. Each 32-bit write pulses
// rs1WrStrobe / rs2WrStrobe for one cycle (GemminiTile.scala), which is "configwrite".
// Gemmini itself never gets any work, so the cycles between writes are "idle".
//
// Measured: configwrite 81.8% (one write every ~1.22 cycles).
// Tried: all 8 lanes writing -> WORSE, 50.1%: 8 lanes storing to the same register are
// serialized at one write every 2 cycles. One lane is the faster way to write.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 256
#endif

#include "ubench.hpp"
#include "gg_common.hpp"

UBENCH_BODY(u) {
  if (u.tid == 0) {
    #pragma unroll 1
    for (uint32_t i = 0; i < u.iters; ++i) {
      // 64 x 4 = 256 register-word writes per pass.
      #pragma unroll
      for (uint32_t k = 0; k < 64; ++k) {
        store_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET,     i);
        store_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET + 4, k);
        store_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET,     i);
        store_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET + 4, k);
      }
    }
  }
  ubench_sink(u, u.tid);
}
