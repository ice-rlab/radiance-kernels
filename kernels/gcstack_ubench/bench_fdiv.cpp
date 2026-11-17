// bench_fdiv -- measures how long one FP32 divide (fdiv.s) takes for a whole warp.
//
// How this bench works, in simple terms:
//   One warp runs a chain of 16384 divides, each one using the previous one's result,
//   so they cannot overlap. With nothing else running, the run time divided by 16384
//   is the latency of one divide for a warp.
//
//   Muon's divide unit has only 2 FP32 lanes (FPPipeParams numFP32DivLanes = 2), so an
//   8-thread warp's divide goes through it in 4 passes; this number includes that.
//
//   The chain rotates over 4 registers so the "destination still busy" check
//   (Hazard.scala hasWAW / hasWAR) never holds an instruction back. It divides by 1.0,
//   so the value never changes and never becomes a denormal or infinity.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 16
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  // FP32 bit patterns held in integer registers (Muon uses zfinx: FP ops on x regs).
  uint32_t x0 = 0x3fc00000u;  // 1.5f
  uint32_t x1 = 0, x2 = 0, x3 = 0;
  uint32_t d = 0x3f800000u;   // 1.0f
  ubench_keep(x0);
  ubench_keep(d);

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 1024 dependent divides (256 x 4) = 8 KiB, fits in the icache.
    asm volatile(".rept 256\n\t"
                 "fdiv.s %1, %0, %4\n\t"
                 "fdiv.s %2, %1, %4\n\t"
                 "fdiv.s %3, %2, %4\n\t"
                 "fdiv.s %0, %3, %4\n\t"
                 ".endr"
                 : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3)
                 : "r"(d));
  }

  ubench_sink(u, x0);
}
