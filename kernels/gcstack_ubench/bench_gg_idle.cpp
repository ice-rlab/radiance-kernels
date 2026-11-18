// bench_gg_idle -- ggstack "idle": Gemmini has nothing to do (unfed by Muon).
//
// The GPU runs a long loop of plain adds and never touches Gemmini, so for the whole
// GPU-active window Gemmini should read 100% idle in the kernel-scoped stack.
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
    asm volatile(".rept 256\n\t"
                 "addi %0, %0, 1\n\taddi %1, %1, 1\n\t"
                 "addi %2, %2, 1\n\taddi %3, %3, 1\n\t"
                 ".endr"
                 : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
  }

  ubench_sink(u, a ^ b ^ c ^ d);
}
