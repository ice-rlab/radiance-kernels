// bench_ifetch -- bucket 4 "ifetch": ibuf empty AND an icache miss is in flight.
//
// Needs a hot code footprint larger than the L0 icache, so every pass misses:
// a big fully-unrolled straight-line body (or a ring of distinct functions)
// iterated u.iters times. Check the size in bench_ifetch.radiance.dump.
// Fewer warps (NUM_WARPS=1) keeps other warps from hiding the miss.

#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
// Each pass is 4096 instructions, so far fewer passes are needed.
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 64
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t a = u.tid, b = a + 1, c = a + 2, d = a + 3;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 4096 straight-line addi x 8 B = 32 KiB of code, twice the 16 KiB L0 icache
    // (L0iCacheConfig: 512 sets x 1 way x 32 B, direct-mapped). Every 32 B line
    // is evicted before the next pass, so every line misses again.
    // Rotating over 4 registers keeps the adds independent, so they don't add
    // comData stalls on top of the misses.
    asm volatile(".rept 1024\n\t"
                 "addi %0, %0, 1\n\t"
                 "addi %1, %1, 1\n\t"
                 "addi %2, %2, 1\n\t"
                 "addi %3, %3, 1\n\t"
                 ".endr"
                 : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
  }

  ubench_sink(u, a ^ b ^ c ^ d);
}
