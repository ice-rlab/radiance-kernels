// bench_control_miss -- bucket 2 "control", made larger with icache misses.
//
// Same idea as bench_control (every instruction is a never-taken branch, one warp),
// but the loop is 32 KiB, twice the 16 KiB L0 icache, so every pass misses.
//
// Why misses make "control" bigger:
//   "control" lasts until the instructions that fetch already asked for after a branch
//   have come back and been thrown away. In bench_control those come back fast (cache
//   hits), so control is only ~2 cycles per branch. Here the fetch past a branch often
//   goes to a line that is not in the cache, so the throw-away has to wait for the miss,
//   and all that waiting is counted as control. The re-fetch after the branch then hits,
//   because the thrown-away fetch already brought the line into the cache.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 64
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 4096 back-to-back never-taken branches x 8 B = 32 KiB.
    asm volatile(".rept 4096\n\tbne zero, zero, .+8\n\t.endr" ::: "memory");
  }

  ubench_sink(u, u.tid);
}
