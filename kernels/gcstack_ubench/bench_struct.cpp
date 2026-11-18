// bench_struct -- bucket 8 "struct": ibuf head is valid but cannot be admitted
// (RS full, LSU reservation unavailable, ...), with no data dependence.
//
// Data stalls outrank struct, so the stream must be dependence-free: e.g. a
// burst of independent stores to SMEM/GMEM (stores have no rd) to exhaust the
// LSU reservations, or many independent long-latency ops to fill the RS.
// Many warps (default 8) contend for the same structures.

#include "ubench.hpp"

UBENCH_BODY(u) {
  uint32_t acc = u.tid;
  // Each thread writes its own column; 64 >= threads per core, so no two
  // threads share a word.
  __global uint32_t* p = u.gmem + u.tid;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 8 independent stores back to back: no destination register, so nothing
    // waits on data, but they use up the load/store unit's slots.
    p[0 * 64] = acc;
    p[1 * 64] = acc;
    p[2 * 64] = acc;
    p[3 * 64] = acc;
    p[4 * 64] = acc;
    p[5 * 64] = acc;
    p[6 * 64] = acc;
    p[7 * 64] = acc;
    ubench_keep(acc);  // its "memory" clobber also keeps all 8 stores every pass
  }

  ubench_sink(u, acc);
}
