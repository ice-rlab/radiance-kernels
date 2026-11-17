// bench_gg_compute -- ggstack "compute": Gemmini's compute array running.
//
// Thread 0 configures Gemmini once, then repeatedly runs a 128x128x128 FP8 matmul on
// whatever is already in the scratchpad, with every load and store skipped
// (matmul_tile_async: skip_lda/ldb/ldd = 1, and no move-out). Only the execute side runs,
// so Gemmini is busy with no memory stall -> "compute". The values are garbage; only the
// timing matters.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 1
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 32
#endif

#include "ubench.hpp"
#include "gg_common.hpp"

UBENCH_BODY(u) {
  if (u.tid == 0) {
    configure_mxgemmini<GG>(GG.TILE_M, GG.TILE_N, GG.TILE_K);

    #pragma unroll 1
    for (uint32_t i = 0; i < u.iters; ++i) {
      matmul_tile_async<GG>(/*tile_k=*/0, /*acc_move_out=*/false);
      gemmini_fence();
    }
  }
  ubench_sink(u, u.tid);
}
