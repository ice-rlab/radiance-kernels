// bench_gg_memdata -- ggstack "memdata": Gemmini moving data while its compute array is idle.
//
// Thread 0 configures Gemmini once, then repeatedly asks its DMA to load an A and a B
// tile (128x128 FP8 each, 32 KiB total) from main memory into the scratchpad, waiting for
// each load to finish. No compute is ever issued, so while the DMA runs the load
// controller is busy and the execute controller is not: perf_io.mem_stall -> "memdata".
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
      copy_gmem_to_smem_async<GG>(GG.TILE_M, GG.TILE_N, GG.TILE_K, 0, 0, /*tile_k=*/0);
      gemmini_fence();
    }
  }
  ubench_sink(u, u.tid);
}
