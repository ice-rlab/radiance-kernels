// bench_gg_awaitstall -- ggstack "awaitstall": Muon offers a command but Gemmini can't take it.
//
// Thread 0 configures Gemmini, then sends 32 compute commands back to back WITHOUT
// waiting between them, so Gemmini's command queue should fill and refuse new commands
// (gemminiIO.ready low) -- the "sequential-config wall".
//
// Expected to read ~0 awaitstall: in GemminiTile.scala "busy" is checked before
// awaitstall, and Gemmini's io.busy includes raw_cmd.valid, i.e. whenever a command is
// queued Gemmini already counts as busy ("compute"). This bench checks that on hardware.
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
    for (uint32_t i = 0; i < u.iters; ++i)
      matmul_tile_async<GG>(/*tile_k=*/0, /*acc_move_out=*/false);
    gemmini_fence();
  }
  ubench_sink(u, u.tid);
}
