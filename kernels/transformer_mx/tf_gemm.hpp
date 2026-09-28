// Derived from UC Berkeley Gemmini (github.com/ucb-bar/gemmini), BSD-3-Clause. See NOTICE.
//
// Serialized MX-FP8 GEMM for transformer_mx, assembled only from primitives that
// ../flash_attention_mx_stable measured correct (FA_ST_NOOVL: 144/144 tile-images at NT72 and
// clean under FA_PHASE1/2). Every rule below is a bug that directory paid for; the citations
// point at the note that records it.
//
//  1. ONE MESH OPERATION AT A TIME, ISSUED AND DRAINED BY WARP 0 WHILE EVERY OTHER WARP IS
//     PARKED AT A BARRIER. No DMA, matmul or scale write ever crosses a stage boundary, and SIMT
//     never touches SMEM while the mesh runs (Scratchpad.scala:220's 4-entry un-backpressured
//     read queue; the atomic 16-subbank grant of the accumulator move-out).
//  2. DRAIN BEFORE CONFIG_SCALE_MEM. ExecuteController latches loop_bound_* from the EX queue head
//     ungated, and ScaleFactorMem's (i,j,k) odometer re-zeroes only on a complete sweep of the
//     LIVE bounds, so a bound change mid-sweep strands it for good (FA_ST_CFGPRE). Every GEMM here
//     has different bounds from the one before it, so this applies at every call.
//  3. FENCE AFTER CONFIG_SCALE_MEM, before LOOP_WS: it is decoded by ExecuteController while
//     LOOP_WS is expanded by the LoopMatmul FSM, so MMIO order does not order them (FA_ST_CFGFENCE).
//  4. DRAIN A MATMUL ON runningLoops (MMIO 0x28), THEN BUSY. io.busy rises several cycles after
//     the issuing store, so a bare gemmini_fence() can return before the matmul starts (FA_SP_WCNT).
//  5. mu_fence_smem() IS NOT A DRAIN FOR SCALE-SRAM STORES. Publish them with mu_fence_smem() and
//     then order them at the gemmini TL port with a gemmini_fence() load (FA_ST_PROLOGF, FA_SP_QGF).
//  6. SCALE SRAM: ONE THREAD, STRICTLY ASCENDING, EVEN WORD COUNT, 8-BYTE-ALIGNED START.
//     FlitMergeNode.scala:56/62 asserts it; lane-parallel writes $finish the simulation.
//  7. OPERAND MOVE-IN BY THE LOOP-FSM DMA (one LOOP_WS, ex/stc skipped), as lib/mxgemm does.
//     The FA kernel used explicit per-16x16-tile gemmini_extended_mvin because it needed SKIP_B
//     and hit H8 with skip_lda=1; neither applies here (every GEMM moves in both operands). On
//     FireSimRadianceSingleClusterSyn the explicit-mvin form livelocked NONDETERMINISTICALLY in
//     the first GEMM (2026-09-12 I4: identical GPU image passed 3 runs, then Gemmini relaunched
//     stalled DMA loads 194M times while warp 0 polled busy), so it is not used.
//  8. ONE MATMUL PER OUTPUT TILE, TILE_K == K: no in-accumulator accumulation across K-tiles and
//     no double-buffer parity. PVF (PE_TILES_K = 16) and QKF (PE_TILES_J = 16) are the verified
//     extremes; M <= 64 was verified, larger is unmeasured.
#ifndef TF_GEMM_HPP
#define TF_GEMM_HPP
#include <stdint.h>
#include <radiance.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "mxgemmini_mmio.h"

constexpr auto GEMMINI_FORMAT_FP8 = 0;
constexpr auto GEMMINI_FORMAT_FULL = 3;
constexpr uint32_t TF_SMEM_ROWS = BANK_NUM * BANK_ROWS;       // scratchpad rows of DIM bytes
constexpr uint32_t TF_SF_HALF_BYTES = GEMMINI_SF_MEM_BUFFER_OFFSET;

// C[M][N] = A[M][K] @ B[K][N]; A scales [K/32][M], B scales [K/32][N] (e8m0, one per 32 along K).
struct TfGemm {
    uint32_t M, N, K;

    // Processing-element tiles are DIM x DIM, and DIM comes from the gemmini_params.h this kernel
    // compiles against (lib/mxgemmini: DIM 8, matching FireSimRadianceSingleClusterSyn's
    // `WithRadianceMxGemmini(dim = 8)`). This used to be a hardcoded 16, inherited from the
    // dim-16 tapeout-config FA core: every loop bound and scale-SRAM bound was then for 16x16
    // tiles on 8x8 hardware, and the Q projection came back with inf/zero cells in a fixed
    // 32-column pattern (2026-09-12 I6 dump). lib/mxgemm's GemmConfig uses DIM the same way.
    constexpr uint32_t PE_I() const { return M / DIM; }
    constexpr uint32_t PE_J() const { return N / DIM; }
    constexpr uint32_t PE_K() const { return K / DIM; }
    constexpr uint32_t A_SCALE_BYTES() const { return M * K / 32; }
    constexpr uint32_t B_SCALE_BYTES() const { return N * K / 32; }

    // SMEM map, in scratchpad rows (one 16x16 fp8 tile = DIM rows):
    //   A [0, A_ROWS)    C (bf16) [C_ROW, C_ROW + C_ROWS)    B [B_END - B_ROWS, B_END)
    constexpr uint32_t A_ROWS() const { return PE_I() * PE_K() * DIM; }
    constexpr uint32_t B_ROWS() const { return PE_K() * PE_J() * DIM; }
    constexpr uint32_t C_ROWS() const { return M * N * sizeof(uint16_t) / DIM; }
    constexpr uint32_t A_ROW() const { return 0; }
    constexpr uint32_t C_ROW() const { return A_ROWS(); }
    constexpr uint32_t B_END() const { return TF_SMEM_ROWS; }

    constexpr bool VALID() const {
        return M >= 32 && N >= 32 && K >= 32 && M % 32 == 0 && N % 32 == 0 && K % 32 == 0
            && M % DIM == 0 && N % DIM == 0 && K % DIM == 0
            // B_ROWS() <= B_END() first: these are unsigned, so a B operand larger than the
            // scratchpad (e.g. K=3072, N=64 -> 24576 rows > 16384) makes B_END() - B_ROWS()
            // underflow to a huge value and the check below pass. Caught 2026-09-18 while
            // sizing BERT-base; without it an oversized config builds and hangs on the FPGA.
            && B_ROWS() <= B_END() && A_ROWS() + C_ROWS() <= B_END() - B_ROWS()
            && C_ROW() + C_ROWS() <= B_END() - B_ROWS()
            && A_SCALE_BYTES() <= TF_SF_HALF_BYTES && B_SCALE_BYTES() <= TF_SF_HALF_BYTES;
    }
};

// CONFIG_SCALE_MEM carries a scale move-out address; nothing is moved out, it only has to be valid.
static uint32_t tf_sf_dummy[128 * 128 / 32] __attribute__((aligned(32))) = {0};

// ---- TF_PHASE<k>: the FA_PHASE harness. Delays cluster 1's warp 0 by k*64 dependent read-only
// MMIO round trips (~2.4k cyc per k) before every GEMM. It computes nothing and touches no data, so
// it can only move the schedule: a build whose output checksum changes under any TF_PHASE<k> was
// never correct. -DTF_PHASE_BOTH delays both clusters (the control).
#if defined(TF_PHASE1) || defined(TF_PHASE2) || defined(TF_PHASE3)
#if defined(TF_PHASE1)
#define TF_PHASE_N 1u
#elif defined(TF_PHASE2)
#define TF_PHASE_N 2u
#else
#define TF_PHASE_N 3u
#endif
static inline void tf_phase_skew() {
#ifndef TF_PHASE_BOTH
    uint32_t cl; asm volatile("csrr %0, 0xCD0" : "=r"(cl));
    if (cl == 0) return;
#endif
    uint32_t z = 0; asm volatile("" : "+r"(z));    // opaque zero: a real dependent chain
    uint32_t v = 0;
    for (uint32_t i = 0; i < 64u * TF_PHASE_N; i++) v = load32_shared(GEMMINI_BUSY_ADDR + (v & z));
    asm volatile("" :: "r"(v));
}
#else
static inline void tf_phase_skew() {}
#endif

static inline void tf_gf() { gemmini_fence(); }                                   // io.busy == 0
// Drain = busy poll only, as lib/mxgemm does in the kernels that pass on this FPGA. The FA rule-4
// runningLoops (MMIO 0x28) poll is NOT used: on FireSimRadianceSingleClusterSyn warp 0 was found
// polling forever with Gemmini idle and every Gemmini counter frozen, right after a GEMM's
// commands (2026-09-12, build C after head 0 S@V; I5 after 12 GEMMs).
// A single busy == 0 read can come before the loop raises busy or while its last SMEM stores are
// still in flight, so drain on a run of GEMMINI_DRAIN_QUIET consecutive idle reads instead.
static inline void tf_drain() { gemmini_drain(); }

// GPU stores to DRAM can sit in the core's L0 D cache, while Gemmini's DMA move-in (and the host)
// read DRAM directly. Rung 1's move-in read stale g_af8 elements that way (2026-09-17: q 2238/4096
// while the host read g_af8/g_afs back exact). The fence instruction that flushes the D cache hangs
// here (kernel.cpp TF_BAR), so flush through the cache's MMIO register: muon core 0's peripheral
// block is at cluster + 0x80000 + 0x200, and its L0D flush register is +0x100 (MuonTile.scala,
// TLNBDCache.scala). The register only starts a flush and has no done bit, so wait on both sides:
// first for the other warps' last stores to reach the cache, then for the flush to write out.
constexpr uint32_t TF_L0D_FLUSH_ADDR = 0x00080300;
// One flush with 16384-poll waits left 1070/4096 q cells stale (ac28aa98, 2026-09-17), so this
// waits 8x longer and flushes twice: a store that had not reached the cache at the first flush is
// written out by the second.
#ifndef TF_L0D_FLUSH_WAIT
#define TF_L0D_FLUSH_WAIT 131072u
#endif
#ifndef TF_L0D_FLUSHES
#define TF_L0D_FLUSHES 2u
#endif
// Measured: 1 flush / 16k-poll waits -> q 3026/4096; 2 flushes / 131k-poll waits -> 4034/4096 (11.7M
// cycles). Timed waits never fully close the window, so the default is the D-cache fence, which is a real
// done signal (SFUPipe.scala): it waits for every warp's load/store queues to empty, starts the L0D
// flush, and resumes only on the cache's flush-done. The core has ONE fence.d slot (not one per warp),
// so it must be issued by a single warp -- that is why mu_fence() in TF_BAR (every warp) hung. This is
// the flash_attention_mx_fp6 / mu_schedule write-drain pattern, from warp 0 only.
// -DTF_L0D_MMIO_FLUSH selects the timed MMIO flush instead.
static __attribute__((noinline)) void tf_l0d_flush() {
#ifdef TF_L0D_MMIO_FLUSH
    for (uint32_t i = 0; i < TF_L0D_FLUSH_WAIT; i++) (void)load32_shared(GEMMINI_BUSY_ADDR);
    for (uint32_t f = 0; f < TF_L0D_FLUSHES; f++) {
        store_shared(TF_L0D_FLUSH_ADDR, 0, (uint32_t)1);
        for (uint32_t i = 0; i < TF_L0D_FLUSH_WAIT; i++) (void)load32_shared(GEMMINI_BUSY_ADDR);
    }
#else
    mu_fence();
#endif
}

// Scale bytes GMEM -> scale SRAM as 4-byte stores, single thread, ascending (rule 6).
template <uint32_t NB>
static __attribute__((noinline)) void tf_load_scales(volatile __shared uint32_t *sf,
                                                     const uint8_t *src) {
    static_assert(NB % 32 == 0, "scale runs must be whole 8-word groups (even, 8B-aligned)");
    const uint32_t *w = reinterpret_cast<const uint32_t *>(src);
    constexpr uint32_t ILP = 8;
    uint32_t unrolled[ILP];
    #pragma unroll 4
    for (uint32_t i = 0; i < NB / 4; i += ILP) {
        #pragma unroll
        for (uint32_t j = 0; j < ILP; j++) unrolled[j] = w[i + j];
        for (uint32_t j = 0; j < ILP; j++) sf[i + j] = unrolled[j];
    }
}

#ifdef TF_DIAG_UNIT_SCALES
// DIAGNOSTIC ONLY (-DTF_DIAG_UNIT_SCALES): every GEMM loads e8m0 scale 127 (2^0) for A and B, so
// the combined scale Gemmini applies is exactly 0 everywhere. Separates "the scale SRAM returns the
// wrong bytes" (output becomes finite and matches the unit-scale golden) from a scale-independent
// accumulator/store fault (the overflow/zero pattern stays). Golden: q_scale_analysis.py --unit-scales.
constexpr uint32_t TF_UNIT_SCALE_BYTES = 512;
static uint8_t tf_unit_scales[TF_UNIT_SCALE_BYTES] __attribute__((aligned(32)));
#endif

// One-time setup: every GEMM in the layer is FP8 x FP8 -> bf16, so config_ex never changes.
static __attribute__((noinline)) void tf_setup_once() {
#ifdef TF_DIAG_UNIT_SCALES
    for (uint32_t i = 0; i < TF_UNIT_SCALE_BYTES; i++) tf_unit_scales[i] = 127;
#endif
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                                GEMMINI_FORMAT_FP8, GEMMINI_FORMAT_FP8, GEMMINI_FORMAT_FULL,
                                false);
    tf_gf();
}

// Warp 0 only, and only between two barriers: C[M][N] = A @ B, left in SMEM at G.C_ROW().
// A/B are fp8 in GMEM with row strides K/N; As/Bs are their e8m0 scales.
template <TfGemm G>
static __attribute__((noinline)) void tf_gemm_serial(const uint8_t *A, const uint8_t *As,
                                                     const uint8_t *B, const uint8_t *Bs) {
    static_assert(G.VALID(), "GEMM shape does not fit the SMEM map / scale SRAM (see TfGemm)");
    tf_phase_skew();
    tf_drain();                                     // rule 2: previous sweep complete

    // rule 9: RESET GEMMINI BEFORE EVERY GEMM. lib/mxgemm's configure_mxgemmini() -- the library
    // the working gemm kernel uses -- opens with gemmini_flush(0), and gm_sim's testbench issues
    // a FLUSH before every case; transformer_mx never did. Draining on io.busy only waits for the
    // current sweep, it does not clear the accelerator's loop state, so that state accumulated
    // across GEMM phases: at H=512 the run survived N weight-GEMM phases and wedged on N+1.
    // Proof (2026-09-21): FF1 GEMM + FF2 GEMM hangs; either one ALONE completes; and swapping
    // FF2's weights, A operand and destination each changed nothing -- it was never FF2.
    gemmini_flush(0);
#ifdef TF_INBLOCK_FENCE
    tf_l0d_flush();                                 // option 2: warp-0 fence inside the block (3155/4096)
#endif

    // rule 7: operand move-in through Gemmini's loop DMA -- one LOOP_WS with ex/stc skipped, the
    // same sequence lib/mxgemm's copy_gmem_to_smem_async (GEMMINI_DMA) issues in the kernels that
    // pass on this FPGA. A lands from row 0 with row stride K; B ends at the SMEM end, stride N.
    //
    // CONFIG_LD first, for A (load state 0) and B (state 1), exactly as lib/mxgemm's
    // configure_mxgemmini does. It is the only command that sets LoadController's block stride
    // (block_mvin_stride = DIM, the SMEM row step inside a move-in block). Without it the stride
    // keeps its power-up value 0, every row of a DIM-row block lands on one SMEM row, and the
    // mesh multiplies stale rows: rung 1 came back 1/4096 on the FPGA (2026-09-17), reproduced
    // exactly in gm_sim, where the unsynthesized RTL also trips LoadController's
    // `assert(block_stride >= rows)`. Sent per GEMM so every shape gets its own K / N strides.
    gemmini_extended3_config_ld(G.K * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(G.N * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
        rad_device_to_host_address(reinterpret_cast<uint32_t>(A)),
        rad_device_to_host_address(reinterpret_cast<uint32_t>(B)), k_LOOP_WS_CONFIG_ADDRS_AB)
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)G.K, (uint64_t)G.N, k_LOOP_WS_CONFIG_STRIDES_AB)
    gemmini_loop_ws_spad(G.PE_I(), G.PE_J(), G.PE_K(), 0, 0, 0,
                         G.A_ROW(), G.B_END(), /*D=*/0, /*C=*/0,
                         false, false, false, false, /*ex_accumulate=*/false,
                         NO_ACTIVATION, 0, 0, false,
                         loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                                           /*skip_ex=*/1, /*skip_stc=*/1));
    tf_drain();                                     // a LOOP_WS: runningLoops, then busy

#ifdef TF_DIAG_UNIT_SCALES
    // Only the QK GEMM (32x32x32). Forcing unit scales on the projections too would corrupt q0/k0
    // and hence the QK operands, so the unit-scale golden would no longer describe this run.
    constexpr bool kUnitScales = (G.M == 32 && G.N == 32 && G.K == 32);
    static_assert(!kUnitScales || (G.A_SCALE_BYTES() <= TF_UNIT_SCALE_BYTES &&
                                   G.B_SCALE_BYTES() <= TF_UNIT_SCALE_BYTES));
    const uint8_t *const As_load = kUnitScales ? tf_unit_scales : As;
    const uint8_t *const Bs_load = kUnitScales ? tf_unit_scales : Bs;
#else
    const uint8_t *const As_load = As;
    const uint8_t *const Bs_load = Bs;
#endif
#ifndef TF_NO_SCALE_WRITES
    tf_load_scales<G.A_SCALE_BYTES()>(
        reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_SF_MEM_A), As_load);
    tf_load_scales<G.B_SCALE_BYTES()>(
        reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_SF_MEM_B), Bs_load);
#else
    // TEST 3 (2026-09-21), HANG TEST ONLY -- the math is meaningless without scales.
    // Gemmini's command path is clean for 32 back-to-back GEMMs in gm_sim, and 6 extra
    // Gemmini-free phases do not hang (test 1), so the accumulation is specific to GEMM
    // phases. The only part of those gm_sim cannot see is this write path: the GPU stores
    // 4-byte words that FlitMergeNode merges in PAIRS using a running counter, and its
    // ordering asserts are compiled out on the FPGA. If dropping these writes clears the
    // hang, that path is the accumulator.
    (void)As_load; (void)Bs_load;
#endif
    mu_fence_smem();                                // rule 5
    tf_gf();

    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&tf_sf_dummy[0])),
        G.PE_I(), G.PE_J(), G.PE_K(), /*act_sel=*/0, /*w_sel=*/0, /*lut_granularity=*/1);
    tf_gf();                                        // rule 3

    gemmini_loop_ws_spad(G.PE_I(), G.PE_J(), G.PE_K(), 0, 0, 0,
                         G.A_ROW(), G.B_END(), /*D=*/0, G.C_ROW(),
                         false, false, false, false, /*ex_accumulate=*/false,
                         NO_ACTIVATION, 0, 0, false,
                         loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                                           /*skip_ex=*/0, /*skip_stc=*/0));
    tf_drain();                                     // rule 4: matmul + accumulator move-out
}

// All threads, after the barrier that follows tf_gemm_serial: C (SMEM, bf16 packed two per word)
// -> DRAM C_gmem[:, col0 : col0 + N] of a [M][fullN] buffer. One thread per row, so no two
// threads ever write halves of the same 32-bit word.
template <TfGemm G>
static __attribute__((noinline)) void tf_store_c(uint16_t *C_gmem, uint32_t col0, uint32_t fullN,
                                                 uint32_t tid, uint32_t thr) {
    constexpr uint32_t NW = G.N / 2;
    const __shared uint32_t *src = reinterpret_cast<const __shared uint32_t *>(G.C_ROW() * DIM);
    // 32-bit stores, not two 16-bit halves. The source is already a 32-bit word and the
    // destination is 4-byte aligned (col0 and fullN are multiples of 32, buffers are 32-byte
    // aligned), so the split served no purpose -- it just doubled the store count and made every
    // write a SUB-WORD store, i.e. partial-line dirty state in the L0 D-cache.
    // Why this is the suspect (2026-09-21): removing tf_store_c entirely clears the H=512 hang
    // (test 4) while everything else inside a gemm_stage is eliminated -- Gemmini runs 256
    // LOOP_WS back to back in gm_sim with no degradation, dropping the scale writes still hangs,
    // and Gemmini-free phases do not hang. The failure tracks the gemm_stage count (56 ok, 64
    // hangs), and this loop is the only per-gemm_stage DRAM traffic that scales with it.
    for (uint32_t r = tid; r < G.M; r += thr) {
        uint32_t *const dst = reinterpret_cast<uint32_t *>(C_gmem + r * fullN + col0);
        for (uint32_t w = 0; w < NW; w++) dst[w] = src[r * NW + w];
    }
}

#endif // TF_GEMM_HPP
