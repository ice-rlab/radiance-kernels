// MXFP8 transformer encoder layer (BERT-style), Radiance mesh + muon SIMT -- built up one rung at a time.
// ===================================================================================
// Bring-up ladder (mirrors golden_tf.py:encoder_layer). A rung is added to this file only after every
// rung already in it passes on the FPGA against include/tf_golden.h:
//   rung 1 (in this file): Q = X @ Wq + bq   mxquant_A(X) on SIMT, Wq GEMM tiles on Gemmini, bias on SIMT
//   rung 2: K, V projections              rung 3: head 0 transpose, Q@K^T (Gemmini), softmax
//   rung 4: all heads, S@V (Gemmini)      rung 5: Wo + bias, LN1, residual
//   rung 6: FF1 + bias, GELU              rung 7: FF2 + bias, LN2, residual (full layer)
//
// Stage discipline, copied from ../flash_attention_mx_stable (tf_gemm.hpp rules 1-8):
//   * every mesh GEMM is issued AND drained by warp 0 inside one stage, with every other warp
//     parked at the barrier;
//   * SIMT never touches SMEM except to copy a drained C tile out, in its own stage;
//   * every SIMT stage ends in fence + barrier before anything reads what it wrote;
//   * all epilogue arithmetic is native bf16 (transformer_mx_impl.hpp).
// Activations are bf16 codes in DRAM; host.cpp checks each rung's buffer (address from tf_syms.h).
// ===================================================================================
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/tf_data.h"
#include "tf_beacon.h"
#include "tf_gemm.hpp"
#include "transformer_mx_impl.hpp"

constexpr uint32_t M  = TF_SEQ;
constexpr uint32_t H  = TF_HIDDEN;
constexpr uint32_t TN = TF_TN;
constexpr uint32_t S  = TF_SEQ;        // sequence length (attention rows/cols)
constexpr uint32_t HD = TF_HEAD_DIM;   // per-head width
constexpr uint32_t NH = H / HD;        // heads (rung 4)
constexpr uint32_t E  = TF_EXP;        // FF expansion width (rung 6)

static_assert(H % TN == 0);
static_assert(sizeof(Wq) == H * TN * (H / TN), "tf_data.h does not match the TF_* dims");
static_assert(sizeof(Wk) == sizeof(Wq) && sizeof(Wv) == sizeof(Wq), "Wk / Wv shape differs from Wq");

// M-TILING (2026-09-27). At TF_SEQ=128 the projection shape TfGemm{128,64,768} needs
// A(12288) + C(2048) = 14336 scratchpad rows against B_END(16384) - B(6144) = 10240 -- it does not
// fit, and the VALID() static_assert rejects it. So the GEMMs run MT rows at a time.
//   * the A fp8 operand is [M][K] row-major, so an M-tile is just a pointer offset (stride stays K)
//   * the A SCALES are [K/32][M] with M INNERMOST, so an M-tile of them would be a strided column
//     slice -- and tf_load_scales() does a flat contiguous copy of M*K/32 bytes. Quantizing PER
//     TILE instead writes [K/32][MT] contiguously at t*(K/32)*MT, which is exactly what the scale
//     SRAM wants. Total size is unchanged: NMT*GK*MT == GK*M, so no buffer grows.
// At TF_SEQ <= 64, MT == M and NMT == 1, so every helper below collapses to the single call this
// kernel made before -- the seq=32 configuration is bit-identical.
#ifndef TF_MT
#define TF_MT ((TF_SEQ > 64) ? 64u : (uint32_t)TF_SEQ)
#endif
constexpr uint32_t MT  = TF_MT;
constexpr uint32_t NMT = M / MT;
static_assert(M % MT == 0, "TF_SEQ must be a whole number of M-tiles");
static_assert(MT % 4 == 0, "mxquant_A needs MT % 4 == 0");

// One GEMM shape per distinct (M, N, K); each output-column tile of a weight is its own GEMM.
constexpr TfGemm PROJ{ .M = MT, .N = TN, .K = H };   // Wq / Wk / Wv
constexpr TfGemm QK{ .M = S, .N = S, .K = HD };    // q_h @ k_h^T (one head)
constexpr TfGemm SV{ .M = S, .N = HD, .K = S };   // softmax(q_h@k_h^T) @ v_h (one head)
constexpr TfGemm FF2G{ .M = MT, .N = TN, .K = E }; // ff @ FF2 (rung 7): K = E
// Every GEMM writes at most FFW columns per call (FFW/TN = 2 N-tiles). A single call with
// fullN = 256 (four tiles, 256-wide store pitch) hung the GPU; two 128-wide calls did not.
// FF1 (N = E) and FF2 (N = H) are therefore issued in E/FFW and H/FFW chunks.
constexpr uint32_t FFW = 128;
static_assert(E % FFW == 0 && H % FFW == 0 && FFW % TN == 0, "FFW must tile E and H");

#define TFA __attribute__((aligned(32)))
// ---- DRAM activations (bf16) ----
// Buffers the host checks (addresses via tf_syms.h) have EXTERNAL linkage: a static buffer that is
// written but never read again on the device is deleted by the compiler together with its writes.
uint16_t g_q[M][H] TFA;
uint16_t g_k[M][H] TFA;
uint16_t g_v[M][H] TFA;
uint16_t g_s[S][S] TFA;                            // head 0 scores, softmaxed (rung 3)
// ---- head-0 working buffers (bf16) ----
uint16_t g_q0[S][HD] TFA;                          // external: host intermediate checks
uint16_t g_k0[S][HD] TFA;
uint16_t g_kt[HD][S] TFA;
// ---- head-0 GEMM operands (fp8 + e8m0) ----
uint8_t g_q0f8[S * HD] TFA;                        // external: host operand check
uint8_t g_q0fs[(HD / 32) * S] TFA;
uint8_t g_ktf8[HD * S] TFA;
uint8_t g_kts[(HD / 32) * S] TFA;
uint16_t g_sraw[S][S] TFA;                         // scores before softmax (host check)
// ---- rung 4: all heads + S@V ----
uint16_t g_ctx[S][H] TFA;                          // external: attention context, all heads
uint16_t g_s0[S][S] TFA;                           // external: head 0 softmax, kept for the rung-3
                                                   // check (g_s is reused by every later head)
uint16_t g_v0[S][HD] TFA;                          // per-head V
uint8_t g_pf8[S * S] TFA;                          // A = softmax probs [S][S]
uint8_t g_pfs[(S / 32) * S] TFA;
uint8_t g_v0f8[S * HD] TFA;                        // B = v_h [S][HD]
uint8_t g_v0fs[(S / 32) * HD] TFA;
// ---- rung 5: Wo + bias, LN1, residual ----
uint16_t g_attn[M][H] TFA;                         // ctx @ Wo + bo, then LN1 (working buffer)
uint16_t g_res[M][H] TFA;                          // external: X + LN1(...) -- the rung-5 check
uint8_t g_ctxf8[M * H] TFA;                        // A = ctx fp8, K = H
uint8_t g_ctxfs[(H / 32) * M] TFA;
// ---- rung 6: FF1 + bias, GELU ----
uint16_t g_ff[M][E] TFA;                           // external: GELU(res @ FF1 + ff1_b)
// Rung 6 restructured (2026-09-18) to match the projections exactly: every GEMM that works on this
// FPGA writes a [M][H] destination with fullN = H = 128 (two N-tiles). The single fullN = E = 256
// call (four tiles, 256-wide store pitch) is the only GEMM shape the kernel had never run, and it
// is the step the bisection blamed. These halves restore the proven shape.
uint16_t g_ffa[M][FFW] TFA;                        // FF1 output chunk 0
uint16_t g_ffb[M][FFW] TFA;                        // FF1 output chunk 1
// ---- rung 7: FF2 + bias, LN2, residual ----
uint16_t g_out[M][H] TFA;                          // external: LN2(ff @ FF2 + ff2_b) + res
uint16_t g_outa[M][FFW] TFA;                       // FF2 output chunk 0
uint16_t g_outb[M][FFW] TFA;                       // FF2 output chunk 1
uint8_t g_fff8[M * E] TFA;                         // A = ff fp8, K = E
uint8_t g_ffs[(E / 32) * M] TFA;
uint8_t g_resf8[M * H] TFA;                        // A = res fp8, K = H
uint8_t g_resfs[(H / 32) * M] TFA;
// What the mesh actually read: the A and B tiles as they sit in SMEM after the q@k^T move-in.
uint8_t g_asmem[S * HD] TFA;
uint8_t g_bsmem[HD * S] TFA;
// ---- DRAM operands (fp8 + e8m0) ----
uint8_t g_af8[M * H] TFA;                          // A = X fp8, K = H (external: host A check)
uint8_t g_afs[(H / 32) * M] TFA;                   // A scales [K/32][M] (external: host A check)

// ---- barriers ----
// Register-only ALU pad around each barrier: the barrier release is a single-cycle unbuffered
// Valid pulse (Synchronizer.sv), and a retiring commit between back-to-back stalls restores slack.
// `_q` must NOT be volatile -- a volatile local is a stack slot, and the stack is DRAM.
#define TF_PAD() do { int _q = 0;                  \
    asm volatile("addi %0,%0,1" : "+r"(_q));       \
    asm volatile("addi %0,%0,1" : "+r"(_q));       \
    asm volatile("addi %0,%0,1" : "+r"(_q));       \
    asm volatile("addi %0,%0,1" : "+r"(_q)); } while (0)
// Barrier ids are 4 bits (MuonCore barrierBits = 4): an id >= 16 silently truncates.
// fence.s only. DO NOT add the general mu_fence() (the D-cache flush) here: it hangs at the FIRST
// barrier, at occupancy 2 and 8 alike (2026-09-12, and again 2026-09-17 with 4c7ed8a3): the core has one
// fence.d slot, so a fence from every warp at once never resumes the others. The D cache is fenced from
// main() between phases instead (tf_write_drain).
#define TF_BAR(id) do { static_assert((id) < 16u);  \
    mu_fence_smem(); TF_PAD(); mu_barrier((id), wpb); TF_PAD(); } while (0)

// ---- beacon (tid 0, one GMEM store each). Every tid==0 block that precedes a barrier carries an
// explicit `else nop`: without it llvm may duplicate the barrier into both paths of the
// warp-divergent branch and deadlock (mu_intrinsics.h). -DTF_NO_BEACON compiles them out. ----
#ifndef TF_NO_BEACON
#define MARK(i) do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
        ((volatile uint32_t *)TF_MARK_GMEM)[i] = _c; } else { asm volatile("nop"); } } while (0)
// Row-level progress inside a phase: [6] = pass id, [7] = last row started. Five hypotheses for
// the H=768 rung-5 hang have now failed (fullN==K, PE_K, barrier ids, store count, per-phase dirty
// volume), so stop guessing -- let the watchdog report say which pass and which row it died on.
// If [7] never advances past 0 the phase never started; if it stops mid-row the stall is inside
// that pass. (2026-09-22)
#ifdef TF_PROGRESS_STAMPS
#define PROG(pass, row) do { if (tid == 0) {                                                   \
        ((volatile uint32_t *)TF_WHERE_GMEM)[6] = (uint32_t)(pass);                            \
        ((volatile uint32_t *)TF_WHERE_GMEM)[7] = (uint32_t)(row) + 1u; }                       \
        else { asm volatile("nop"); } } while (0)
#else
#define PROG(pass, row) do { (void)(pass); (void)(row); } while (0)
#endif

#define WHERE(id) do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
        ((volatile uint32_t *)TF_WHERE_GMEM)[1] = _c;                                          \
        ((volatile uint32_t *)TF_WHERE_GMEM)[0] = (uint32_t)(id) + 1u; }                       \
        else { asm volatile("nop"); } } while (0)
#define TF_DONE_STAMP() do { if (tid == 0) { ((volatile uint32_t *)TF_WHERE_GMEM)[5] = TF_DONE_MAGIC; } \
        else { asm volatile("nop"); } } while (0)
#else
#define MARK(i)         do { } while (0)
#define WHERE(id)       do { } while (0)
#define TF_DONE_STAMP() do { } while (0)
#endif

// ---- one GEMM = two stages ----
//   stage a: warp 0 moves in, loads scales, configures, computes and drains; all others parked.
//   stage b: every thread copies the drained C tile SMEM -> DRAM; the mesh is idle.
template <TfGemm G>
static __attribute__((noinline)) void gemm_stage(const uint8_t *A, const uint8_t *As,
                                                 const uint8_t *B, const uint8_t *Bs,
                                                 uint16_t *C, uint32_t col0, uint32_t fullN,
                                                 uint32_t tid, uint32_t thr) {
    const uint32_t wpb = thr / MU_NUM_THREADS;
    if (tid == 0) { tf_gemm_serial<G>(A, As, B, Bs); } else { asm volatile("nop"); }
    TF_BAR(1);
#ifndef TF_NO_STORE_C
    tf_store_c<G>(C, col0, fullN, tid, thr);
#ifdef TF_FLUSH_PER_TILE
    // Bound the dirty state BETWEEN flushes, not per phase. tf_write_drain only runs at phase
    // boundaries, but one phase holds many gemm_stage calls -- gemm_body is 3 GEMMs x fullN/TN
    // tiles, i.e. 24,576 word-stores at H=512 but 36,864 at H=768. The tf_store_c fix (2 x 16-bit
    // -> 1 x 32-bit) halved the count and made H=512 pass; H=768 crosses the same threshold again
    // (65,536 words ok, 98,304 hangs), and raising TF_WB_WAIT to 131072 does NOT help -- so it is
    // the dirty-line COUNT between flushes, not the flush duration (2026-09-22).
    if (tid == 0) {
        for (uint32_t f = 0; f < TF_L0D_FLUSHES; f++) {
            store_shared(TF_L0D_FLUSH_ADDR, 0, (uint32_t)1);
            for (uint32_t i = 0; i < TF_WB_WAIT; i++) (void)load32_shared(GEMMINI_BUSY_ADDR);
        }
    } else {
        asm volatile("nop");
    }
#endif
#else
    // TEST 4 (2026-09-21), HANG TEST ONLY -- C is never copied out, so the math is meaningless.
    // Everything else inside a gemm_stage is eliminated: Gemmini's command stream runs 256
    // LOOP_WS back to back in gm_sim with no degradation (4x the FPGA threshold), the scale
    // writes can be removed entirely and it still hangs, and Gemmini-free phases do not hang.
    // tf_store_c is the only remaining per-gemm_stage work -- GPU reads of SMEM feeding DRAM
    // writes -- and it scales exactly with the gemm_stage count that predicts the failure.
    (void)C; (void)col0; (void)fullN;
#endif
    TF_BAR(2);
}

// C[M][fullN] = A @ W for a static weight emitted N-tiled by gen_data.py: block nj is
// W[nj*K : (nj+1)*K][TN], scales Ws[nj*GK : (nj+1)*GK][TN].
template <TfGemm G>
static inline void gemm_weight(const uint8_t *A, const uint8_t *As,
                               const uint8_t *W, const uint8_t *Ws,
                               uint16_t *C, uint32_t fullN, uint32_t tid, uint32_t thr) {
    constexpr uint32_t GK = G.K / 32;
    for (uint32_t nj = 0; nj < fullN / G.N; nj++)
        gemm_stage<G>(A, As, W + nj * G.K * G.N, Ws + nj * GK * G.N,
                      C, nj * G.N, fullN, tid, thr);
}

// Quantize A one M-tile at a time, so each tile's scales land contiguous as [K/32][MT].
template <uint32_t K>
static inline void mxquant_A_m(const uint16_t *src, uint8_t *fp8, uint8_t *scales,
                               uint32_t tid, uint32_t thr) {
    constexpr uint32_t GK = K / 32;
    for (uint32_t t = 0; t < NMT; t++)
        mxquant_A<MT, K>(src + t * MT * K, fp8 + t * MT * K, scales + t * GK * MT, tid, thr);
}

// C[M][fullN] = A @ W, issued NMT tiles of G.M rows. C is [M][fullN], so tile t starts at row
// t*G.M, i.e. C + t*G.M*fullN; tf_store_c's row stride is fullN either way.
template <TfGemm G>
static inline void gemm_weight_m(const uint8_t *A, const uint8_t *As,
                                 const uint8_t *W, const uint8_t *Ws,
                                 uint16_t *C, uint32_t fullN, uint32_t tid, uint32_t thr) {
    constexpr uint32_t GK = G.K / 32;
    for (uint32_t t = 0; t < NMT; t++)
        gemm_weight<G>(A + t * G.M * G.K, As + t * GK * G.M, W, Ws,
                       C + t * G.M * fullN, fullN, tid, thr);
}

// ---- phases: one mu_schedule per phase, with a write-drain between them in main() ----
// Modelled on flash_attention_mx_fp6 and mu_schedule(): GPU stores to DRAM can sit in the core's L0 D
// cache, while Gemmini's DMA move-in and the host read DRAM directly. The D-cache fence (mu_fence) is
// the done signal for that, but the core has a single fence.d slot, so it is issued only from main()
// (warp 0), after a barrier, between phases. Each phase body ends at a barrier all its threads reach,
// so the manager thread does not return to main() while workers still run.
// 2026-09-17: with the fences inside one mu_schedule, rung 1 read stale g_af8 in the move-in
// (q 2238..4034/4096 while the host read A back exact).

static void setup_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_SETUP);
    if (tid == 0) {
        tf_setup_once();
#ifndef TF_NO_BEACON
        ((volatile uint32_t *)TF_WHERE_GMEM)[3] = M;
        ((volatile uint32_t *)TF_WHERE_GMEM)[4] = H;
#endif
    } else { asm volatile("nop"); }
    TF_BAR(3);
}

// ============================ rung 1: Q = X @ Wq + bq (rung 2 adds K, V) ============================
static void quant_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    MARK(0);
    WHERE(TFS_QUANT_X);
    mxquant_A_m<H>(&X_in[0][0], g_af8, g_afs, tid, thr);
    TF_BAR(4);
}

static void gemm_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    // All three projections use the quantized X drained into DRAM by the previous phase; each GEMM stage
    // leaves its output tile in SMEM and copies it into DRAM, so nothing here reads a DRAM buffer this
    // phase wrote. Each gemm_stage ends at TF_BAR(2).
    WHERE(TFS_GEMM_Q);
    gemm_weight_m<PROJ>(g_af8, g_afs, &Wq[0][0], &Wq_s[0][0], &g_q[0][0], H, tid, thr);
    // ============================ rung 2: K, V = X @ {Wk, Wv} + {bk, bv} ============================
    WHERE(TFS_GEMM_K);
    gemm_weight_m<PROJ>(g_af8, g_afs, &Wk[0][0], &Wk_s[0][0], &g_k[0][0], H, tid, thr);
    WHERE(TFS_GEMM_V);
    gemm_weight_m<PROJ>(g_af8, g_afs, &Wv[0][0], &Wv_s[0][0], &g_v[0][0], H, tid, thr);
}

// ==================== rung 3: head 0 -- gather q/k, k^T, q@k^T, softmax ====================
// Every buffer Gemmini's move-in reads (g_q0f8/g_q0fs, g_ktf8/g_kts) is written here and drained by
// main() before the GEMM phase; the GPU reading back its own DRAM writes inside one phase is fine.
static void head_prep_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    const uint32_t hd = (uint32_t)(uintptr_t)arg;   // head index (rung 4)
    WHERE(TFS_HD_GATHER);
    gather_cols<S, H, HD>(&g_q[0][0], &g_q0[0][0], hd * HD, tid, thr);
    gather_cols<S, H, HD>(&g_k[0][0], &g_k0[0][0], hd * HD, tid, thr);
    gather_cols<S, H, HD>(&g_v[0][0], &g_v0[0][0], hd * HD, tid, thr);
    TF_BAR(6);
    WHERE(TFS_HD_TRANSPOSE);
    transpose_g<S, HD>(&g_k0[0][0], &g_kt[0][0], tid, thr);
    TF_BAR(7);
    WHERE(TFS_HD_QUANT_QK);
    mxquant_A<S, HD>(&g_q0[0][0], g_q0f8, g_q0fs, tid, thr);   // A = q_h  [S][HD]
    mxquant_B<HD, S>(&g_kt[0][0], g_ktf8, g_kts, tid, thr);    // B = k_h^T [HD][S]
    TF_BAR(8);
}

static void qk_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_HD_GEMM_QK);
    gemm_stage<QK>(g_q0f8, g_q0fs, g_ktf8, g_kts, &g_s[0][0], 0, S, tid, thr);
}


static void softmax_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    const uint32_t hd = (uint32_t)(uintptr_t)arg;
    WHERE(TFS_HD_SOFTMAX);
    if (hd == 0) {                                  // head 0 only: the rung-3 host checks
        for (uint32_t r = tid; r < S; r += thr)
            for (uint32_t c = 0; c < S; c++) g_sraw[r][c] = g_s[r][c];
    } else {
        asm volatile("nop");
    }
    softmax_g<S, S>(&g_s[0][0], h_bits(TFB(TF_ATTN_SCALE)), tid, thr);
    if (hd == 0) {
        for (uint32_t r = tid; r < S; r += thr)
            for (uint32_t c = 0; c < S; c++) g_s0[r][c] = g_s[r][c];
    } else {
        asm volatile("nop");
    }
    TF_BAR(9);
}

// rung 4: quantize this head's softmax probs (A) and v_h (B) for the S@V GEMM. Separate phase so
// main()'s write-drain publishes both to DRAM before Gemmini's move-in reads them.
static void sv_prep_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_HD_QUANT_QK);
    mxquant_A<S, S>(&g_s[0][0], g_pf8, g_pfs, tid, thr);      // A = P   [S][S]
    mxquant_B<S, HD>(&g_v0[0][0], g_v0f8, g_v0fs, tid, thr);  // B = v_h [S][HD]
    TF_BAR(11);
}

// ==================== rung 5: attn out = ctx @ Wo + bo, LN1, residual ====================
static void o_prep_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_QUANT_X);
    mxquant_A_m<H>(&g_ctx[0][0], g_ctxf8, g_ctxfs, tid, thr);   // A = ctx [M][H]
    TF_BAR(13);
}

static void o_gemm_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_GEMM_Q);
    gemm_weight_m<PROJ>(g_ctxf8, g_ctxfs, &Wo[0][0], &Wo_s[0][0], &g_attn[0][0], H, tid, thr);
}

// Rung-5 epilogue, SPLIT into one phase per pass (2026-09-22). It used to do bias + LN1 +
// residual in a single phase: at H=768 that is 3 x 32x768 = 73,728 bf16 stores with no flush
// between them, vs 49,152 at H=512 (which passed). Rungs 3-4 issue far MORE stores in total
// (~73,728 in gather/transpose alone) but spread over ~60 phases, and they pass -- so what
// matters is dirty state BETWEEN flushes, not the global count. main() drains after each phase,
// so splitting caps the dirty volume per flush at one pass regardless of H.
static void o_bias_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_BIAS_Q);
    PROG(1, 0);
    for (uint32_t r = tid; r < M; r += thr) {
        PROG(1, r);
        bias_add_row<H>(&g_attn[r][0], bo);
    }
    TF_BAR(14);
}

static void o_ln_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    PROG(2, 0);
    for (uint32_t r = tid; r < M; r += thr) {
        PROG(2, r);
        layernorm_row<H>(&g_attn[r][0], ln1_g, ln1_b, h_bits(TFB(TF_LN_EPS)));
    }
    TF_BAR(15);
}

static void o_res_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    PROG(3, 0);
    for (uint32_t r = tid; r < M; r += thr) {
        PROG(3, r);
        residual_add_row<H>(&g_res[r][0], &X_in[r][0], &g_attn[r][0]);
    }
    // Barrier id 10, NOT 0. TF_BAR(id) is mu_barrier(id, wpb) where wpb == TF_OCCUPANCY, while
    // main()'s tf_write_drain() uses mu_barrier(0, MU_NUM_CORES) and MU_NUM_CORES is 1. Using id 0
    // here therefore arms barrier 0 for TF_OCCUPANCY participants and then main re-uses the SAME
    // id expecting 1 -- the counts only agree at occupancy 1. That matches the observed occupancy
    // dependence exactly: at H=768, rung 6 hangs entering the FF1 GEMM at occupancy 2 but runs to
    // DONE at occupancy 1, and occupancy 8 was already known to hang. Rung 5 alone survived at
    // occupancy 2 only because it ends a few phases later. Id 10 is the one id no body used.
    // (2026-09-23)
    TF_BAR(10);
}

// ==================== rung 6: ff = GELU(res @ FF1 + ff1_b) ====================
static void ff_prep_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_QUANT_X);
    mxquant_A_m<H>(&g_res[0][0], g_resf8, g_resfs, tid, thr);
    TF_BAR(3);
}

// FF1 is emitted as 4 blocks of [K][TN]; gemm_weight walks blocks 0..fullN/TN-1 from its W pointer,
// so the second half starts 2 blocks in (2*K*TN bytes of codes, 2*(K/32)*TN of scales).
static void ff_gemm_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_GEMM_K);
#ifdef TF_FF1_SINGLE
    // Test (2026-09-21): every observed hang has fullN != PROJ.K, every pass has fullN == K:
    //   K=128 fullN=128 OK | K=128 fullN=256 hang | K=256 fullN=256 OK (Wo) | K=256 fullN=128 hang
    // So issue FF1 as ONE fullN = E call into g_ff, exactly as rung 5 issues Wo.
    gemm_weight_m<PROJ>(g_resf8, g_resfs, &FF1[0][0], &FF1_s[0][0], &g_ff[0][0], E, tid, thr);
#else
    gemm_weight_m<PROJ>(g_resf8, g_resfs, &FF1[0][0], &FF1_s[0][0], &g_ffa[0][0], FFW, tid, thr);
#endif
}

static void ff_gemm_body_b(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_GEMM_V);
    // FF1 is FF1[512][64] = 4 blocks of [K][TN] stacked by ROW, and FF1_s is FF1_s[16][64] =
    // 4 blocks of [K/32][TN]. Block 2 therefore starts at row 2*K (not 2*K*N -- that was an
    // element offset applied to a row index, which read far out of bounds and made ff columns
    // 128..255 non-finite while 0..127 were exact, 2026-09-18).
    gemm_weight_m<PROJ>(g_resf8, g_resfs, &FF1[(FFW / TN) * PROJ.K][0],
                      &FF1_s[(FFW / TN) * (PROJ.K / 32)][0], &g_ffb[0][0], FFW, tid, thr);
}

static void ff_post_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_BIAS_K);
#ifdef TF_FF1_SINGLE
    // FF1 wrote g_ff directly (one fullN = E call, the form that completes at H=256), so the
    // epilogue runs over the full E-wide row and there are no halves to assemble.
    bias_add_g<M, E>(&g_ff[0][0], ff1_b, tid, thr);
    TF_BAR(4);
#if TF_RUNG6_STEP >= 4 && TF_RUNG6_STEP != 5
    gelu_g<M, E>(&g_ff[0][0], tid, thr);
    TF_BAR(6);
#endif
#else
    bias_add_g<M, FFW>(&g_ffa[0][0], ff1_b, tid, thr);
    bias_add_g<M, FFW>(&g_ffb[0][0], ff1_b + FFW, tid, thr);
    TF_BAR(4);
#if TF_RUNG6_STEP >= 4 && TF_RUNG6_STEP != 5
    gelu_g<M, FFW>(&g_ffa[0][0], tid, thr);
    gelu_g<M, FFW>(&g_ffb[0][0], tid, thr);
    TF_BAR(6);
#endif
    for (uint32_t r = tid; r < M; r += thr) {       // assemble the [M][E] buffer the host checks
        for (uint32_t c = 0; c < FFW; c++) g_ff[r][c] = g_ffa[r][c];
        for (uint32_t c = 0; c < FFW; c++) g_ff[r][FFW + c] = g_ffb[r][c];
    }
#endif
    TF_BAR(7);
}

// ==================== rung 7: out = LN2(ff @ FF2 + ff2_b) + res ====================
// FF2 is FF2[512][64] = 2 blocks of [K=256][TN=64], so fullN = H = 128 is two N-tiles -- already
// the shape every working GEMM uses; no split needed (unlike FF1, which had four).
static void out_prep_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_QUANT_X);
    mxquant_A_m<E>(&g_ff[0][0], g_fff8, g_ffs, tid, thr);
    TF_BAR(8);
}

static void out_gemm_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_GEMM_Q);
#ifdef TF_FF2_SINGLE
    // Discriminators for the H=512 rung-7 hang (2026-09-21). rung 6's FF1 GEMM is structurally
    // identical (same descriptor class, K, fullN, single call, destination width) and PASSES;
    // rung 7's FF2 GEMM hangs. Only two things differ: the weight array and the A operand.
    //   TF_FF2_DIAG_W  = FF2 GEMM with FF1's weights   (wrong math; isolates the weight array)
    //   TF_FF2_DIAG_A  = FF2 GEMM with res as operand  (wrong math; isolates the A operand)
#if defined(TF_FF2_DIAG_D)
    // W (FF1 weights) and A (res operand) BOTH hang, so neither the weight data nor the A operand
    // explains it. The destination is the remaining difference from rung 6's passing FF1 GEMM:
    // that writes g_ff, this writes g_out. Same size, same alignment, different address.
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF2[0][0], &FF2_s[0][0], &g_ff[0][0], H, tid, thr);
#elif defined(TF_FF2_DIAG_S)
    // W (FF1 codes + FF1 scales) COMPLETED, so the ff-derived A operand is fine and the FF2
    // weight data is implicated -- but W swapped codes AND scales. This splits them: FF1 codes
    // with FF2's SCALES. FF2_s is the only operand above 0x10200000 (2026-09-21).
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF1[0][0], &FF2_s[0][0], &g_out[0][0], H, tid, thr);
#elif defined(TF_FF2_DIAG_C)
    // Complement: FF2's CODES with FF1's scales.
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF2[0][0], &FF1_s[0][0], &g_out[0][0], H, tid, thr);
#elif defined(TF_FF2_DIAG_W)
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF1[0][0], &FF1_s[0][0], &g_out[0][0], H, tid, thr);
#elif defined(TF_FF2_DIAG_A)
    gemm_weight_m<FF2G>(g_resf8, g_resfs, &FF2[0][0], &FF2_s[0][0], &g_out[0][0], H, tid, thr);
#else
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF2[0][0], &FF2_s[0][0], &g_out[0][0], H, tid, thr);
#endif
#else
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF2[0][0], &FF2_s[0][0], &g_outa[0][0], FFW, tid, thr);
#endif
}

static void out_gemm_body_b(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    WHERE(TFS_GEMM_K);
    gemm_weight_m<FF2G>(g_fff8, g_ffs, &FF2[(FFW / TN) * FF2G.K][0],
                      &FF2_s[(FFW / TN) * (FF2G.K / 32)][0], &g_outb[0][0], FFW, tid, thr);
}

static void out_post_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_BIAS_V);
#ifndef TF_FF2_SINGLE
    for (uint32_t r = tid; r < M; r += thr) {       // chunks -> the H-wide row LN2 needs
        for (uint32_t c = 0; c < FFW; c++) g_out[r][c] = g_outa[r][c];
        for (uint32_t c = 0; c < FFW; c++) g_out[r][FFW + c] = g_outb[r][c];
    }
    TF_BAR(12);                                     // only needed to publish the assembled row
#endif
    // Shaped exactly like rung 5's o_post_body, which does the same three ops at the same width
    // and completes at H=512: same barrier ids, same order, no leading barrier. rung 7 hung here
    // while step 2 (quantize + FF2 GEMM) completed, so the epilogue is what differs (2026-09-21).
    bias_add_g<M, H>(&g_out[0][0], ff2_b, tid, thr);
    TF_BAR(14);
    layernorm_g<M, H>(&g_out[0][0], ln2_g, ln2_b, h_bits(TFB(TF_LN_EPS)), tid, thr);
    TF_BAR(15);
    residual_add_g<M, H>(&g_out[0][0], &g_out[0][0], &g_res[0][0], tid, thr);
    TF_BAR(10);   // see o_res_body: id 0 collides with main()'s drain barrier
}

// Dummy phase: a mu_schedule + drain that issues NO Gemmini work at all. Added to a known-good
// build to separate "Gemmini state accumulates across GEMM phases" from "phase/scheduler state
// accumulates across mu_schedule phases" -- the hang is cumulative (removing any one GEMM phase
// fixes it) but gemmini_flush(0) per GEMM does not clear it (2026-09-21).
static void dummy_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    TF_BAR(5);
}

static void done_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    MARK(1);                                        // post-QKV / attention
    WHERE(TFS_DONE);
    TF_DONE_STAMP();
    TF_BAR(12);
}

// rung 4: ctx[:, hd*HD : (hd+1)*HD] = P @ v_h. gemm_stage writes C straight into the right columns.
static void sv_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)tb_id;
    const uint32_t hd = (uint32_t)(uintptr_t)arg;
    WHERE(TFS_HD_GEMM_QK);
    gemm_stage<SV>(g_pf8, g_pfs, g_v0f8, g_v0fs, &g_ctx[0][0], hd * HD, H, tid, thr);
}

static void bias_body(void *arg, uint32_t tid, uint32_t thr, uint32_t tb_id) {
    (void)arg; (void)tb_id;
    const uint32_t wpb = thr / MU_NUM_THREADS;
    WHERE(TFS_BIAS_Q);
    bias_add_g<M, H>(&g_q[0][0], bq, tid, thr);
    WHERE(TFS_BIAS_K);
    bias_add_g<M, H>(&g_k[0][0], bk, tid, thr);
    WHERE(TFS_BIAS_V);
    bias_add_g<M, H>(&g_v[0][0], bv, tid, thr);
    TF_BAR(5);
}

// Write-drain from main() (warp 0 of the core), as in flash_attention_mx_fp6.
#ifndef TF_RUNG6_STEP
#define TF_RUNG6_STEP 0
#endif
// Stop scheduling after rung N, so a hang can be bisected to one rung (2026-09-18, H=256).
#ifndef TF_MAX_RUNG
#define TF_MAX_RUNG 7
#endif
// Rung-7 sub-steps, to bisect inside it: 1 = quantize ff, 2 = + FF2 GEMM, 3 = + bias/LN2/residual.
#ifndef TF_RUNG7_STEP
#define TF_RUNG7_STEP 3
#endif
volatile uint32_t tf_never = 0;                    // never true; keeps step-5 code reachable

// FENCE-ONLY DRAIN (default since 2026-09-27). `fence` is a complete handshake, so none of the
// blind waits below are needed:
//   TLNBDCache.scala  flush MMIO reg is RegField.w -- WRITE-ONLY, with `// !fio.busy` commented
//                     out, so an MMIO flush can never report completion. THAT is the only reason
//                     TF_WB_WAIT existed: a missing return path, not a timing tradeoff.
//                     The CacheFlushNode path does expose it: done := RegNext(busy) && !busy.
//   SFUPipe.scala     the D-fence waits on exactly that (`done = flushIO.d.done`), and only starts
//                     the flush once `globalQueuesEmpty && sharedQueuesEmpty`. Those signals are
//                     CORE-WIDE, not per-warp, so one warp's fence covers every warp's stores and
//                     scales with occupancy -- which a fixed poll count cannot.
//   TLNBDCache.scala  also says "we assume the core flush doesn't collide with mmio flush", and
//                     the old drain drove fio.start from BOTH paths a few hundred cycles apart.
// MEASURED at H=768 rung 6: 36,330,261 cycles vs 77,145,884 (2.12x) with numerics bit-identical
// (ff 24558/24576 both ways, rungs 1-6 PASS, rungs 1-2 bit-exact). Restore the old drain with
// -DTF_MMIO_WRITEBACK if a regression ever needs bisecting against it.
#if !defined(TF_MMIO_WRITEBACK) && !defined(TF_NO_MMIO_WRITEBACK)
#define TF_NO_MMIO_WRITEBACK 1
#endif
#ifndef TF_DRAIN_ITERS
#define TF_DRAIN_ITERS 0u
#endif
static inline void tf_write_drain() {
    mu_barrier(0, MU_NUM_CORES);
    mu_fence();
    for (volatile uint32_t d = 0; d < TF_DRAIN_ITERS; d++) asm volatile("" ::: "memory");
    mu_fence();
#ifndef TF_NO_MMIO_WRITEBACK
    // mu_fence() alone leaves whole dirty L0 lines in the cache: rung 3's QK operands came back
    // from the scratchpad with rows 24,25 and 28,29 of A (and 7,8 / 18,19 of B) reading zero --
    // 64-byte chunks, i.e. exactly one cache line each, on 64-byte-aligned addresses, while the
    // same DRAM buffers read back correct from the host afterwards (2026-09-17 rung3rows). The
    // move-in DMA therefore read DRAM before those lines were written back. Force the writeback
    // through the L0 D-cache flush register and give it time to retire.
    // NOT TF_L0D_FLUSH_WAIT (131072): that constant belongs to the one-shot diagnostic
    // tf_l0d_flush(). Used here it costs 2*131072 polling loads on EVERY phase boundary, which
    // put ~3.7M cycles in front of attention alone and ran rung 6 past the host watchdog
    // (2026-09-18). The wait only has to cover the writeback of a few dirty lines.
#ifndef TF_WB_WAIT
#define TF_WB_WAIT 16384u    // VALIDATED at H=512 (2026-09-22): rungs 1-7 all PASS with numerics
                             // identical to 131072 (ff maxerr 7813, out maxerr 0) at 59.0M cycles
                             // vs 229.2M -- a 3.9x speedup, the drains being most of the runtime.
                             // x2 flushes = 32768 loads, 2x the known-bad 16384x1-flush config.
                             // NOTE: blind poll count, write-only flush reg; dirty volume grows
                             // with H, so re-validate (rungs 1-2 are bit-exact) at each new size.
#endif
    for (uint32_t f = 0; f < TF_L0D_FLUSHES; f++) {
        store_shared(TF_L0D_FLUSH_ADDR, 0, (uint32_t)1);
        for (uint32_t i = 0; i < TF_WB_WAIT; i++) (void)load32_shared(GEMMINI_BUSY_ADDR);
    }
    mu_fence();
#endif
}

int main() {
    // Occupancy: warps per core. 2 ran to the last stage on the 2026-09-10 bitstream; occupancy 8 hung.
#ifndef TF_OCCUPANCY
#define TF_OCCUPANCY 2
#endif
    mu_schedule(setup_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();
    mu_schedule(quant_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // g_af8 / g_afs in DRAM before the move-in
    mu_schedule(gemm_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // g_q (tf_store_c) in DRAM
    mu_schedule(bias_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // q / k / v in DRAM (host, and rung 3 reads them)
    // rung 4: every head, each one QK -> softmax -> S@V, writing its own HD columns of g_ctx.
    // The head index rides in mu_schedule's arg, so every warp sees the same head.
    for (uint32_t hd = 0; hd < NH * (TF_MAX_RUNG >= 3 ? 1u : 0u); hd++) {
        void *const harg = (void *)(uintptr_t)hd;
        mu_schedule(head_prep_body, harg, TF_OCCUPANCY);
        tf_write_drain();                           // q_h/k_h^T fp8 operands in DRAM before move-in
        mu_schedule(qk_body, harg, TF_OCCUPANCY);
        tf_write_drain();                           // scores in DRAM before softmax reads them
        mu_schedule(softmax_body, harg, TF_OCCUPANCY);
        tf_write_drain();                           // probs in DRAM before the quantizer reads them
#if TF_MAX_RUNG >= 4
        mu_schedule(sv_prep_body, harg, TF_OCCUPANCY);
        tf_write_drain();                           // P/v_h fp8 operands in DRAM before move-in
        mu_schedule(sv_body, harg, TF_OCCUPANCY);
        tf_write_drain();                           // this head's ctx columns in DRAM
#endif
    }
#if TF_MAX_RUNG >= 5
    mu_schedule(o_prep_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // ctx fp8 operand in DRAM before the move-in
    mu_schedule(o_gemm_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // Wo output in DRAM before the epilogue
    mu_schedule(o_bias_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // bias pass flushed before LN1 starts
    mu_schedule(o_ln_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // LN1 pass flushed before the residual
    mu_schedule(o_res_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // g_res in DRAM for the host
#endif
    // Rung 6 is added one piece at a time (TF_RUNG6_STEP): the rung-6 build stalled at the Wv
    // projection -- unchanged rung-1/2 code -- so the trigger is what rung 6 adds globally, not its
    // logic. Steps are ordered by how much NEW code each pulls in:
    //   1 quantize res (mxquant_A<M,H> already instantiated by quant_body -- ~no new code)
    //   2 + FF1 GEMM   (gemm_weight<PROJ>, fullN is a runtime arg -- ~no new code)
    //   3 + bias       (bias_add_g<M,E>: new instantiation)
    //   4 + GELU       (gelu_g<M,E>: new instantiation, the erf polynomial -- the big one)
#if TF_RUNG6_STEP >= 1 && TF_MAX_RUNG >= 6
    mu_schedule(ff_prep_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // res fp8 operand in DRAM before the move-in
#endif
#if (TF_RUNG6_STEP >= 2 || TF_RUNG6_STEP == 5) && TF_MAX_RUNG >= 6
    // STEP 5 = step 2's IMAGE without step 2's EXECUTION: tf_never is a volatile zero, so the
    // compiler must keep ff_gemm_body (and therefore the FF1 weights) in the image, but the GEMM
    // never runs. Step 1 completes and step 2 hangs; if step 5 hangs too, the trigger is the
    // larger text/rodata, not executing the FF1 GEMM. (2026-09-18)
    if (TF_RUNG6_STEP != 5 || tf_never) {
        mu_schedule(ff_gemm_body, nullptr, TF_OCCUPANCY);
        tf_write_drain();                           // FF1 chunk 0 in DRAM
#if !defined(TF_FF1_ONE_CHUNK)
        // Chunk 1 reads FF1 from an OFFSET base (&FF1[(FFW/TN)*PROJ.K]); -DTF_FF1_ONE_CHUNK drops
        // it to test whether the offset call alone is what hangs at H=256 (2026-09-18).
        mu_schedule(ff_gemm_body_b, nullptr, TF_OCCUPANCY);
        tf_write_drain();                           // FF1 chunk 1 in DRAM
#endif
    }
#endif
#if TF_RUNG6_STEP >= 3 && TF_RUNG6_STEP != 5 && TF_MAX_RUNG >= 6
    mu_schedule(ff_post_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // g_ff in DRAM for the host
#endif
#if TF_RUNG6_STEP >= 4 && TF_RUNG6_STEP != 5 && TF_MAX_RUNG >= 7
    mu_schedule(out_prep_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // ff fp8 operand in DRAM before the move-in
#if TF_RUNG7_STEP >= 2
    mu_schedule(out_gemm_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // FF2 output in DRAM
#endif
#ifndef TF_FF2_SINGLE
    mu_schedule(out_gemm_body_b, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // FF2 chunk 1 in DRAM
#endif
#if TF_RUNG7_STEP >= 3
    mu_schedule(out_post_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // g_out in DRAM for the host
#endif
#endif
#ifdef TF_DUMMY_PHASES
    for (uint32_t d = 0; d < TF_DUMMY_PHASES; d++) {
        mu_schedule(dummy_body, nullptr, TF_OCCUPANCY);
        tf_write_drain();
    }
#endif
    mu_schedule(done_body, nullptr, TF_OCCUPANCY);
    tf_write_drain();                               // beacon in DRAM for the host
    return 0;
}
