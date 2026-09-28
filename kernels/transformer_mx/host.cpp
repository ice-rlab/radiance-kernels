//inttypes.h is for the fixed width types
#include <inttypes.h>
//stdio.h is for the printf statements
#include <stdio.h>
//radiance.h stores the macros for host GPU interface
#include <radiance.h>

#include "tf_beacon.h"
// Golden: python3 golden_tf.py --emit include/tf_golden.h <same dims as tf_data.h>
#if __has_include("include/tf_golden.h")
#include "include/tf_golden.h"
#define TF_HAVE_GOLDEN 1
#endif
// Device addresses of the kernel's DRAM buffers (g_q ...), generated from the linked GPU elf by the
// Makefile, so the host reads them with no change to the GPU code.
// Golden device-quantized A (emit_quant_golden.py) for the A check below.
#if __has_include("include/tf_golden_quant.h")
#include "include/tf_golden_quant.h"
// Only usable when it describes the CURRENT model. emit_quant_golden.py is regenerated far less
// often than tf_data.h/tf_golden.h, so it silently goes stale: at H=768 it still held H=128/HD=32
// data (TFGQ_K 128 vs TFG_HIDDEN 768), and every operand line it printed -- `A OK (4096)`, the
// SRAW0 dev/gold/unit hex dumps, the AQDIFF rows -- was comparing against a different model while
// costing ~2-3 KB of console at ~3 BYTES/s, i.e. >10 minutes of FPGA occupancy per run
// (2026-09-27). Gate on the dimensions rather than a hand-set flag so it disables itself when
// stale and comes back automatically once regenerated at the right size.
#if TFGQ_M == TFG_SEQ && TFGQ_K == TFG_HIDDEN
#define TF_HAVE_GOLDEN_QUANT 1
#else
#define TF_QUANT_GOLDEN_STALE 1
#endif
#endif
#if __has_include("include/tf_syms.h")
#include "include/tf_syms.h"
#define TF_HAVE_SYMS 1
#endif

static const char *const PHASE_NAMES[TF_NUM_PHASES] = TF_PHASE_NAMES_INIT;
#define TF_STAGE_NAME(n, s) s,
static const char *const STAGE_NAMES[] = { TF_STAGES(TF_STAGE_NAME) };
#undef TF_STAGE_NAME

// Gemmini tile command regmap (GemminiTile.scala's TLRegisterNode), at
// clusterBaseAddr(0x40000000) + peripheralAddrOffset(0x80000) + 0x4000. All read-only status
// fields, so polling them from the host is side-effect free.
//   0x08 = gemminiIO.ready   0x20 = gemmini.io.busy   0x28 = running loops
static constexpr uint64_t GEMMINI_REG_READY = 0x40084008ull;
static constexpr uint64_t GEMMINI_REG_BUSY  = 0x40084020ull;
static constexpr uint64_t GEMMINI_REG_LOOPS = 0x40084028ull;

// bf16 code -> value in integer micro-units (1e-6). INTEGER ONLY: soc/start.S does no FPU setup,
// so the host is kept free of floating point entirely.
static inline int64_t bf16_to_micro(uint16_t code) {
    const int64_t e = (code >> 7) & 0xff, m = code & 0x7f;
    if (e == 0) return 0;                                   // zero / subnormal
    if (e == 0xff) return (code & 0x8000) ? -2000000000LL : 2000000000LL;
    const int64_t base = (128 + m) * 1000000LL;             // value = base * 2^(e - 134) micro
    const int64_t sh = e - 134;
    int64_t v = sh >= 0 ? (sh > 20 ? 2000000000LL : base << sh) : (-sh >= 40 ? 0 : base >> -sh);
    if (v > 2000000000LL) v = 2000000000LL;
    return (code & 0x8000) ? -v : v;
}
static inline int bf16_nonfinite(uint16_t code) { return ((code >> 7) & 0xff) == 0xff; }

// One line per checked buffer, compared cell by cell on the host in integers. Console output costs
// ~3 BYTES/s wall clock on the FPGA (HTIF), so every line is kept short.
// Tolerance = depth * (max|golden| + 1e-3), as golden_tf._phase_tol. Returns 1 if the buffer passes.
// stride = row pitch of BOTH device and golden buffers; 0 means they are exactly `cols` wide.
// A non-zero stride checks a column slice of a wider buffer (e.g. one head's columns of g_ctx).
static int phase_line(const char *name, uint64_t dev_addr, const uint16_t *gold, uint32_t rows,
                      uint32_t cols, int64_t tol_milli, uint32_t stride = 0) {
    const uint16_t *p = (const uint16_t *)rad_device_to_host_address(dev_addr);
    const uint32_t pitch = stride ? stride : cols;
    uint32_t exact = 0, lost = 0, nonfinite = 0, bad_rows = 0;
    int64_t max_err = 0, max_ref = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t row_bad = 0;
        for (uint32_t c = 0; c < cols; c++) {
            const uint16_t got = p[r * pitch + c], exp = gold[r * pitch + c];
            const int64_t ref = bf16_to_micro(exp);
            const int64_t aref = ref < 0 ? -ref : ref;
            if (aref > max_ref) max_ref = aref;
            exact += (got == exp);
            lost += ((got & 0x7fffu) == 0 && (exp & 0x7fffu) != 0);
            if (bf16_nonfinite(got)) { nonfinite++; row_bad = 1; continue; }
            int64_t err = bf16_to_micro(got) - ref;
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
        }
        bad_rows += row_bad;
    }
    const int64_t tol = (max_ref + 1000) * tol_milli / 1000;
    const int ok = (nonfinite == 0 && lost == 0 && max_err <= tol);
    printf("[host] %-3s exact %4u/%u lost %u nonfin %u(%u rows) maxerr %d tol %d %s\n", name, exact,
           rows * cols, lost, nonfinite, bad_rows, (int)max_err, (int)tol, ok ? "OK" : "BAD");
    return ok;
}

// The per-buffer bad-cell MAPS are the expensive part of the report: each is ~420 characters at
// H=768 (4 tile rows x 96 tile cols + the row map) and the console runs at ~3 BYTES/s over HTIF,
// so eight maps cost ~20 minutes of FPGA occupancy. Worse, a run that is stopped while still
// printing loses the REST of the report: the 150M-poll hang run (2026-09-23--02-30-31) was cut off
// mid-line at "k exact 24576/2", so s0/ctx/res -- the buffers that say WHERE it stopped -- never
// printed at all. Default OFF for hang-hunting; build with -DTF_HOST_MAPS (or use
// host_debug_backup.cpp) when a map's SHAPE is actually the question. The one-line phase_line
// compares are kept: they are ~90 chars each and carry the pass/fail verdict.
#ifdef TF_HOST_MAPS
// Compact bad-cell map of one buffer (~200 bytes): a row map (one char per row: '.' clean,
// 'I' has non-finite, 'x' finite but out of tolerance) and the bad-cell count of every tile x tile
// block ('0'-'9', '+' for >9), tile rows top to bottom, '|' between tile rows. tile = the hardware
// Gemmini DIM (8 on FireSimRadianceSingleClusterSyn), so a structural GEMM fault shows its shape.
static void phase_map(const char *name, uint64_t dev_addr, const uint16_t *gold, uint32_t rows,
                      uint32_t cols, int64_t tol_milli, uint32_t tile) {
    const uint16_t *p = (const uint16_t *)rad_device_to_host_address(dev_addr);
    int64_t max_ref = 0;
    for (uint32_t i = 0; i < rows * cols; i++) {
        int64_t a = bf16_to_micro(gold[i]);
        if (a < 0) a = -a;
        if (a > max_ref) max_ref = a;
    }
    const int64_t tol = (max_ref + 1000) * tol_milli / 1000;
    char rowmap[129];
    static char grid[16 * 64 + 64];
    uint32_t gn = 0;
    const uint32_t trows = rows / tile, tcols = cols / tile;
    for (uint32_t r = 0; r < rows && r < 128; r++) rowmap[r] = '.';
    for (uint32_t tr = 0; tr < trows; tr++) {
        for (uint32_t tc = 0; tc < tcols; tc++) {
            uint32_t bad = 0;
            for (uint32_t r = tr * tile; r < (tr + 1) * tile; r++) {
                for (uint32_t c = tc * tile; c < (tc + 1) * tile; c++) {
                    const uint16_t got = p[r * cols + c], exp = gold[r * cols + c];
                    int is_bad = 0;
                    if (bf16_nonfinite(got)) { is_bad = 1; if (r < 128) rowmap[r] = 'I'; }
                    else {
                        int64_t e = bf16_to_micro(got) - bf16_to_micro(exp);
                        if (e < 0) e = -e;
                        if (e > tol) { is_bad = 1; if (r < 128 && rowmap[r] == '.') rowmap[r] = 'x'; }
                    }
                    bad += is_bad;
                }
            }
            if (gn + 2 < sizeof(grid)) grid[gn++] = bad == 0 ? '.' : (bad > 9 ? '+' : (char)('0' + bad));
        }
        if (gn + 2 < sizeof(grid)) grid[gn++] = '|';
    }
    rowmap[rows < 128 ? rows : 128] = 0;
    grid[gn] = 0;
    printf("[host] %s rows %s\n[host] %s tiles %s\n", name, rowmap, name, grid);
}
#else
static inline void phase_map(const char *, uint64_t, const uint16_t *, uint32_t, uint32_t, int64_t,
                             uint32_t) {}
#endif

int main() {
    //RISCV mail box= tohost=1 is simulation done
    tohost = 0;
    *tocpu = tohost;

    // Zero the beacon, phase marks, the DONE flag and every checked buffer BEFORE deasserting reset,
    // so nothing left in DRAM by a previous run can read as this run's progress or result.
    volatile uint32_t *where = (volatile uint32_t *)rad_device_to_host_address(TF_WHERE_GMEM);
    volatile uint32_t *mark = (volatile uint32_t *)rad_device_to_host_address(TF_MARK_GMEM);
    for (uint32_t i = 0; i < 6; i++) where[i] = 0;
    for (uint32_t i = 0; i < TF_NUM_PHASES; i++) mark[i] = 0;
#if defined(TF_HAVE_SYMS) && defined(TF_HAVE_GOLDEN)
    {
        volatile uint16_t *q = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_q);
        volatile uint16_t *k = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_k);
        volatile uint16_t *v = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_v);
        for (uint32_t i = 0; i < (uint32_t)(TFG_SEQ * TFG_HIDDEN); i++) { q[i] = 0; k[i] = 0; v[i] = 0; }
        volatile uint16_t *s0 = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_s);
        for (uint32_t i = 0; i < (uint32_t)(TFG_SEQ * TFG_SEQ); i++) s0[i] = 0;
        // ctx / res / ff / out were NOT being zeroed, though the comment above claimed "every
        // checked buffer" (found 2026-09-24). DRAM survives between runs, so on a HUNG run these
        // four showed the PREVIOUS run's contents: the rung-7 hang reported `res 24411/24576` and
        // an earlier hang reported `ctx exact 24576/24576`, neither of which the hung run had
        // necessarily written. A buffer the kernel never reached must read as zeros, not as the
        // last run's answer, or "how far did the data get" is unanswerable on exactly the runs
        // where it matters most. g_s0 is the rung-3 copy and is written from g_s, so it is left
        // alone; g_s (zeroed above) is the one the kernel writes per head.
        volatile uint16_t *ctx = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_ctx);
        volatile uint16_t *res = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_res);
        volatile uint16_t *out = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_out);
        for (uint32_t i = 0; i < (uint32_t)(TFG_SEQ * TFG_HIDDEN); i++) { ctx[i] = 0; res[i] = 0; out[i] = 0; }
        volatile uint16_t *ffb = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_ff);
        for (uint32_t i = 0; i < (uint32_t)(TFG_SEQ * TFG_EXP); i++) ffb[i] = 0;
    }
#endif
#if defined(TF_HAVE_SYMS) && defined(TF_HAVE_GOLDEN_QUANT)
    {   // zero the quantized A so an element the GPU never wrote cannot match by accident
        volatile uint8_t *a = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_af8);
        volatile uint8_t *as = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_afs);
        for (uint32_t i = 0; i < TFGQ_M * TFGQ_K; i++) a[i] = 0;
        for (uint32_t i = 0; i < TFGQ_GK * TFGQ_M; i++) as[i] = 0;
        volatile uint8_t *qf = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_q0f8);
        volatile uint8_t *qs = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_q0fs);
        volatile uint8_t *kf = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_ktf8);
        volatile uint8_t *ks = (volatile uint8_t *)rad_device_to_host_address(TFSYM_g_kts);
        volatile uint16_t *sr = (volatile uint16_t *)rad_device_to_host_address(TFSYM_g_sraw);
        for (uint32_t i = 0; i < TFGQ_S * TFGQ_HD; i++) { qf[i] = 0; kf[i] = 0; }
        for (uint32_t i = 0; i < TFGQ_S; i++) { qs[i] = 0; ks[i] = 0; }
        for (uint32_t i = 0; i < TFGQ_S * TFGQ_S; i++) sr[i] = 0;
    }
#endif

    // Captured, not printed: a printf here would land after reset-deassert and cost ~1.9e9 target
    // cycles of HTIF latency inside the measurement window.
    {   // zero the beacon progress slots: an unwritten pass must read 0, not stale DRAM (pass=204)
        volatile uint32_t *w = (volatile uint32_t *)rad_device_to_host_address(TF_WHERE_GMEM);
        w[6] = 0; w[7] = 0;
    }
    const uint32_t base_busy  = READ_MMIO_32(GEMMINI_REG_BUSY);
    const uint32_t base_ready = READ_MMIO_32(GEMMINI_REG_READY);
    const uint32_t base_loops = READ_MMIO_32(GEMMINI_REG_LOOPS);

    //Write to MMIO address RAD_HOST_GPU_RESET.
    // Deassert reset
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

    // NOTHING IS PRINTED INSIDE THIS LOOP -- ON PURPOSE (HTIF latency, see above).
    // WATCHDOG: bounded wait, so a hung GPU still produces a report and ends the simulation.
    //
    // POLL RATE, MEASURED FROM COMPLETING RUNS (2026-09-23). A completing run polls for its whole
    // duration -- it exits the loop the moment the GPU raises ALL_FINISHED -- so cycles/polls at
    // completion IS the rate, not a lower bound. Four runs across three model sizes agree:
    //     H=768 rung-2  16,692,998 cyc /   221,263 polls = 75
    //     H=768 rung-4  68,605,004 cyc /   886,782 polls = 77
    //     H=512 full    59,044,508 cyc /   764,210 polls = 77
    //     H=512 full   229,249,141 cyc / 2,987,491 polls = 77
    // So a poll costs ~76 GPU cycles. Every earlier figure in this comment (9, 1.07, 3) was wrong,
    // and the "~3 cycles/poll" one was the worst kind of wrong: it came from a run whose stage
    // stamp had FROZEN, so the cycle it read was the hang site, not the elapsed time.
    //
    // 12M polls ~= 912M GPU cycles. A healthy H=768 rung-5 run should land near 80M cycles (the
    // rung-4 stop is 68.6M), so this is ~11x headroom -- while costing only ~2 minutes of wall
    // clock when it does expire, instead of the ~25 minutes that 150M polls cost. That matters:
    // the console runs at ~3 BYTES/s, and the 150M-poll run (2026-09-23--02-30-31) spent so long
    // in the watchdog that its report was still printing when the run ended, truncating the log
    // mid-line at "k exact 24576/2".
    //
    // The hang this is watching for is REAL, not a watchdog artifact: the 20M-poll and 150M-poll
    // runs of the same build froze at cycle 61,869,748 and 61,873,827 respectively -- 130M extra
    // polls bought 4,079 cycles of progress.
    static constexpr uint32_t TF_POLL_LIMIT = 12000000u;
    uint32_t polls = 0;
    uint32_t finished = 0;
    while (!finished && polls < TF_POLL_LIMIT) {
        SYNC_GPU();
        polls++;
        //Poll the MMIO address to see when the GPU is finished with the kernel
        finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    }
    const uint32_t timed_out = !finished;

    // Snapshot everything while the values are still live, before touching reset.
    uint32_t phase_snap[TF_NUM_PHASES];
    uint32_t where_snap[6];
    for (uint32_t i = 0; i < TF_NUM_PHASES; i++) phase_snap[i] = mark[i];
    for (uint32_t i = 0; i < 6; i++) where_snap[i] = where[i];
    const uint32_t gem_busy  = READ_MMIO_32(GEMMINI_REG_BUSY);
    const uint32_t gem_ready = READ_MMIO_32(GEMMINI_REG_READY);
    const uint32_t gem_loops = READ_MMIO_32(GEMMINI_REG_LOOPS);

    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);

    // Safe to print now: the GPU is back in reset.
    if (timed_out)
        printf("[host] WATCHDOG: GPU not finished after %u polls -- GPU HUNG, reporting beacon\n", polls);
    printf("[host] gemmini at reset-deassert: busy=%u ready=%u loops=%u\n",
           base_busy, base_ready, base_loops);
    for (uint32_t i = 0; i < TF_NUM_PHASES; i++) {
        if (phase_snap[i] != 0)
            printf("[host] phase %u (%s) reached at cycle %u\n", i, PHASE_NAMES[i], phase_snap[i]);
    }
    if (where_snap[0] != 0) {
        const uint32_t id = where_snap[0] - 1;
        printf("[host] last stage: %s (stage %u, cycle %u)\n",
               id < TFS_COUNT ? STAGE_NAMES[id] : "??unknown??", id, where_snap[1]);
    }
    {   // row-level progress inside the last phase: [6] = pass (1 bias, 2 LN1, 3 residual),
        // [7] = last row started + 1 (0 = that pass never began). Added 2026-09-22 after five
        // hypotheses for the H=768 rung-5 hang failed -- this says WHERE it stops, not why.
        const volatile uint32_t *w = (const volatile uint32_t *)rad_device_to_host_address(TF_WHERE_GMEM);
        printf("[host] progress: pass=%u last_row=%d\n", w[6], (int)w[7] - 1);
    }
    printf("[host] gemmini busy=%u ready=%u loops=%u (polls %u) done=0x%08x\n",
           gem_busy, gem_ready, gem_loops, polls, where_snap[5]);

    // ---- rung checks: one map + one line per rung buffer, in pipeline order. The first BAD line is
    // where the bug enters. The buffers are checked even without the done flag, so a hang still shows
    // how far the data got. ----
#if defined(TF_HAVE_SYMS) && defined(TF_HAVE_GOLDEN)
    const uint32_t seq = TFG_SEQ, hidden = TFG_HIDDEN;
    int all_ok = !timed_out && where_snap[5] == TF_DONE_MAGIC;
    // rung 1: Q = X @ Wq + bq
    phase_map("q", TFSYM_g_q, &G_q[0][0], seq, hidden, 60, 8u);
    all_ok &= phase_line("q", TFSYM_g_q, &G_q[0][0], seq, hidden, 60);
    printf("[host] RUNG 1 %s\n", all_ok ? "PASS" : "FAIL");
    // rung 2: K = X @ Wk + bk, V = X @ Wv + bv (a rung passes only if every earlier rung does)
    phase_map("k", TFSYM_g_k, &G_k[0][0], seq, hidden, 60, 8u);
    all_ok &= phase_line("k", TFSYM_g_k, &G_k[0][0], seq, hidden, 60);
    phase_map("v", TFSYM_g_v, &G_v[0][0], seq, hidden, 60, 8u);
    all_ok &= phase_line("v", TFSYM_g_v, &G_v[0][0], seq, hidden, 60);
    printf("[host] RUNG 2 %s\n", all_ok ? "PASS" : "FAIL");
    // rung 3: head 0 scores, S = softmax(q0 @ k0^T * 1/sqrt(HD))
    // g_s is reused by every head now, so the rung-3 check reads g_s0 (head 0, kept aside).
    phase_map("s0", TFSYM_g_s0, &G_s0[0][0], seq, seq, 60, 8u);
    all_ok &= phase_line("s0", TFSYM_g_s0, &G_s0[0][0], seq, seq, 60);
    printf("[host] RUNG 3 %s\n", all_ok ? "PASS" : "FAIL");

    // rung 4: all heads + S@V. Report head 0's columns separately from the full context, so one
    // run says whether S@V itself works (head 0) and whether the per-head loop works (all heads).
    phase_map("ctx", TFSYM_g_ctx, &G_ctx[0][0], seq, hidden, 120, 8u);
    {
        const int h0 = phase_line("ctx0", TFSYM_g_ctx, &G_ctx[0][0], seq, TFG_HIDDEN / 4, 120,
                                  /*stride=*/hidden);
        all_ok &= h0;
    }
    all_ok &= phase_line("ctx", TFSYM_g_ctx, &G_ctx[0][0], seq, hidden, 120);
    printf("[host] RUNG 4 %s\n", all_ok ? "PASS" : "FAIL");

    // rung 5: Wo + bias, LN1, residual
    phase_map("res", TFSYM_g_res, &G_res[0][0], seq, hidden, 120, 8u);
    all_ok &= phase_line("res", TFSYM_g_res, &G_res[0][0], seq, hidden, 120);
    printf("[host] RUNG 5 %s\n", all_ok ? "PASS" : "FAIL");

    // rung 6: FF1 + bias, GELU
    phase_map("ff", TFSYM_g_ff, &G_ff[0][0], seq, TFG_EXP, 120, 8u);
    all_ok &= phase_line("ff", TFSYM_g_ff, &G_ff[0][0], seq, TFG_EXP, 120);
    printf("[host] RUNG 6 %s\n", all_ok ? "PASS" : "FAIL");

    // rung 7: FF2 + bias, LN2, residual -- the full encoder layer
    phase_map("out", TFSYM_g_out, &G_out[0][0], seq, hidden, 120, 8u);
    all_ok &= phase_line("out", TFSYM_g_out, &G_out[0][0], seq, hidden, 120);
    printf("[host] RUNG 7 %s\n", all_ok ? "PASS" : "FAIL");
#ifdef TF_HAVE_GOLDEN_QUANT
    // GPU-made operands vs golden: one line each. A = mxquant_A(X) (rungs 1-2); the head-0 buffers
    // and the pre-softmax scores are the rung-3 chain, in the order the kernel builds them.
    {
        auto cmp8 = [](const char *tag, uint64_t addr, const uint8_t *gold, uint32_t n) {
            const uint8_t *p = (const uint8_t *)rad_device_to_host_address(addr);
            uint32_t bad = 0, first = 0xffffffffu;
            for (uint32_t i = 0; i < n; i++)
                if (p[i] != gold[i]) { if (!bad) first = i; bad++; }
            if (bad) printf("[host] %-5s bad %4u/%4u BAD (first i%u dev %02x gold %02x)\n", tag, bad, n,
                            first, p[first], gold[first]);
            else printf("[host] %-5s OK (%u)\n", tag, n);
        };
        auto cmp16 = [](const char *tag, uint64_t addr, const uint16_t *gold, uint32_t n) {
            const uint16_t *p = (const uint16_t *)rad_device_to_host_address(addr);
            uint32_t bad = 0, nonfin = 0, first = 0xffffffffu;
            for (uint32_t i = 0; i < n; i++) {
                nonfin += ((p[i] >> 7) & 0xff) == 0xff;
                if (p[i] != gold[i]) { if (!bad) first = i; bad++; }
            }
            if (bad) printf("[host] %-5s bad %4u/%4u nonfin %u BAD (first i%u dev %04x gold %04x)\n", tag,
                            bad, n, nonfin, first, p[first], gold[first]);
            else printf("[host] %-5s OK (%u)\n", tag, n);
        };
        cmp8("A", TFSYM_g_af8, TFGQ_af8, TFGQ_M * TFGQ_K);
        cmp8("As", TFSYM_g_afs, TFGQ_afs, TFGQ_GK * TFGQ_M);
        cmp16("q0", TFSYM_g_q0, TFGQ_q0, TFGQ_S * TFGQ_HD);
        cmp16("k0", TFSYM_g_k0, TFGQ_k0, TFGQ_S * TFGQ_HD);
        cmp16("kt", TFSYM_g_kt, TFGQ_kt, TFGQ_HD * TFGQ_S);
        cmp8("q0f8", TFSYM_g_q0f8, TFGQ_q0f8, TFGQ_S * TFGQ_HD);
        cmp8("q0fs", TFSYM_g_q0fs, TFGQ_q0fs, TFGQ_S);
        cmp8("ktf8", TFSYM_g_ktf8, TFGQ_ktf8, TFGQ_HD * TFGQ_S);
        cmp8("kts", TFSYM_g_kts, TFGQ_kts, TFGQ_S);
        // Gemmini stores operands in the scratchpad TILED: tile (i,k) at (i*tiles_k + k)*DIM rows,
        // each DIM x DIM tile contiguous. The read-back copies SMEM linearly, so byte
        // ((i*tiles_k + k)*DIM + r)*DIM + c is operand element [i*DIM + r][k*DIM + c], not
        // row-major. Comparing row-major reported ~950/1024 "bad" on correct data (2026-09-17).
        auto cmp8_tiled = [](const char *tag, uint64_t addr, const uint8_t *gold,
                             uint32_t rows, uint32_t cols) {
            const uint8_t *p = (const uint8_t *)rad_device_to_host_address(addr);
            const uint32_t D = 8, tk = cols / D;
            uint32_t bad = 0, firsti = 0, firstj = 0, dv = 0, gv = 0;
            for (uint32_t i = 0; i < rows / D; i++)
                for (uint32_t k = 0; k < tk; k++)
                    for (uint32_t r = 0; r < D; r++)
                        for (uint32_t c = 0; c < D; c++) {
                            uint32_t si = ((i * tk + k) * D + r) * D + c;
                            uint32_t gi = (i * D + r) * cols + (k * D + c);
                            if (p[si] != gold[gi]) {
                                if (!bad) { firsti = i * D + r; firstj = k * D + c; dv = p[si]; gv = gold[gi]; }
                                bad++;
                            }
                        }
            if (bad) {
                printf("[host] %-5s bad %4u/%4u BAD (first [%u][%u] dev %02x gold %02x)\n", tag,
                       bad, rows * cols, firsti, firstj, dv, gv);
                // per-tile bad count (hex, capped at f) and how many of the bad bytes are zero:
                // distinguishes "move-in skipped whole tiles" from "partial rows".
                // per-operand-ROW mismatch count (hex, capped at f): tells a single dropped
                // 128-byte DMA chunk (4 consecutive rows) apart from a per-tile row truncation.
                printf("[host] %-5s rows  ", tag);
                for (uint32_t rr = 0; rr < rows; rr++) {
                    uint32_t rb = 0;
                    for (uint32_t cc = 0; cc < cols; cc++) {
                        uint32_t si = (((rr / D) * tk + cc / D) * D + (rr % D)) * D + (cc % D);
                        if (p[si] != gold[rr * cols + cc]) rb++;
                    }
                    printf("%x", rb > 15 ? 15 : rb);
                }
                printf("\n");
                printf("[host] %-5s tiles ", tag);
                uint32_t zeros = 0;
                for (uint32_t i = 0; i < rows / D; i++)
                    for (uint32_t k = 0; k < tk; k++) {
                        uint32_t tb = 0;
                        for (uint32_t r = 0; r < D; r++)
                            for (uint32_t c = 0; c < D; c++) {
                                uint32_t si = ((i * tk + k) * D + r) * D + c;
                                uint32_t gi = (i * D + r) * cols + (k * D + c);
                                if (p[si] != gold[gi]) { tb++; zeros += (p[si] == 0); }
                            }
                        printf("%x", tb > 15 ? 15 : tb);
                    }
                printf("  (zero-valued bad bytes %u/%u)\n", zeros, bad);
            }
            else printf("[host] %-5s OK (%u, tiled)\n", tag, rows * cols);
        };
        cmp8_tiled("Asmem", TFSYM_g_asmem, TFGQ_q0f8, TFGQ_S, TFGQ_HD);   // what the mesh read
        cmp8_tiled("Bsmem", TFSYM_g_bsmem, TFGQ_ktf8, TFGQ_HD, TFGQ_S);
        // Golden-free: scratchpad read-back vs the device's OWN DRAM operand. Both are device data,
        // so a mismatch here cannot be a golden artifact -- it means the move-in did not deliver
        // what DRAM holds. (The golden checks above already agree, since q0f8/ktf8 match DRAM.)
        cmp8_tiled("AsmDR", TFSYM_g_asmem,
                   (const uint8_t *)rad_device_to_host_address(TFSYM_g_q0f8), TFGQ_S, TFGQ_HD);
        cmp8_tiled("BsmDR", TFSYM_g_bsmem,
                   (const uint8_t *)rad_device_to_host_address(TFSYM_g_ktf8), TFGQ_HD, TFGQ_S);
        // Report against both goldens: HOST_CFLAGS is ?= in common.mk so -DTF_DIAG_UNIT_SCALES
        // never reaches this file, and which one applies depends on the device build.
        cmp16("sraw", TFSYM_g_sraw, TFGQ_scores, TFGQ_S * TFGQ_S);
        cmp16("srawU", TFSYM_g_sraw, TFGQ_scores_unit, TFGQ_S * TFGQ_S);
        {   // one row of raw scores, device then golden: ratio/shift tells scale vs addressing
            static const char HEX[] = "0123456789abcdef";
            const uint16_t *p = (const uint16_t *)rad_device_to_host_address(TFSYM_g_sraw);
            char line[TFGQ_S * 5 + 1];
            for (int which = 0; which < 3; which++) {
                const uint16_t *src = which == 0 ? p : (which == 1 ? TFGQ_scores : TFGQ_scores_unit);
                uint32_t n = 0;
                for (uint32_t c = 0; c < TFGQ_S; c++) {
                    const uint16_t v = src[c];
                    line[n++] = ' ';
                    line[n++] = HEX[(v >> 12) & 0xf]; line[n++] = HEX[(v >> 8) & 0xf];
                    line[n++] = HEX[(v >> 4) & 0xf];  line[n++] = HEX[v & 0xf];
                }
                line[n] = 0;
                printf("SRAW0 %s%s\n", which == 0 ? "dev " : (which == 1 ? "gold" : "unit"), line);
            }
        }
    }
#endif

#ifdef TF_HAVE_GOLDEN_QUANT
    // A check: the GPU-quantized X (g_af8 / g_afs), read back after the run, against the golden
    // mxquant_A. Equal -> the GPU computed A right, so a wrong q means Gemmini read A too early.
    {
        const uint8_t *a = (const uint8_t *)rad_device_to_host_address(TFSYM_g_af8);
        const uint8_t *as = (const uint8_t *)rad_device_to_host_address(TFSYM_g_afs);
        char rows[TFGQ_M + 1];
        uint32_t bad_a = 0, bad_s = 0, zero_a = 0, shown = 0;
        for (uint32_t m = 0; m < TFGQ_M; m++) {
            uint32_t ba = 0, bs = 0;
            for (uint32_t k = 0; k < TFGQ_K; k++) {
                const uint8_t d = a[m * TFGQ_K + k], g = TFGQ_af8[m * TFGQ_K + k];
                if (d != g) {
                    ba++;
                    zero_a += (d == 0);
                    if (shown < 12) { printf("AQDIFF a r%u k%u dev %02x gold %02x\n", m, k, d, g); shown++; }
                }
            }
            for (uint32_t gk = 0; gk < TFGQ_GK; gk++) {
                const uint8_t d = as[gk * TFGQ_M + m], g = TFGQ_afs[gk * TFGQ_M + m];
                if (d != g) {
                    bs++;
                    if (shown < 12) { printf("AQDIFF s r%u g%u dev %02x gold %02x\n", m, gk, d, g); shown++; }
                }
            }
            bad_a += ba; bad_s += bs;
            rows[m] = (ba == 0 && bs == 0) ? '.' : (bs ? 'S' : 'x');
        }
        rows[TFGQ_M] = 0;
        printf("[host] A rows %s\n", rows);
        printf("[host] A fp8 bad %u/%u (dev zero %u)  scales bad %u/%u  %s\n", bad_a, TFGQ_M * TFGQ_K, zero_a,
               bad_s, TFGQ_M * TFGQ_GK, (bad_a || bad_s) ? "A BAD" : "A OK");
    }
#endif

#endif

#ifdef TF_QUANT_GOLDEN_STALE
    printf("[host] quantgold SKIPPED (stale dims)\n");
#endif
    printf("finished\n");
    tohost = 1;
    return 0;
}
