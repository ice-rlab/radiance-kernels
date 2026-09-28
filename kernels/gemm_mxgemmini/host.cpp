#include <inttypes.h>
#include <stdio.h>
#include <radiance.h>

// -DMXGEMM_VERIFY_M64N64K128: check C of mxgemm.fp8.m64n64k128.tm64tn64tk64.fullout against its
// bit-exact golden (C_out_bf16, bf16 row-major [64][64] at GPU GMEM 0x40000000). Build only that
// variant with it: every *.soc.elf in this directory links the same host object.
#ifdef MXGEMM_VERIFY_M64N64K128
#include "mxgemm.data.fp8.m64n64k128.h"

static constexpr uint32_t MXV_C_DEV = 0x40000000u;
static constexpr uint32_t MXV_TILE = 8;            // hardware DIM on FireSimRadianceSingleClusterSyn
static constexpr uint32_t MXV_POLL_LIMIT = 2000000u;

// bf16 code -> integer micro-units (INTEGER ONLY: soc/start.S does no FPU setup).
static inline int64_t mxv_micro(uint16_t code) {
  const int64_t e = (code >> 7) & 0xff, m = code & 0x7f;
  if (e == 0) return 0;
  if (e == 0xff) return (code & 0x8000) ? -2000000000LL : 2000000000LL;
  const int64_t base = (128 + m) * 1000000LL, sh = e - 134;
  int64_t v = sh >= 0 ? (sh > 20 ? 2000000000LL : base << sh) : (-sh >= 40 ? 0 : base >> -sh);
  if (v > 2000000000LL) v = 2000000000LL;
  return (code & 0x8000) ? -v : v;
}
static inline int mxv_nonfinite(uint16_t code) { return ((code >> 7) & 0xff) == 0xff; }

#ifdef MXGEMM_DIAG_LATE_COPY
// Second copy of C made by the GPU after an extra delay/fence/barrier (mxgemm_lib.hpp).
static constexpr uint32_t MXV_C_LATE_DEV = 0x40010000u;
#endif

// One line: exact count, plus for rows 32..63 a 16-hex-digit mask of zero cells (bit k = col k).
static void mxv_zero_summary(const char *tag, const uint16_t *c) {
  static const char HEX[] = "0123456789abcdef";
  uint32_t exact = 0, lost = 0;
  for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; i++) {
    exact += c[i] == C_out_bf16[i / MATMUL_N][i % MATMUL_N];
    lost += (c[i] & 0x7fffu) == 0 && (C_out_bf16[i / MATMUL_N][i % MATMUL_N] & 0x7fffu) != 0;
  }
  printf("[mxv] %s exact %u lost %u\n", tag, exact, lost);
  for (uint32_t r = 32; r < MATMUL_M; r += 8) {
    char line[8 * 17 + 1];
    uint32_t n = 0;
    for (uint32_t rr = r; rr < r + 8; rr++) {
      uint64_t z = 0;
      for (uint32_t k = 0; k < MATMUL_N; k++)
        if ((c[rr * MATMUL_N + k] & 0x7fffu) == 0) z |= 1ull << k;
      line[n++] = ' ';
      for (int d = 15; d >= 0; d--) line[n++] = HEX[(z >> (4 * d)) & 0xf];
    }
    line[n] = 0;
    printf("ZMASK %s r%u%s\n", tag, r, line);
  }
}

static void mxv_dump_row(const uint16_t *c, uint32_t r) {
  static const char HEX[] = "0123456789abcdef";
  char chunk[5 * 16 + 1];
  printf("MXDUMP c %u", r);
  for (uint32_t c0 = 0; c0 < MATMUL_N; c0 += 16) {
    uint32_t n = 0;
    for (uint32_t k = c0; k < c0 + 16 && k < MATMUL_N; k++) {
      const uint16_t v = c[r * MATMUL_N + k];
      chunk[n++] = ' ';
      chunk[n++] = HEX[(v >> 12) & 0xf];
      chunk[n++] = HEX[(v >> 8) & 0xf];
      chunk[n++] = HEX[(v >> 4) & 0xf];
      chunk[n++] = HEX[v & 0xf];
    }
    chunk[n] = 0;
    printf("%s", chunk);
  }
  printf("\n");
}
#endif

int main() {
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  tohost = 0;
  *tocpu = tohost;

#ifdef MXGEMM_VERIFY_M64N64K128
  // Zero C before releasing the GPU so a stale result from an earlier run cannot pass.
  {
    volatile uint16_t *cz = (volatile uint16_t *)rad_device_to_host_address(MXV_C_DEV);
    for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; i++) cz[i] = 0;
#ifdef MXGEMM_DIAG_LATE_COPY
    volatile uint16_t *cl = (volatile uint16_t *)rad_device_to_host_address(MXV_C_LATE_DEV);
    for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; i++) cl[i] = 0;
#endif
  }
#endif

  // printf("start GPU\n");
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  uint32_t finished = 0;
#ifdef MXGEMM_VERIFY_M64N64K128
  uint32_t polls = 0;
  while (!finished && polls < MXV_POLL_LIMIT) {
    SYNC_GPU();
    polls++;
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
  }
#else
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    // uint32_t core0 = READ_MMIO_32(RAD_HOST_GPU_CORES);
    // uint32_t core1 = READ_MMIO_32(RAD_HOST_GPU_CORES + 4);
    // uint32_t core2 = READ_MMIO_32(RAD_HOST_GPU_CORES + 8);
    // uint32_t core3 = READ_MMIO_32(RAD_HOST_GPU_CORES + 12);
    // printf("%d %d %d %d\n", core0, core1, core2, core3);
  }
#endif
  // printf("finished\n");

  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);

#ifdef MXGEMM_VERIFY_M64N64K128
  // Printed only after the GPU is back in reset (console latency is huge on the FPGA).
  {
    const uint16_t *c = (const uint16_t *)rad_device_to_host_address(MXV_C_DEV);
    uint32_t exact = 0, lost = 0, nonfin = 0, bad_rows = 0;
    int64_t max_err = 0;
    char rowmap[MATMUL_M + 1];
    char grid[(MATMUL_M / MXV_TILE) * (MATMUL_N / MXV_TILE + 1) + 1];
    uint32_t gn = 0;
    for (uint32_t r = 0; r < MATMUL_M; r++) {
      rowmap[r] = '.';
      for (uint32_t k = 0; k < MATMUL_N; k++) {
        const uint16_t got = c[r * MATMUL_N + k], exp = C_out_bf16[r][k];
        exact += (got == exp);
        lost += ((got & 0x7fffu) == 0 && (exp & 0x7fffu) != 0);
        if (mxv_nonfinite(got)) { nonfin++; rowmap[r] = 'I'; continue; }
        int64_t err = mxv_micro(got) - mxv_micro(exp);
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (got != exp && rowmap[r] == '.') rowmap[r] = 'x';
      }
      bad_rows += (rowmap[r] != '.');
    }
    rowmap[MATMUL_M] = 0;
    for (uint32_t ti = 0; ti < MATMUL_M / MXV_TILE; ti++) {
      for (uint32_t tj = 0; tj < MATMUL_N / MXV_TILE; tj++) {
        uint32_t bad = 0;
        for (uint32_t r = ti * MXV_TILE; r < (ti + 1) * MXV_TILE; r++)
          for (uint32_t k = tj * MXV_TILE; k < (tj + 1) * MXV_TILE; k++)
            bad += (c[r * MATMUL_N + k] != C_out_bf16[r][k]);
        grid[gn++] = bad == 0 ? '.' : (bad > 9 ? '+' : (char)('0' + bad));
      }
      grid[gn++] = '|';
    }
    grid[gn] = 0;
    if (!finished) printf("[mxv] WATCHDOG: GPU not finished after %u polls\n", polls);
    printf("[mxv] polls %u rows %s\n[mxv] tiles %s\n", polls, rowmap, grid);
    printf("[mxv] C exact %u/%u lost %u nonfin %u bad_rows %u maxerr %d\n", exact,
           MATMUL_M * MATMUL_N, lost, nonfin, bad_rows, (int)max_err);
    printf("MXGEMM %s\n", (finished && exact == MATMUL_M * MATMUL_N) ? "PASS" : "FAIL");
#ifdef MXGEMM_DIAG_LATE_COPY
    mxv_zero_summary("early", c);
    mxv_zero_summary("late", (const uint16_t *)rad_device_to_host_address(MXV_C_LATE_DEV));
#else
    mxv_dump_row(c, 0);
    mxv_dump_row(c, 8);
#endif
  }
#endif

  tohost = 1;
  return 0;
}
