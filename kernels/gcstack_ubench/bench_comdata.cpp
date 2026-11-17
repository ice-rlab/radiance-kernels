// bench_comdata -- bucket 7 "comData": an RS entry waits on an operand whose
// outstanding producer is a compute op (and no memory-blocked row exists).
//
// Measured on the FPGA (2026-11-12): every running warp 83.6% comData, 14.3% struct.
//
// How this bench works, in simple terms:
//   4 warps each run a chain that alternates an FP divide (fdiv.s, ~86 cycles per warp)
//   and an integer multiply (mul, ~24 cycles per warp). Each step needs the previous
//   step's result, so a warp spends almost all its time waiting for a result: comData.
//   Divide by 1.0 and multiply by 1 leave the value unchanged, so it never overflows.
//
// Why these choices (each one fixed a run that came out as "struct" instead):
//
//   * Destination register must change every step. An instruction is only let into the
//     reservation station (RS) if its destination register is not still being written or
//     read by an earlier unfinished instruction (Hazard.scala hasWAW / hasWAR). A chain
//     like `fdiv.s a0, a0, x` is held back OUTSIDE the RS, and GCStack counts that as
//     struct (8-warp fmul chain: comData 0.00, struct 4.91).
//
//   * Two registers ping-ponging (x1 = x0 / 1.0, then x0 = x1 * 1), not 4 rotating.
//     The RS has only 8 entries for ALL warps. With 4 rotating registers a warp could put
//     its whole upcoming chain into the RS and keep most of the entries; only ~2 warps
//     got in (comData ~2 of 8). With two registers the next divide writes x1, which the
//     waiting multiply still has to read, so it is held back until that multiply has
//     gone -- each warp holds at most 2 entries.
//
//   * Slow units. The divider (2 FP32 lanes, iterative) and the integer multiplier each
//     take one warp's operation at a time, so a warp waiting for its turn still has its
//     next step sitting in the RS waiting on a result -- comData. Fast pipelined ops
//     (fmul/fadd) finish too quickly for many warps to be waiting at once.
//
//   * 4 warps, not 8. 4 warps x 2 entries = the 8-entry RS, so no warp is locked out.
//     With 8 warps (16 entries needed) about half were always outside: comData 2.31 of 8.
//     The divider runs at the same ~81 cycles per warp-divide either way.
//
//   The remaining ~14% struct is the short hasWAR hold-back after each multiply, which
//   GCStack counts as struct although it is a data dependency.
#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 4
#endif
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 4
#endif

#include "ubench.hpp"

UBENCH_BODY(u) {
  // FP32 bit patterns held in integer registers (Muon uses zfinx: FP ops on x regs).
  uint32_t x0 = 0x3fc00000u;  // 1.5f
  uint32_t x1 = 0;
  uint32_t fone = 0x3f800000u;  // 1.0f, for the divide
  uint32_t ione = 1;            // 1, for the integer multiply
  ubench_keep(x0);
  ubench_keep(fone);
  ubench_keep(ione);

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 1024 dependent steps (512 x 2) = 8 KiB: FP div, int mul, FP div, int mul, ...
    asm volatile(".rept 512\n\t"
                 "fdiv.s %1, %0, %2\n\t"
                 "mul %0, %1, %3\n\t"
                 ".endr"
                 : "+r"(x0), "+r"(x1)
                 : "r"(fone), "r"(ione));
  }

  ubench_sink(u, x0);
}
