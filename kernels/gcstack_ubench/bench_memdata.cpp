// bench_memdata -- bucket 6 "memData": an RS entry of this warp waits on an
// operand whose outstanding producer is a memory op (load).
//
// How this bench works, in simple terms:
//   Every warp follows a chain of pointers through memory: each load reads the
//   address of the next load, so it cannot start until the previous one is back.
//   Each warp has its OWN small chain of 8 slots (512 B); all 8 chains together are
//   4 KiB, exactly the size of the L0 data cache, so after the first lap every load
//   HITS in the L0. Each warp still has to wait for its previous load (memData), but
//   no warp has to queue for the memory system.
//
// Why the chains must fit in the L0 (what went wrong with bigger chains):
//   The L0 data cache can only track 4 misses at a time (L0dCacheConfig nMSHRs = 4).
//   With chains bigger than the caches, every load missed, only ~2-3 loads could be in
//   flight for all 8 warps together, and the other warps' loads sat waiting for the
//   memory system -- counted as "struct" (4.2-4.8 of 8), with memData near 0.
//
// Why there is an `addi` after every load:
//   memData is only counted when an instruction of the warp is already INSIDE the
//   reservation station waiting on a load (ReservationStation.scala, busyMemTable).
//   A load that depends on a load never gets in there -- it waits outside for the
//   load/store unit, which is counted as "struct" (what happened with loads alone).
//   The `addi a0, a0, 0` is a plain ALU op, so it gets in and waits on the load, and
//   the warp is counted as memData. It adds 0, so the chain is unchanged.
//
// Why 256 load+addi pairs per loop:
//   After a loop branch the warp stops fetching until the branch finishes, so its
//   instruction buffer can run empty, which is counted as idle. One branch per 512
//   instructions (4 KiB, fits in the icache) makes that rare.
#define UBENCH_SETUP
#ifndef UBENCH_ITERS
#define UBENCH_ITERS 64
#endif

#define RING_LEN 8       // slots per warp
#define RING_STRIDE 16   // words between slots (16 x 4 B = 64 B, one L0d line)
#define UBENCH_GMEM_WORDS (MU_NUM_MAX_WARPS * RING_LEN * RING_STRIDE)  // 4 KiB = the L0d

#include "ubench.hpp"

// Word index of slot s of warp w's ring.
static inline uint32_t slot(uint32_t w, uint32_t s) {
  return (w * RING_LEN + s) * RING_STRIDE;
}

// Each slot holds the ADDRESS of the next slot in the same warp's ring.
void ubench_setup() {
  for (uint32_t w = 0; w < MU_NUM_MAX_WARPS; ++w)
    for (uint32_t s = 0; s < RING_LEN; ++s)
      ubench_gmem[slot(w, s)] =
          reinterpret_cast<uint32_t>(&ubench_gmem[slot(w, (s + 1) % RING_LEN)]);
}

// Why the chain ROTATES over 4 registers (p1 = *p0, p2 = *p1, p3 = *p2, p0 = *p3):
//   An instruction is only let into the reservation station if its destination register
//   is not still waiting to be written or read by an earlier unfinished instruction
//   (Hazard.scala: hasWAW / hasWAR). `lw a0, 0(a0)` writes the register the previous
//   load is still writing, so it was blocked OUTSIDE the reservation station -- counted
//   as "struct" (4.2-4.8 of 8), not memData. Rotating keeps the same one-after-another
//   chain, but each load can get in and wait there for the previous load.
UBENCH_BODY(u) {
  uint32_t p0 = reinterpret_cast<uint32_t>(&u.gmem[slot(u.warp, 0)]);
  uint32_t p1 = 0, p2 = 0, p3 = 0;

  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    // 256 dependent loads (64 x 4): each load's address is the previous load's result.
    asm volatile(".rept 64\n\t"
                 "lw.global %1, 0(%0)\n\t"
                 "lw.global %2, 0(%1)\n\t"
                 "lw.global %3, 0(%2)\n\t"
                 "lw.global %0, 0(%3)\n\t"
                 ".endr"
                 : "+r"(p0), "+r"(p1), "+r"(p2), "+r"(p3) :: "memory");
  }

  ubench_sink(u, p0);
}
