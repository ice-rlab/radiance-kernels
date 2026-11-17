#pragma once

// gcstack_ubench harness: everything a bench_*.cpp needs except the loop body.
//
// A bench file is just:
//
//     #include "ubench.hpp"
//     UBENCH_BODY(u) {
//       // per-thread code; runs on UBENCH_NUM_WARPS warps x MU_NUM_THREADS lanes
//     }
//
// and the harness supplies the scratch buffers, the mu_schedule launch, main(),
// and an anti-DCE sink. host.cpp (shared, never edited) resets the GPU, waits
// for it to finish, and ends the run.
//
// Knobs (-D, or `make NUM_WARPS=.. ITERS=..` which applies to every bench).
// A bench may set its own default with `#define` *before* the include, but a
// make-level override then wins only if the bench guards it with #ifndef.
//   UBENCH_NUM_WARPS   warps launched per core (occupancy)           (default 8)
//   UBENCH_ITERS       loop count handed to the body as u.iters      (default 4096)
//   UBENCH_GMEM_WORDS  size of the u.gmem DRAM scratch, in words     (default 64K)
//   UBENCH_SETUP       if defined, main() calls `void ubench_setup()` (which the
//                      bench defines) once before launch, e.g. to build a
//                      pointer-chase ring in ubench_gmem.
//
// Divergence trap: the compiler cannot prove that values loaded from memory or
// derived from u.tid/u.lane/u.warp are warp-uniform, so any loop bound or
// branch on them gets wrapped in vx_split / vx_pred / vx_join -- a vx_pred is a
// control hazard, so that adds a control stall to EVERY iteration. That is why
// u.iters is a compile-time constant. Keep loop bounds constant unless the
// divergence is what you're measuring, and check the loop in the .radiance.dump.

#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#ifndef UBENCH_NUM_WARPS
#define UBENCH_NUM_WARPS 8
#endif

#ifndef UBENCH_ITERS
#define UBENCH_ITERS 4096
#endif

#ifndef UBENCH_GMEM_WORDS
#define UBENCH_GMEM_WORDS (64 * 1024)
#endif

static_assert(UBENCH_NUM_WARPS >= 1 && UBENCH_NUM_WARPS <= MU_NUM_MAX_WARPS,
              "UBENCH_NUM_WARPS out of range");

extern "C" uint32_t __mu_num_warps = UBENCH_NUM_WARPS;

// Barrier id 0 is taken by mu_schedule() (with MU_NUM_CORES participants).
// Reusing an id with a different participant count hangs the core, so benches
// should start their own ids here.
#define UBENCH_FIRST_FREE_BARRIER 1

// Per-thread view handed to the body.
struct Ubench {
  uint32_t tid;              // thread id in the threadblock, 0 .. nthreads-1
  uint32_t lane;             // 0 .. MU_NUM_THREADS-1
  uint32_t warp;             // 0 .. UBENCH_NUM_WARPS-1
  uint32_t nthreads;         // UBENCH_NUM_WARPS * MU_NUM_THREADS (per core)
  static constexpr uint32_t iters = UBENCH_ITERS;
  __global uint32_t* gmem;   // UBENCH_GMEM_WORDS words of DRAM scratch
  __shared uint32_t* smem;   // SMEM scratch (MU_SMEM_SIZE_BYTES)
  __global uint32_t* out;    // one sink word per thread; see ubench_sink()
};

// Pin a value in a register: stops the compiler from folding, hoisting, or
// sinking code that produces it. Emits no instructions.
template <typename T>
static inline void ubench_keep(T& v) {
  asm volatile("" : "+r"(v) : : "memory");
}

// Make a result observable so the body is not dead-code-eliminated.
// One global store per call -- call it once after the timed loop, not inside it.
template <typename T>
static inline void ubench_sink(const Ubench& u, T v) {
  u.out[u.tid] = __builtin_bit_cast(uint32_t, v);
}

// ---- harness ---------------------------------------------------------------

__global uint32_t ubench_gmem[UBENCH_GMEM_WORDS] = {};
__global uint32_t ubench_out[MU_NUM_MAX_WARPS * MU_NUM_THREADS] = {};

struct UbenchArgs {
  __global uint32_t* gmem;
  __global uint32_t* out;
};

UbenchArgs ubench_args = {
  .gmem = nullptr,
  .out = nullptr,
};

static inline __attribute__((always_inline)) void ubench_body(const Ubench& u);

static void ubench_entry(void* arg,
                         uint32_t tid_in_threadblock,
                         uint32_t threads_per_threadblock,
                         uint32_t /*threadblock_id*/) {
  auto* a = reinterpret_cast<UbenchArgs*>(arg);
  const Ubench u = {
    .tid = tid_in_threadblock,
    .lane = tid_in_threadblock % MU_NUM_THREADS,
    .warp = tid_in_threadblock / MU_NUM_THREADS,
    .nthreads = threads_per_threadblock,
    .gmem = a->gmem,
    .smem = reinterpret_cast<__shared uint32_t*>(0x0),
    .out = a->out,
  };
  ubench_body(u);
}

#ifdef UBENCH_SETUP
void ubench_setup();
#endif

int main() {
  ubench_args.gmem = ubench_gmem;
  ubench_args.out = ubench_out;
#ifdef UBENCH_SETUP
  ubench_setup();
#endif
  mu_schedule(ubench_entry, &ubench_args, UBENCH_NUM_WARPS);
  return 0;
}

#define UBENCH_BODY(u) \
  static inline __attribute__((always_inline)) void ubench_body(const Ubench& u)
