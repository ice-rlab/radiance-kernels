# gcstack_ubench

Microbenchmarks that each drive one bucket of Muon's per-warp GCStack
(`Backend.scala`, the `bucket` priority chain). The harness is done; a bench is
just a loop body.

## Writing a bench

Edit one of the `bench_<bucket>.cpp` stubs, or `cp _template.cpp bench_<name>.cpp`.
Each stub's header says what the RTL counts for that bucket and what masks it.

```cpp
#include "ubench.hpp"
UBENCH_BODY(u) {        // runs per thread
  uint32_t acc = u.tid;
  #pragma unroll 1
  for (uint32_t i = 0; i < u.iters; ++i) {
    /* pattern under test */
    ubench_keep(acc);   // pin in a register, no code emitted
  }
  ubench_sink(u, acc);  // one store so nothing is DCE'd
}
```

`u` gives `tid`, `lane`, `warp`, `nthreads`, `iters` (compile-time),
`gmem` (DRAM scratch), `smem` (SMEM base), `out`. Define `UBENCH_SETUP` before
the include to get a `void ubench_setup()` hook that runs once before launch.
**Don't edit `host.cpp`.** It is the shared CPU-side driver (reset the GPU, wait,
exit), the same file every kernel here uses.

## Build

```bash
source <chipyard>/env.sh
make                               # every bench_*.cpp -> bench_*.soc.elf
make NUM_WARPS=1 ITERS=16384 -B    # global knob overrides
make list
```

Check the hot loop in `bench_<name>.radiance.dump` (symbol `ubench_entry`).

## Buckets (priority order, top wins)

| # | bucket       | counted when                                        | stub |
|---|--------------|-----------------------------------------------------|------|
| 0 | unoccupied   | warp not resident                                   | (launch / early exit) |
| 1 | base         | warp issued this cycle                              | bench_base |
| 2 | control      | fetch discarding after a control hazard             | bench_control, bench_control_miss |
| 3 | sync         | parked at `vx_bar`                                  | bench_sync |
| 4 | ifetch       | ibuf empty, **any** fetch in flight (hits too, not just misses) | bench_ifetch |
| 5 | idle         | ibuf empty, nothing in flight                       | bench_idle |
| 6 | memData      | RS row waits on a memory producer                   | bench_memdata |
| 7 | comData      | RS row waits on a compute producer                  | bench_comdata |
| 8 | struct       | ibuf valid but not admitted: RS full, LSU busy, **or a WAW/WAR hold-back** | bench_struct |

Two of these do not mean what their RTL description string says:

- **ifetch** uses `icacheInFlight`, which is "this warp has an outstanding fetch request"
  (`WarpScheduler.scala`, `icacheInFlights` counter), with no miss check. A fetch that
  hits still counts. 8 warps running a 10-instruction loop that never misses read 70% ifetch.
- **struct** is any `ibuf.valid && !ibuf.ready`. That includes an instruction held back by
  `hasWAW`/`hasWAR` in `Hazard.scala` (its destination register is still being written or
  read), which is a data dependency, not a structural limit. See the gotchas below.

`bench_empty` is the launch/teardown baseline (~10.5k cycles with 8 warps). It only
matches benches that launch the same number of warps; for 1-warp benches don't subtract
it -- the overhead is a few % of each run, so read the shares of the running warps instead.

## Gotchas

- **Divergence trap.** Loop bounds or branches on loaded or tid-derived values
  get wrapped in `vx_split`/`vx_pred`/`vx_join`, which adds a control hazard to
  every iteration. Use constant bounds unless that divergence is the thing you
  are measuring.
- Control hazards include `csrr` (SYSTEM) and `fence`, not just branches.
- Barrier id 0 belongs to `mu_schedule`. Start at `UBENCH_FIRST_FREE_BARRIER`.
- Only one instruction issues per core per cycle, so the sum of `base` across
  warps is at most 1.0. To see a warp in `base` most of the time, use 1 warp.
- **Never reuse a register in a dependency chain.** `x = f(x)` (e.g. `fmul.s a0, a0, m`,
  `lw a0, 0(a0)`) is held back before the RS by `hasWAW`, so the wait shows up as
  **struct**, not comData/memData. Rotate the destination over several registers.
- **The RS has 8 entries shared by all warps** (`numIssueQueueEntries = 8`). With
  rotating registers a warp can admit its whole upcoming chain and keep most of the
  entries; the warps that don't get in are struct. A 2-register ping-pong
  (`x1 = f(x0); x0 = g(x1)`) caps each warp at ~2 entries (the next step is held by
  `hasWAR` until the previous one has read its operand), so at most ~4 warps fit.
- **"Buffer empty" outranks memData/comData/struct.** A loop branch right after a
  long-latency op lets the warp's ibuf drain while it waits, so the wait is counted as
  idle. Unroll so the branch is rare (`.rept` in inline asm).
- **Slow units take one warp at a time:** integer mul/div (`MulDivPipe`:
  `req.ready := !busy || resp.fire`) and FP div/sqrt (iterative, 2 FP32 lanes, so 4
  passes per 8-thread warp). Warps queued for them that have nothing waiting in the RS are
  struct. FP add/mul (CVFPU, 8 lanes) is pipelined.
- **Memory-level parallelism is small.** The L0 data cache tracks 4 outstanding misses
  (`L0dCacheConfig.nMSHRs = 4`). A chain that misses everywhere is limited by that, not by
  latency. Keep per-warp chains inside the L0 (4 KiB total) to give every warp its own
  waiting load.
- Separate data per warp: if warps chase the same lines, a warp that follows another hits
  in the cache, finishes early, and the rest of the run reads as `unoccupied`.

## Measured on the FPGA

`xilinx_vcu118_firesim_radiance_singleclustersyn_1GB_no_nic`, `AUTO=20`, 2026-11-11/12.

| bench | warps | pattern | result |
|---|---|---|---|
| bench_base | 1 | 1024 independent adds, 4 regs | **base 93%** |
| bench_control | 1 | 1024 never-taken branches (8 KiB, in icache) | control 23% |
| bench_control_miss | 1 | 4096 never-taken branches (32 KiB, misses) | **control 45%** (top bucket) |
| bench_sync | 8 | warp 0 does 1024 adds, all 8 meet at a barrier | **sync 6.43 of 8** |
| bench_memdata | 8 | pointer chase, 4 rotating regs, per-warp ring in L0 | **memData 5.02 of 8** |
| bench_comdata | 4 | FP div / int mul, 2-register ping-pong | **comData 84% of each running warp** |
| bench_fdiv | 1 | 1024 dependent fdiv.s | comData 99%, **~86 cycles per warp fdiv** |
| bench_fdiv8 | 8 | same, 8 warps | comData 2.00 of 8; divider does **not** overlap warps (81 cyc/fdiv) |

Hardware numbers used above: L0 icache 16 KiB direct-mapped, 32 B lines; L0 dcache 4 KiB,
64 B lines, 4 MSHRs; L1 64 KiB; RS 8 entries; FP32 div 2 lanes. Main-memory load ~170
cycles (1 warp, dependent); integer `mul` ~24 cycles per warp.

## Running

Copy `bench_<name>.soc.elf` into the FireSim radiance workload like any other
kernel. `gcstack_post_run_hook.py` writes `gcstack_summary.{txt,csv}`.

The runs above used `sims/firesim-workloads/radiance-workload/gcub_<name>.soc.elf` with a
marshal config `marshal-configs/gcub_<name>.json` (copy of `mx_gemmini.json` with the name
and `bin` changed), then:

```bash
make radiance-raw-workload JSON=gcub_<name>
make run HW=xilinx_vcu118_firesim_radiance_singleclustersyn_1GB_no_nic WORKLOAD=gcub_<name>.json AUTO=20
```

None of these benches use Gemmini, so the ggstack PNGs are flat; the GCStack data is in the
Muon panel of `ggstack_timeline_tile2.html` (zoom to the few windows where the GPU ran).
