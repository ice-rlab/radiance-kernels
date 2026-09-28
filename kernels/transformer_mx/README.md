# transformer_mx

One MX-FP8 BERT-style encoder layer (Q/K/V/O projections, per-head softmax attention, FFN with
GELU, two LayerNorms) on Radiance: GEMMs on the Gemmini mesh, epilogues on muon SIMT.

**This is a stability build.** It follows `../flash_attention_mx_stable` (FA_ST_NOOVL, 144/144
tile-images at NT72): correct under schedule perturbation first, cycles second. Don't add
mesh/SIMT overlap here.

## Structure

| file | role |
|---|---|
| `kernel.cpp` | the layer as a sequence of barriered stages |
| `tf_gemm.hpp` | one fully serialized GEMM (rules 1-8 in its header, each with its FA citation) |
| `transformer_mx_impl.hpp` | MX quantization and epilogues, native bf16 only |
| `tf_beacon.h` | fixed GMEM addresses and stage names, shared by kernel and host |
| `host.cpp` | waits for the GPU, then prints progress, output checksum and golden check |
| `gen_data.py` / `golden_tf.py` | operands (`include/tf_data.h`) and op-for-op golden (`include/tf_golden.h`) |

Every GEMM runs as two stages:
1. Warp 0 drains, moves in A and B per 16x16 tile, and writes the scales (one thread,
   ascending), then fences. It drains again, sets `CONFIG_SCALE_MEM`, fences, issues `LOOP_WS`
   and drains on `runningLoops` and then busy. Every other warp waits at the barrier.
2. All threads copy C out of SMEM into DRAM.

SIMT never touches SMEM at any other time. There is one matmul per output tile, with `TILE_K == K`.

The epilogues use no fp32 at all: every op is a named `f*.h` instruction. The FA campaign found the
bf16<->fp32 path unvalidated, and it produced inf/garbage there. SIMT threads own whole 32-bit
words: rows for bf16 data, 4-row or 4-column groups for fp8 and scales.

## Build

```sh
python3 gen_data.py --hidden 128 --expansion 256 --heads 4 --seq 32 --tile_n 64 --out include/tf_data.h
python3 golden_tf.py --test
python3 golden_tf.py --emit include/tf_golden.h --hidden 128 --expansion 256 --heads 4 --seq 32 --tile_n 64
make RISCV=<chipyard>/.conda-env/riscv-tools      # or source chipyard's env.sh first
```

If `RISCV` is unset, the build fails inside libc++ (`ldiv_t`, `FP_NAN` undeclared), because
`common.mk` derives the newlib include dir and the rv64 host toolchain from it. The host compiles
the golden comparison in only when `include/tf_golden.h` exists, and its dims must match `tf_data.h`.

## Bring-up ladder

`kernel.cpp` holds only the rungs that already pass on the FPGA. A rung goes in only after every
rung before it passes, and `host.cpp` gains that rung's check at the same time:

| rung | adds | checked buffer |
|---|---|---|
| 1 | `mxquant_A(X)`, Wq GEMM tiles (Gemmini), Q bias | `g_q` vs `G_q` |
| 2 | K and V projections + biases | `g_k`, `g_v` |
| 3 | head 0: gather, transpose, Q@K^T (Gemmini), softmax | `g_s` vs `G_s0` |
| 4 | all heads, S@V (Gemmini), scatter | `g_ctx` |
| 5 | Wo + bias, LN1, residual | `g_res` |
| 6 | FF1 + bias, GELU | `g_ff` |
| 7 | FF2 + bias, LN2, residual | output vs `G_out` |

Buffers the host checks need external linkage (see `kernel.cpp`), and `tf_syms.h` has to list them.
Rung 1 is currently in the tree.

## Gates

The uartlog ends with one map and one compare line per checked buffer, then the verdict:

```
[host] q rows ................................
[host] q tiles ................|................|...
[host] q   exact 1234/4096 lost 0 nonfin 0(0 rows) maxerr ... tol ... OK
[host] RUNG 1 PASS
```

* **PASS** needs the DONE flag, no watchdog timeout, and every checked buffer OK. The tolerance is
  depth x (max|golden| + 1e-3), the same as `golden_tf._phase_tol`. The golden models `fexp.h` as
  correctly rounded, so don't expect bit-exact.
* **Static checks, before every FPGA run:** run
  `python3 ../flash_attention_mx_stable/fa_baraudit.py <kernel .s>`. Every `vx_bar` must be at depth
  0, with no duplicated ids. Build the `.s` by rerunning the `kernel.cpp` compile line with `-S`.
* **Run looks hung:** read `/home/fpga/iso2107-FIRESIM_RUNS_DIR/sim_slot_0/uartlog` on the run host
  (`fpga@192.168.0.115`) before the run is stopped. A stopped run's copied uartlog has no `[host]` lines. `make run` without `AUTO` samples
  autocounters only once per 2^32 cycles, so the autocounter file can't show where a short kernel
  stalled.
* **Stability**, once a rung passes: the result must not change under
  `make clean && make EXTRA_MU_CFLAGS=-DTF_PHASE1` (also `TF_PHASE2` and `TF_PHASE3`, with
  `-DTF_PHASE_BOTH` as the control).

`-DTF_NO_BEACON` removes the per-stage GMEM stamps. The host then can't see the DONE flag, so every
rung reports FAIL.
