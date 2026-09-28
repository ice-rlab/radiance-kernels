#!/usr/bin/env python3
"""Generate the MXFP8 weight/activation data header for transformer_mx (one encoder layer).

PERF-FIRST / correctness-unverified: emits random, magnitude-bounded fp8 weights + e8m0
block scales (same operand distribution as gemm_mxgemmini's generator, so long dot products
don't saturate the fp8 accumulators), the bf16 bias / LayerNorm gamma+beta vectors, the bf16
layer input, and the dimension #defines.

The operand *values* are produced by build_operands() (pure, returns numpy arrays in logical
[K][N] form); this file emits them, and golden_tf.py consumes the same function so the golden
sees byte-identical operands. build_operands() is the single source of operand truth.

N-TILED WEIGHTS (so the driver can loop output-column tiles with contiguous operands):
each weight W[K][N] is split into N/TILE_N column blocks and stacked along axis 0:
  fp8    W[(N/TILE_N)*K][TILE_N]     -- block nj at rows [nj*K : (nj+1)*K]
  scales W_s[(N/TILE_N)*GK][TILE_N]  -- block nj at rows [nj*GK : (nj+1)*GK], GK = K/32
A weight's A operand (the activation) and K (attention key) are produced/quantized at runtime.

Usage: python3 gen_data.py --hidden 768 --expansion 3072 --heads 12 --seq 128 \
                           --tile_n 128 --out include/tf_data.h
"""
import argparse
import hashlib
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden"))
from golden import rand_fp8   # magnitude-bounded fp8 e4m3 operand codes (|value| < 2)

GROUP = 32

# Logical weights (K x N) generated for each name, in build order.
#   name -> (K_expr, N_expr) as callables of the config dict.
WEIGHT_SHAPES = {
    "Wq":  lambda c: (c["H"], c["H"]),
    "Wk":  lambda c: (c["H"], c["H"]),
    "Wv":  lambda c: (c["H"], c["H"]),
    "Wo":  lambda c: (c["H"], c["H"]),
    "FF1": lambda c: (c["H"], c["E"]),   # [hidden][expansion]
    "FF2": lambda c: (c["E"], c["H"]),   # [expansion][hidden]
}
# Bias / affine vectors: name -> (length_expr, kind).
BIAS_SPECS = [
    ("bq", "H", "bias"), ("bk", "H", "bias"), ("bv", "H", "bias"), ("bo", "H", "bias"),
    ("ff1_b", "E", "bias"), ("ff2_b", "H", "bias"),
    ("ln1_g", "H", "gamma"), ("ln2_g", "H", "gamma"),
    ("ln1_b", "H", "beta"), ("ln2_b", "H", "beta"),
]


def _seed(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def f32_to_bf16(x):
    """Round-to-nearest-even float32 -> bf16 (uint16 codes)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Operand generation (pure). Each operand is independently seeded by its name,
# so build order does not affect any operand's value.
# ---------------------------------------------------------------------------
def gen_weight(name, K, N):
    """Logical fp8 weight W[K][N] (uint8 codes) + e8m0 scales S[K/32][N] (uint8 codes)."""
    assert K % GROUP == 0
    GK = K // GROUP
    rng = np.random.default_rng(_seed(name))
    W = rand_fp8(rng, K * N).reshape(K, N).astype(np.uint8)
    S = rng.integers(0x7B, 0x83, size=(GK, N), dtype=np.uint8)     # scales [GK][N], near 2^0
    return W, S


def e4m3_value(codes):
    """e4m3 byte codes -> float64 values (same decode as golden_tf.e4m3_to_f32)."""
    c = np.asarray(codes, dtype=np.int32)
    sign = np.where((c & 0x80) != 0, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 0x7
    return sign * np.where(e == 0, (m / 8.0) * 2.0 ** (1 - 7), (1.0 + m / 8.0) * 2.0 ** (e - 7))


def dequant_bf16(W, S):
    """Logical fp8 weight [K][N] + e8m0 scales [K/32][N] -> bf16 codes [K][N]. Used by the all-SIMT
    kernel path (-DTF_ALL_SIMT), which multiplies in bf16 instead of on Gemmini."""
    se = np.repeat(np.asarray(S, dtype=np.int32) - 127, GROUP, axis=0)
    return f32_to_bf16((e4m3_value(W) * 2.0 ** se).astype(np.float32))


def gen_bias(name, n, kind):
    """bf16 (uint16) code vector for a bias (random), gamma (ones), or beta (zeros)."""
    rng = np.random.default_rng(_seed(name))
    if kind == "gamma":
        v = np.ones(n, dtype=np.float32)
    elif kind == "beta":
        v = np.zeros(n, dtype=np.float32)
    else:
        v = rng.standard_normal(n).astype(np.float32) * 0.02
    return f32_to_bf16(v)


def gen_input(S, H):
    """bf16 (uint16) code layer input X[seq][hidden]."""
    rng = np.random.default_rng(_seed("X_in"))
    return f32_to_bf16(rng.standard_normal((S, H)) * 0.5)


def build_operands(hidden, expansion, heads, seq, tile_n, ln_eps=1e-5):
    """The single source of operand truth. Returns a dict of numpy arrays in LOGICAL form:

      dims:    H, E, NH, S, HD, TN, LN_EPS, ATTN_SCALE
      X_in:    bf16 codes [S][H]
      weights: {name: (W_fp8[K][N] u8, S_codes[K/32][N] u8, K, N)}
      biases:  {name: bf16 codes [n]}   (includes ln gammas/betas)

    Weights are un-tiled here; gen_data.py tiles them for the header, golden_tf.py uses them
    directly. Assertions mirror the header contract (dims divisible by heads/tile_n/GROUP)."""
    H, E, NH, S, TN = hidden, expansion, heads, seq, tile_n
    assert H % NH == 0 and H % TN == 0 and E % TN == 0, "hidden/expansion must be multiples of tile_n"
    assert H % GROUP == 0 and E % GROUP == 0 and S % GROUP == 0, "dims must be multiples of 32"
    HD = H // NH
    assert HD % GROUP == 0, "head_dim (hidden/heads) must be a multiple of 32"
    cfg = {"H": H, "E": E, "NH": NH, "S": S, "TN": TN, "HD": HD,
           "LN_EPS": ln_eps, "ATTN_SCALE": 1.0 / (HD ** 0.5)}

    weights = {}
    for name, shp in WEIGHT_SHAPES.items():
        K, N = shp(cfg)
        W, Sc = gen_weight(name, K, N)
        weights[name] = (W, Sc, K, N)

    biases = {}
    for name, dim_key, kind in BIAS_SPECS:
        n = cfg[dim_key]
        biases[name] = gen_bias(name, n, kind)

    return {**cfg, "X_in": gen_input(S, H), "weights": weights, "biases": biases}


# ---------------------------------------------------------------------------
# Header emit
# ---------------------------------------------------------------------------
def _rows(fmt, arr):
    return ",\n".join("  { " + ", ".join(fmt % v for v in row) + " }" for row in arr)


def emit_u8(f, name, R, C, arr):
    f.write(f"static const uint8_t {name}[{R}][{C}] = {{\n{_rows('0x%02x', arr)}\n}};\n\n")


def emit_bf16(f, name, n, vec):
    vals = ", ".join(f"0x{v:04x}" for v in vec)
    f.write(f"static const uint16_t {name}[{n}] = {{ {vals} }};\n\n")


def emit_bf16_2d(f, name, R, C, arr):
    f.write(f"static const uint16_t {name}[{R}][{C}] = {{\n"
            + ",\n".join("  { " + ", ".join(f"0x{v:04x}" for v in row) + " }" for row in arr)
            + "\n};\n\n")


def emit_weight_ntiled(f, name, W, S, K, N, TN):
    """Emit fp8 [(N/TN)*K][TN] + e8m0 scales [(N/TN)*GK][TN], N split into TN-column blocks."""
    assert N % TN == 0
    GK, nt = K // GROUP, N // TN
    Wb = np.concatenate([W[:, j * TN:(j + 1) * TN] for j in range(nt)], axis=0)   # [nt*K][TN]
    Sb = np.concatenate([S[:, j * TN:(j + 1) * TN] for j in range(nt)], axis=0)   # [nt*GK][TN]
    emit_u8(f, name, nt * K, TN, Wb)
    emit_u8(f, name + "_s", nt * GK, TN, Sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--expansion", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--tile_n", type=int, default=128)
    ap.add_argument("--ln_eps", type=float, default=1e-5)
    ap.add_argument("--out", type=str, default="include/tf_data.h")
    a = ap.parse_args()

    op = build_operands(a.hidden, a.expansion, a.heads, a.seq, a.tile_n, a.ln_eps)
    H, E, NH, S, TN, HD = op["H"], op["E"], op["NH"], op["S"], op["TN"], op["HD"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("// @generated by gen_data.py -- do not edit\n")
        f.write("#ifndef TF_DATA_H\n#define TF_DATA_H\n#include <stdint.h>\n\n")
        f.write(f"#define TF_HIDDEN {H}\n#define TF_EXP {E}\n#define TF_HEADS {NH}\n")
        f.write(f"#define TF_SEQ {S}\n#define TF_HEAD_DIM {HD}\n#define TF_TN {TN}\n")
        f.write(f"#define TF_LN_EPS {a.ln_eps:.8e}f\n")
        f.write(f"#define TF_ATTN_SCALE {op['ATTN_SCALE']:.8e}f\n\n")

        f.write("// ---- layer input x[seq][hidden] (bf16; quantized to fp8 on device) ----\n")
        emit_bf16_2d(f, "X_in", S, H, op["X_in"])

        f.write("// ---- projection weights (N-tiled fp8 [nt*K][TN] + scales [nt*GK][TN]) ----\n")
        for w in ("Wq", "Wk", "Wv", "Wo"):
            W, Sc, K, N = op["weights"][w]
            emit_weight_ntiled(f, w, W, Sc, K, N, TN)
        f.write("// ---- FFN weights ----\n")
        for w in ("FF1", "FF2"):
            W, Sc, K, N = op["weights"][w]
            emit_weight_ntiled(f, w, W, Sc, K, N, TN)

        f.write("// ---- dequantized bf16 weights [K][N], for the all-SIMT kernel path (-DTF_ALL_SIMT) ----\n")
        for w in ("Wq", "Wk", "Wv", "Wo", "FF1", "FF2"):
            W, Sc, K, N = op["weights"][w]
            emit_bf16_2d(f, w + "_bf", K, N, dequant_bf16(W, Sc))

        f.write("// ---- biases (bf16) ----\n")
        for b in ("bq", "bk", "bv", "bo", "ff1_b", "ff2_b"):
            v = op["biases"][b]
            emit_bf16(f, b, len(v), v)
        f.write("// ---- LayerNorm gamma/beta (bf16) ----\n")
        for g in ("ln1_g", "ln2_g", "ln1_b", "ln2_b"):
            v = op["biases"][g]
            emit_bf16(f, g, len(v), v)

        f.write("#endif // TF_DATA_H\n")

    print(f"wrote {a.out}  (hidden={H} exp={E} heads={NH} seq={S} head_dim={HD} tile_n={TN})")


if __name__ == "__main__":
    main()
