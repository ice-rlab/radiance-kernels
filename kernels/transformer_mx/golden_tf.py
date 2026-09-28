#!/usr/bin/env python3
"""golden_tf.py -- reference golden + per-primitive tests for the transformer_mx kernel.

Why this exists: the gemmini reference transformer.c is a *cycle-count* benchmark (zero static
operands, int8, no golden). It cannot verify transformer_mx's numerics. This module builds an
independent high-precision golden that reproduces kernel.cpp's exact op graph so a per-phase
comparison flags real kernel bugs while tolerating the expected fp8/bf16/approximation noise.

Two engines, matching the device (see lib/golden/golden.py docstring):
  * GEMMs go through the bit-exact mx_matmul backend (mx_golden, verified vs spike libgemmini).
    The A/B fp8 operands + e8m0 scales are produced by a faithful numpy port of the device's
    mxquant_A / mxquant_B (floor-log2 block scale targeting exponent 0; truncating e4m3), so the
    bytes fed to mx_matmul equal what the device computes -> the GEMM output is bit-exact.
  * SIMT epilogues (bias_add / softmax / gelu / layernorm / residual / transpose) mirror the *_g
    helpers in transformer_mx_impl.hpp OP FOR OP: the device does all of this math in native
    bf16 (one fadd.h/fmul.h/fdiv.h/fexp.h... per op, no fp32), so every op here is rounded to bf16
    with hb() in the same order, including the linear sum chains and the 6-step Newton rsqrt.
    fexp.h is modelled as a correctly rounded exp; that assumption is the remaining source of
    device-vs-golden difference, and it lands in the comparison tolerance.

Usage:
  python3 golden_tf.py --test                     # C: per-primitive unit tests (host-only)
  python3 golden_tf.py --emit include/tf_golden.h # B: emit per-phase expected tensors
      [--hidden .. --expansion .. --heads .. --seq .. --tile_n .. --ln_eps ..]
  python3 golden_tf.py --check <dump_dir>         # B: compare device DRAM dumps vs golden
"""
import argparse
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                  # for gen_data (same dir)
sys.path.insert(0, str(_HERE.parents[1] / "lib" / "golden"))    # radiance-kernels/lib/golden
import golden as G                       # noqa: E402  (mx_matmul, rand_fp8, bf16 helpers)
from gen_data import build_operands, GROUP   # noqa: E402  (single source of operands)

INV_SQRT2 = 0.70710678118654752


# ============================ bf16 <-> f32 ============================
def to_bf16(x):
    """float32 -> bf16 codes (round-to-nearest-even), matching the device float->bf16 store."""
    u = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + np.uint64(0x7FFF)
    return ((u + r) >> 16).astype(np.uint16)


def from_bf16(c):
    """bf16 codes -> float32."""
    return (np.asarray(c, dtype=np.uint32) << 16).view(np.float32)


# ============================ device fp8 (e4m3) quantization ============================
# Faithful numpy port of transformer_mx_impl.hpp:bf16_to_e4m3_scaled (truncating, branchless)
# and the mxquant_A / mxquant_B block-scale rule (se = floor_log2(block_max), targeting exp 0).
EMAX, EMIN, BIAS = 8, -6, 7


def _e4m3_scaled(codes, se):
    """bf16 codes -> e4m3 bytes after dividing by 2^se (se broadcastable to codes' shape)."""
    b = codes.astype(np.int32)
    se = np.asarray(se, dtype=np.int32)
    exp = (b >> 7) & 0xFF
    m3 = (b >> 4) & 0x7                       # top 3 mantissa bits -> truncation
    E = exp - 127 - se
    over = E > EMAX
    E = np.where(over, EMAX, E)
    m3 = np.where(over, 6, m3)
    clampm = (E == EMAX) & (m3 > 6)
    m3 = np.where(clampm, 6, m3)
    code = ((b >> 8) & 0x80) | ((E + BIAS) << 3) | m3
    keep = ~((exp == 0) | (E < EMIN))
    return np.where(keep, code, 0).astype(np.uint8)


def _block_se(abs_codes, axis):
    """e8m0 exponent se = floor_log2(max |bf16| in block) = (max_abs_code >> 7 & 0xff) - 127."""
    mx = abs_codes.max(axis=axis)            # max nonneg bf16 code == max magnitude
    return ((mx >> 7) & 0xFF).astype(np.int32) - 127


def mxquant_A(X):
    """A operand [M][K] bf16 -> fp8 [M][K] + e8m0 scales [K/32][M]. Groups along K (columns)."""
    M, K = X.shape
    GK = K // GROUP
    fp8 = np.empty((M, K), np.uint8)
    sc = np.empty((GK, M), np.uint8)
    for g in range(GK):
        blk = X[:, g * GROUP:(g + 1) * GROUP]           # [M,32]
        se = _block_se(blk & 0x7FFF, axis=1)            # [M]
        sc[g, :] = (se + 127).astype(np.uint8)
        fp8[:, g * GROUP:(g + 1) * GROUP] = _e4m3_scaled(blk, se[:, None])
    return fp8, sc


def mxquant_B(X):
    """B operand [K][N] bf16 -> fp8 [K][N] + e8m0 scales [K/32][N]. Groups along K (rows)."""
    K, N = X.shape
    GK = K // GROUP
    fp8 = np.empty((K, N), np.uint8)
    sc = np.empty((GK, N), np.uint8)
    for g in range(GK):
        blk = X[g * GROUP:(g + 1) * GROUP, :]           # [32,N]
        se = _block_se(blk & 0x7FFF, axis=0)            # [N]
        sc[g, :] = (se + 127).astype(np.uint8)
        fp8[g * GROUP:(g + 1) * GROUP, :] = _e4m3_scaled(blk, se[None, :])
    return fp8, sc


def e4m3_to_f32(code):
    """Decode e4m3 byte -> float32 (for round-trip tests / dequant)."""
    c = np.asarray(code, dtype=np.int32)
    sign = np.where((c & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = (c >> 3) & 0xF
    man = c & 0x7
    val = np.where(exp == 0,
                   (man.astype(np.float32) / 8.0) * (2.0 ** (1 - BIAS)),        # subnormal
                   (1.0 + man.astype(np.float32) / 8.0) * (2.0 ** (exp - BIAS)))  # normal
    return (sign * val).astype(np.float32)


def dequant_A(fp8, sc):
    """fp8 [M][K] + scales [K/32][M] -> f32 [M][K] (device dequant: value * 2^se)."""
    M, K = fp8.shape
    se = sc.astype(np.int32) - 127                       # [K/32][M]
    se_full = np.repeat(se, GROUP, axis=0).T             # [M][K]
    return (e4m3_to_f32(fp8) * (2.0 ** se_full)).astype(np.float32)


# ============================ GEMM (bit-exact mesh backend) ============================
def mm_aw(A_bf16, W_fp8, W_sc, M, N, K):
    """C = A @ W with A a bf16 activation (device-quantized) and W a static fp8 weight."""
    Afp8, Asc = mxquant_A(A_bf16)
    return G.mx_matmul(Afp8, W_fp8, Asc, W_sc, M, N, K, fmt="fp8")     # bf16 [M,N]


def mm_ab(A_bf16, B_bf16, M, N, K):
    """C = A @ B with both operands bf16 activations (device-quantized). K = contraction."""
    Afp8, Asc = mxquant_A(A_bf16)
    Bfp8, Bsc = mxquant_B(B_bf16)
    return G.mx_matmul(Afp8, Bfp8, Asc, Bsc, M, N, K, fmt="fp8")       # bf16 [M,N]


# ============================ SIMT epilogues (bf16 in / bf16 out) ============================
def bias_add(T, bias):
    """tile[r,c] += bias[c], fp32 add, bf16 store. Matches bias_add_g."""
    return to_bf16(from_bf16(T) + from_bf16(bias)[None, :])


def residual_add(a, b):
    """out = a + b, elementwise, fp32 add, bf16 store. Matches residual_add_g."""
    return to_bf16(from_bf16(a) + from_bf16(b))


def hb(x):
    """The result of one native bf16 instruction: round float32 value(s) to bf16 and back."""
    return from_bf16(to_bf16(np.asarray(x, dtype=np.float32)))


def _c(v):
    """A float literal as the device materializes it: tf_bf16(v) -> bf16 value."""
    return hb(np.float32(v))


def gelu(T):
    """Op-for-op mirror of impl.hpp:h_gelu -- 0.5*x*(1+erf(x/sqrt2)), erf by Abramowitz-Stegun
    7.1.26 with |x|/sqrt2 clamped at 6, every op rounded to bf16."""
    x = from_bf16(T)
    one = _c(1.0)
    z = np.minimum(hb(np.abs(x) * _c(0.70710678)), _c(6.0))
    t = hb(one / hb(one + hb(_c(0.3275911) * z)))
    poly = _c(1.061405429)
    for a in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
        poly = hb(hb(poly * t) + _c(a))
    poly = hb(poly * t)
    ez = hb(np.exp(-hb(z * z)))
    erf = np.copysign(np.abs(hb(one - hb(poly * ez))), x)
    return to_bf16(hb(_c(0.5) * x) * hb(one + erf))


def _rsqrt(v):
    """Op-for-op mirror of impl.hpp:h_rsqrt: exponent seed 2^-(floor(e/2)+1), 6 Newton steps."""
    e = ((to_bf16(v).astype(np.int32) >> 7) & 0xFF) - 127
    k = (e >> 1) + 1
    y = from_bf16((((127 - k) & 0xFF) << 7).astype(np.uint16))
    hv = hb(_c(0.5) * v)
    c15 = _c(1.5)
    for _ in range(6):
        y = hb(y * hb(c15 - hb(hv * hb(y * y))))
    return y


def layernorm(T, gamma, beta, eps):
    """Op-for-op mirror of layernorm_g: linear bf16 sum chains in column order, biased variance
    clamped at 0, Newton rsqrt, affine."""
    x = from_bf16(T)
    R, N = x.shape
    inv_n = _c(1.0 / N)
    sx = np.zeros(R, np.float32)
    sxx = np.zeros(R, np.float32)
    for c in range(N):
        v = x[:, c]
        sx = hb(sx + v)
        sxx = hb(sxx + hb(v * v))
    mean = hb(sx * inv_n)
    var = np.maximum(hb(hb(sxx * inv_n) - hb(mean * mean)), np.float32(0))
    inv = _rsqrt(hb(var + _c(eps)))
    d = hb(x - mean[:, None])
    d = hb(d * inv[:, None])
    d = hb(d * from_bf16(gamma)[None, :])
    return to_bf16(d + from_bf16(beta)[None, :])


def softmax(T, scale):
    """Op-for-op mirror of softmax_g: m = max(bf16(x*scale)); e = fexp(bf16(x*scale) - m);
    linear bf16 sum chain; out = e * bf16(1/sum)."""
    x = from_bf16(T)
    xs = hb(x * _c(scale))
    m = xs.max(axis=1, keepdims=True)
    e = hb(np.exp(hb(xs - m)))
    acc = np.zeros(x.shape[0], np.float32)
    for c in range(x.shape[1]):
        acc = hb(acc + e[:, c])
    inv = hb(_c(1.0) / acc)
    return to_bf16(e * inv[:, None])


def transpose(T):
    """dst[c][r] = src[r][c]. Matches transpose_gmem."""
    return np.ascontiguousarray(T.T)


# ============================ end-to-end encoder layer (B) ============================
def encoder_layer(op):
    """Run one transformer_mx encoder layer through the golden, mirroring kernel.cpp phase by
    phase. Returns a dict of per-phase expected tensors as bf16 codes, keyed to the MARK()s:
      q,k,v   (MARK1, after Q/K/V + bias)
      ctx     (MARK2, after the per-head softmax(QK^T)@V loop, scattered back)
      res     (MARK3, after Wo + bias + LN1 + residual)  == FFN input / residual branch
      ff      (MARK4, after FF1 + bias + GELU)
      out     (MARK5, after FF2 + bias + LN2 + residual) == layer output
    Also returns s0 = softmax(Q0 K0^T) for head 0 (a spot-check tensor)."""
    H, E, NH, S, HD = op["H"], op["E"], op["NH"], op["S"], op["HD"]
    eps, scale = op["LN_EPS"], op["ATTN_SCALE"]
    X = op["X_in"]                                        # bf16 [S][H]
    W = op["weights"]
    b = op["biases"]

    def w(name):
        Wf, Sc, K, N = W[name]
        return Wf, Sc, K, N

    # ---- Q / K / V = X @ {Wq,Wk,Wv} (+bias) ----
    Wqf, Wqs, _, _ = w("Wq"); q = bias_add(mm_aw(X, Wqf, Wqs, S, H, H), b["bq"])
    Wkf, Wks, _, _ = w("Wk"); k = bias_add(mm_aw(X, Wkf, Wks, S, H, H), b["bk"])
    Wvf, Wvs, _, _ = w("Wv"); v = bias_add(mm_aw(X, Wvf, Wvs, S, H, H), b["bv"])

    # ---- per head: S = softmax(Q_h @ K_h^T); O_h = S @ V_h -> ctx[:, head] ----
    ctx = np.empty((S, H), np.uint16)
    s0 = None
    for h in range(NH):
        c0 = h * HD
        qh = np.ascontiguousarray(q[:, c0:c0 + HD])      # [S][HD]
        kh = np.ascontiguousarray(k[:, c0:c0 + HD])
        vh = np.ascontiguousarray(v[:, c0:c0 + HD])
        kt = transpose(kh)                               # [HD][S]
        sc_mat = mm_ab(qh, kt, S, S, HD)                 # [S][S]
        sc_mat = softmax(sc_mat, scale)
        if h == 0:
            s0 = sc_mat.copy()
        oh = mm_ab(sc_mat, vh, S, HD, S)                 # [S][HD]
        ctx[:, c0:c0 + HD] = oh

    # ---- Wo (+bias); LN1; residual = X + LN1(Wo_out) ----
    Wof, Wos, _, _ = w("Wo")
    tmp = bias_add(mm_aw(ctx, Wof, Wos, S, H, H), b["bo"])
    tmp = layernorm(tmp, b["ln1_g"], b["ln1_b"], eps)
    res = residual_add(X, tmp)                            # X + out

    # ---- FF1 (+bias) -> GELU ----
    Ff1, Ff1s, _, _ = w("FF1")
    ff = bias_add(mm_aw(res, Ff1, Ff1s, S, E, H), b["ff1_b"])
    ff = gelu(ff)

    # ---- FF2 (+bias); LN2; out = LN2(FF2_out) + res ----
    Ff2, Ff2s, _, _ = w("FF2")
    tmp2 = bias_add(mm_aw(ff, Ff2, Ff2s, S, H, E), b["ff2_b"])
    tmp2 = layernorm(tmp2, b["ln2_g"], b["ln2_b"], eps)
    out = residual_add(tmp2, res)

    return {"q": q, "k": k, "v": v, "ctx": ctx, "res": res, "ff": ff, "out": out, "s0": s0}


def matmul_bf16(A_codes, B_codes):
    """Op-for-op mirror of impl.hpp:matmul_g -- C[r][c] = linear bf16 chain over k in k order of
    bf16(A[r][k] * B[k][c]). Inputs and output are bf16 codes."""
    Af, Bf = from_bf16(A_codes), from_bf16(B_codes)
    out = np.zeros((Af.shape[0], Bf.shape[1]), np.float32)
    for kk in range(Af.shape[1]):
        out = hb(out + hb(Af[:, kk:kk + 1] * Bf[kk:kk + 1, :]))
    return to_bf16(out)


def encoder_layer_simt(op):
    """Golden for the all-SIMT kernel path (-DTF_ALL_SIMT): the same op graph as encoder_layer, but
    every GEMM is a bf16 matmul (matmul_bf16) on dequantized weights (gen_data.dequant_bf16) instead
    of an MX fp8 GEMM. Returns the same keys as encoder_layer."""
    from gen_data import dequant_bf16
    H, E, NH, S, HD = op["H"], op["E"], op["NH"], op["S"], op["HD"]
    eps, scale = op["LN_EPS"], op["ATTN_SCALE"]
    X, W, b = op["X_in"], op["weights"], op["biases"]
    Wb = {n: dequant_bf16(W[n][0], W[n][1]) for n in ("Wq", "Wk", "Wv", "Wo", "FF1", "FF2")}

    q = bias_add(matmul_bf16(X, Wb["Wq"]), b["bq"])
    k = bias_add(matmul_bf16(X, Wb["Wk"]), b["bk"])
    v = bias_add(matmul_bf16(X, Wb["Wv"]), b["bv"])
    ctx = np.empty((S, H), np.uint16)
    s0 = None
    for h in range(NH):
        c0 = h * HD
        qh = np.ascontiguousarray(q[:, c0:c0 + HD])
        kh = np.ascontiguousarray(k[:, c0:c0 + HD])
        vh = np.ascontiguousarray(v[:, c0:c0 + HD])
        s = softmax(matmul_bf16(qh, transpose(kh)), scale)
        if h == 0:
            s0 = s.copy()
        ctx[:, c0:c0 + HD] = matmul_bf16(s, vh)
    tmp = layernorm(bias_add(matmul_bf16(ctx, Wb["Wo"]), b["bo"]), b["ln1_g"], b["ln1_b"], eps)
    res = residual_add(X, tmp)
    ff = gelu(bias_add(matmul_bf16(res, Wb["FF1"]), b["ff1_b"]))
    tmp2 = layernorm(bias_add(matmul_bf16(ff, Wb["FF2"]), b["ff2_b"]), b["ln2_g"], b["ln2_b"], eps)
    out = residual_add(tmp2, res)
    return {"q": q, "k": k, "v": v, "ctx": ctx, "res": res, "ff": ff, "out": out, "s0": s0}


# ============================ emit golden header (B) ============================
def _emit_bf16_2d(f, name, arr):
    R, C = arr.shape
    f.write(f"static const uint16_t {name}[{R}][{C}] = {{\n"
            + ",\n".join("  { " + ", ".join(f"0x{v:04x}" for v in row) + " }" for row in arr)
            + "\n};\n\n")


def emit_golden(out_path, op, phases):
    with open(out_path, "w") as f:
        f.write("// @generated by golden_tf.py -- do not edit\n")
        f.write("// Per-phase expected tensors (bf16 codes) for transformer_mx, keyed to MARK()s.\n")
        f.write("// Compare device DRAM (g_q/g_k/g_v/g_ctx/g_res/g_ff/g_out) against these with a\n")
        f.write("// tolerance (fp8/bf16/approx noise): see golden_tf.py --check.\n")
        f.write("#ifndef TF_GOLDEN_H\n#define TF_GOLDEN_H\n#include <stdint.h>\n\n")
        f.write(f"#define TFG_SEQ {op['S']}\n#define TFG_HIDDEN {op['H']}\n")
        f.write(f"#define TFG_EXP {op['E']}\n\n")
        _emit_bf16_2d(f, "G_q", phases["q"])
        _emit_bf16_2d(f, "G_k", phases["k"])
        _emit_bf16_2d(f, "G_v", phases["v"])
        _emit_bf16_2d(f, "G_s0", phases["s0"])     # head 0 softmax(Q_0 @ K_0^T), checked at TF_UPTO=3
        _emit_bf16_2d(f, "G_ctx", phases["ctx"])
        _emit_bf16_2d(f, "G_res", phases["res"])
        _emit_bf16_2d(f, "G_ff", phases["ff"])
        _emit_bf16_2d(f, "G_out", phases["out"])
        f.write("#endif // TF_GOLDEN_H\n")
    print(f"wrote {out_path}  (seq={op['S']} hidden={op['H']} exp={op['E']})")


# ============================ check device dumps (B) ============================
_PHASE_FILES = {"g_q": "q", "g_k": "k", "g_v": "v", "g_ctx": "ctx",
                "g_res": "res", "g_ff": "ff", "g_out": "out"}


def check_dumps(dump_dir, op, phases):
    """Compare raw device DRAM dumps (little-endian uint16 bf16, row-major) against the golden.
    Expects files named <buf>.bin in dump_dir, e.g. g_q.bin. Reports per-phase max/mean abs err."""
    d = pathlib.Path(dump_dir)
    print(f"{'phase':10} {'shape':>12} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10}  status")
    all_ok = True
    for fname, key in _PHASE_FILES.items():
        path = d / f"{fname}.bin"
        exp = phases[key]
        if not path.exists():
            print(f"{fname:10} {'-':>12} {'':>10} {'':>10} {'':>10}  MISSING")
            continue
        got = np.frombuffer(path.read_bytes(), dtype="<u2")[: exp.size].reshape(exp.shape)
        ge, gg = from_bf16(exp), from_bf16(got)
        aerr = np.abs(gg - ge)
        rel = aerr / (np.abs(ge) + 1e-6)
        ok = float(aerr.max()) <= _phase_tol(key, ge)
        all_ok &= ok
        print(f"{fname:10} {str(exp.shape):>12} {aerr.max():>10.4f} {aerr.mean():>10.4f} "
              f"{rel.max():>10.4f}  {'OK' if ok else 'FAIL'}")
    print("ALL PHASES OK" if all_ok else "SOME PHASES FAILED")
    return all_ok


def _phase_tol(key, ref):
    """Absolute-error budget per phase. Early phases (one GEMM) are tight; later phases accumulate
    fp8/bf16/approx error through the chain, so the budget scales with tensor magnitude."""
    scale = float(np.abs(ref).max()) + 1e-3
    depth = {"q": 0.06, "k": 0.06, "v": 0.06, "ctx": 0.12,
             "res": 0.15, "ff": 0.20, "out": 0.25}[key]
    return depth * scale


# ============================ C: per-primitive unit tests ============================
def _relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b) / (np.abs(b) + 1e-6)


def _rand_bf16(rng, shape, scale=0.5):
    return to_bf16((rng.standard_normal(shape) * scale).astype(np.float32))


def test_quant_scale():
    """Block scale must bring each block's max |element| into [1,2): dequant max in that band."""
    rng = np.random.default_rng(1)
    X = _rand_bf16(rng, (64, 128), scale=3.0)
    fp8, sc = mxquant_A(X)
    deq = dequant_A(fp8, sc)
    for g in range(X.shape[1] // GROUP):
        blk_in = np.abs(from_bf16(X[:, g * GROUP:(g + 1) * GROUP]))
        blk_dq = np.abs(deq[:, g * GROUP:(g + 1) * GROUP])
        # dequant block-max approximates input block-max within one e4m3 step (12.5%)
        assert _relerr(blk_dq.max(1), blk_in.max(1)).max() < 0.15, "block scale off"
    print("  [ok] test_quant_scale")


def test_quant_roundtrip():
    """dequant(quant(X)) reconstructs X within e4m3 relative precision for near-max elements."""
    rng = np.random.default_rng(2)
    X = _rand_bf16(rng, (32, 64), scale=1.0)
    deq = dequant_A(*mxquant_A(X))
    xf = from_bf16(X)
    big = np.abs(xf) > 0.25 * np.abs(xf).max()           # ignore tiny elems (huge rel err by design)
    assert _relerr(deq[big], xf[big]).max() < 0.2, "e4m3 round-trip too lossy"
    print("  [ok] test_quant_roundtrip")


def test_gemm_vs_fp32():
    """mm_aw (device fp8 path) agrees with fp32 (dequantA @ dequantW) within fp8 noise."""
    rng = np.random.default_rng(3)
    M, K, N = 32, 128, 64
    A = _rand_bf16(rng, (M, K), 0.5)
    Wf = G.rand_fp8(rng, K * N).reshape(K, N).astype(np.uint8)
    Ws = rng.integers(0x7B, 0x83, (K // GROUP, N), dtype=np.uint8)
    C = from_bf16(mm_aw(A, Wf, Ws, M, N, K))
    Adq = dequant_A(*mxquant_A(A))
    Wdq = e4m3_to_f32(Wf) * (2.0 ** np.repeat(Ws.astype(np.int32) - 127, GROUP, axis=0))
    ref = Adq @ Wdq
    assert np.median(_relerr(C, ref)) < 0.05, "gemm median rel err too high"
    print(f"  [ok] test_gemm_vs_fp32 (median rel {np.median(_relerr(C, ref)):.4f})")


def test_bias_residual_transpose():
    rng = np.random.default_rng(4)
    T = _rand_bf16(rng, (16, 64)); bs = _rand_bf16(rng, (64,))
    exp = to_bf16(from_bf16(T) + from_bf16(bs)[None, :])
    assert np.array_equal(bias_add(T, bs), exp), "bias_add mismatch"
    a, b = _rand_bf16(rng, (16, 64)), _rand_bf16(rng, (16, 64))
    assert np.array_equal(residual_add(a, b), to_bf16(from_bf16(a) + from_bf16(b))), "residual mismatch"
    assert np.array_equal(from_bf16(transpose(T)), from_bf16(T).T), "transpose mismatch"
    print("  [ok] test_bias_residual_transpose")


def test_gelu():
    rng = np.random.default_rng(5)
    T = _rand_bf16(rng, (8, 64), 2.0)
    x = from_bf16(T).astype(np.float64)
    ref = 0.5 * x * (1.0 + np.vectorize(math.erf)(x * INV_SQRT2))     # exact erf oracle
    got = from_bf16(gelu(T))
    assert np.abs(got - ref).max() < 0.02, "gelu off vs exact erf"
    print(f"  [ok] test_gelu (max abs {np.abs(got - ref).max():.4f})")


def test_layernorm():
    rng = np.random.default_rng(6)
    T = _rand_bf16(rng, (8, 128), 1.5)
    gamma = _rand_bf16(rng, (128,), 0.3); beta = _rand_bf16(rng, (128,), 0.1)
    ref = G.layernorm(from_bf16(T), from_bf16(gamma), from_bf16(beta), 1e-5)   # biased-var oracle
    got = from_bf16(layernorm(T, gamma, beta, 1e-5))
    assert np.abs(got - ref).max() < 0.03, "layernorm off vs oracle"
    print(f"  [ok] test_layernorm (max abs {np.abs(got - ref).max():.4f})")


def test_softmax():
    rng = np.random.default_rng(7)
    T = _rand_bf16(rng, (8, 64), 3.0)
    scale = 1.0 / math.sqrt(64)
    x = from_bf16(T).astype(np.float64) * scale
    e = np.exp(x - x.max(1, keepdims=True))
    ref = e / e.sum(1, keepdims=True)                    # exact softmax oracle
    got = from_bf16(softmax(T, scale))
    assert np.abs(got - ref).max() < 0.01, "softmax off vs oracle"
    assert np.abs(got.sum(1) - 1.0).max() < 0.03, "softmax rows not ~normalized"
    print(f"  [ok] test_softmax (max abs {np.abs(got - ref).max():.4f})")


def run_tests():
    print("C: per-primitive unit tests (mx_matmul backend + SIMT epilogues)")
    test_quant_scale()
    test_quant_roundtrip()
    test_gemm_vs_fp32()
    test_bias_residual_transpose()
    test_gelu()
    test_layernorm()
    test_softmax()
    print("ALL PRIMITIVE TESTS PASSED")


# ============================ CLI ============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="C: run per-primitive unit tests")
    ap.add_argument("--emit", metavar="PATH", help="B: emit per-phase golden header")
    ap.add_argument("--check", metavar="DUMP_DIR", help="B: compare device dumps vs golden")
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--expansion", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--tile_n", type=int, default=128)
    ap.add_argument("--ln_eps", type=float, default=1e-5)
    ap.add_argument("--graph", choices=("mx", "simt"), default="mx",
                    help="mx: GEMMs as MX fp8 (default kernel); simt: GEMMs as bf16 matmul (-DTF_ALL_SIMT)")
    a = ap.parse_args()

    if a.test:
        run_tests()
    if a.emit or a.check:
        op = build_operands(a.hidden, a.expansion, a.heads, a.seq, a.tile_n, a.ln_eps)
        phases = encoder_layer_simt(op) if a.graph == "simt" else encoder_layer(op)
        print(f"golden graph: {a.graph}")
        if a.emit:
            emit_golden(a.emit, op, phases)
        if a.check:
            ok = check_dumps(a.check, op, phases)
            sys.exit(0 if ok else 1)
    if not (a.test or a.emit or a.check):
        ap.print_help()


if __name__ == "__main__":
    main()
