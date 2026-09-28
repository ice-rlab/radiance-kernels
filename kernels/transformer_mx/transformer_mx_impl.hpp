#ifndef _TRANSFORMER_MX_IMPL_H
#define _TRANSFORMER_MX_IMPL_H
#include <stdint.h>
#include <mu_intrinsics.h>

// SIMT epilogues for transformer_mx, bf16 in DRAM (uint16 codes, row-major).
//
// ALL ARITHMETIC IS NATIVE bf16, ONE NAMED INSTRUCTION PER OP. flash_attention_mx_stable
// (kernel.cpp, hardware fact 2) found no validated bf16 <-> fp32 path on this core: a softmax
// that accumulated in fp32 put 5,120 of 8,192 cells at +-inf, and the same kernel written in bf16
// was correct. clang reaches for fp32 on its own -- fmaxf() and `+`/`*` on _Float16 compile to a
// promote/op/demote triple, and a*b-c contracts into one fmsub.h -- so every op goes through an
// asm helper below. Nothing here may use float, fmaxf, or a bare arithmetic operator on _Float16.
// fsqrt.h assembles but no kernel in this tree uses it, so LayerNorm uses a Newton rsqrt instead.
//
// THREAD OWNERSHIP IS BY WHOLE 32-BIT WORDS. Sub-word stores to one word from different warps are
// an open suspicion on this SMEM/GMEM path (kernel.cpp hardware fact 3), so every function below
// gives each thread whole rows (bf16: N even) or whole 4-byte groups (scales, fp8). golden_tf.py
// mirrors these functions op for op; keep the two in sync.

static inline _Float16 h_add(_Float16 a, _Float16 b) {
    _Float16 o; asm("fadd.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 h_sub(_Float16 a, _Float16 b) {
    _Float16 o; asm("fsub.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 h_mul(_Float16 a, _Float16 b) {
    _Float16 o; asm("fmul.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 h_div(_Float16 a, _Float16 b) {
    _Float16 o; asm("fdiv.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 h_max(_Float16 a, _Float16 b) {
    _Float16 o; asm("fmax.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 h_min(_Float16 a, _Float16 b) {
    _Float16 o; asm("fmin.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline uint16_t h_bits(_Float16 a) { return __builtin_bit_cast(uint16_t, a); }
static inline _Float16 h_abs(_Float16 a) { return as_bf16((uint16_t)(h_bits(a) & 0x7fffu)); }
static inline _Float16 h_neg(_Float16 a) { return as_bf16((uint16_t)(h_bits(a) ^ 0x8000u)); }

// float literal -> bf16 code at compile time (integer ops only; same rounding as gen_data.py's
// f32_to_bf16). Materializes as an `li`, never as an fp conversion.
constexpr uint16_t tf_bf16(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
#define TFB(x) as_bf16(tf_bf16(x))

// ---------------------------------------------------------------------------------------------
// MX quantization (integer + fmax.h only)
// ---------------------------------------------------------------------------------------------

static inline int bf16_floor_log2(uint16_t b) {  // base-2 exponent of a bf16 (its e8m0 block scale)
    return (int)((b >> 7) & 0xff) - 127;
}

// bf16 * 2^-se -> e4m3 byte, truncating. Branchless, so threads never diverge.
// Overflow saturates to the e4m3 max; zero or underflow -> 0.
static inline uint8_t bf16_to_e4m3_scaled(uint16_t b, int se) {
    const int emax = 8, emin = -6;
    int exp = (int)((b >> 7) & 0xff);
    int m3  = (int)((b >> 4) & 0x7);
    int E   = exp - 127 - se;
    int over = -(int)(E > emax);
    E  = (E  & ~over) | (emax & over);
    m3 = (m3 & ~over) | (6    & over);
    int clampm = -(int)((E == emax) & (m3 > 6));
    m3 = (m3 & ~clampm) | (6 & clampm);               // e4m3 max mantissa at emax is 6 (=448)
    int code = ((b >> 8) & 0x80) | ((E + 7) << 3) | m3;
    int keep = -(int)!((exp == 0) | (E < emin));
    return (uint8_t)(code & keep);
}

// A operand [M][K] bf16 -> fp8 [M][K] + e8m0 scales [K/32][M]. Groups along K. Each thread owns
// four rows, i.e. whole scale words and whole fp8 words.
template <uint32_t M, uint32_t K>
static __attribute__((noinline)) void mxquant_A(
        const uint16_t *in, uint8_t *fp8, uint8_t *scales, uint32_t tid, uint32_t thr) {
    static_assert(M % 4 == 0 && K % 32 == 0);
    constexpr uint32_t GK = K / 32;
    for (uint32_t r0 = tid * 4; r0 < M; r0 += thr * 4) {
        for (uint32_t m = r0; m < r0 + 4; m++) {
            for (uint32_t g = 0; g < GK; g++) {
                _Float16 hmax = as_bf16((uint16_t)0);
                for (uint32_t k = 0; k < 32; k++)
                    hmax = h_max(hmax, h_abs(as_bf16(in[m * K + g * 32 + k])));
                const int se = bf16_floor_log2(h_bits(hmax));
                scales[g * M + m] = (uint8_t)(se + 127);
                for (uint32_t k = 0; k < 32; k++)
                    fp8[m * K + g * 32 + k] = bf16_to_e4m3_scaled(in[m * K + g * 32 + k], se);
            }
        }
    }
}

// B operand [K][N] bf16 -> fp8 [K][N] + e8m0 scales [K/32][N]. Groups along K. Each thread owns
// four columns, i.e. whole words of every row.
template <uint32_t K, uint32_t N>
static __attribute__((noinline)) void mxquant_B(
        const uint16_t *in, uint8_t *fp8, uint8_t *scales, uint32_t tid, uint32_t thr) {
    static_assert(N % 4 == 0 && K % 32 == 0);
    constexpr uint32_t GK = K / 32;
    for (uint32_t n0 = tid * 4; n0 < N; n0 += thr * 4) {
        for (uint32_t n = n0; n < n0 + 4; n++) {
            for (uint32_t g = 0; g < GK; g++) {
                _Float16 hmax = as_bf16((uint16_t)0);
                for (uint32_t r = 0; r < 32; r++)
                    hmax = h_max(hmax, h_abs(as_bf16(in[(g * 32 + r) * N + n])));
                const int se = bf16_floor_log2(h_bits(hmax));
                scales[g * N + n] = (uint8_t)(se + 127);
                for (uint32_t r = 0; r < 32; r++)
                    fp8[(g * 32 + r) * N + n] = bf16_to_e4m3_scaled(in[(g * 32 + r) * N + n], se);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data movement (one thread per destination row)
// ---------------------------------------------------------------------------------------------

// dst[C][R] = src[R][C]
template <uint32_t R, uint32_t C>
static __attribute__((noinline)) void transpose_g(
        const uint16_t *src, uint16_t *dst, uint32_t tid, uint32_t thr) {
    static_assert(R % 2 == 0);
    for (uint32_t c = tid; c < C; c += thr)
        for (uint32_t r = 0; r < R; r++)
            dst[c * R + r] = src[r * C + c];
}

// dst[M][SUB] = src[M][c0 : c0 + SUB]
template <uint32_t M, uint32_t FULL, uint32_t SUB>
static __attribute__((noinline)) void gather_cols(
        const uint16_t *src, uint16_t *dst, uint32_t c0, uint32_t tid, uint32_t thr) {
    for (uint32_t r = tid; r < M; r += thr)
        for (uint32_t c = 0; c < SUB; c++)
            dst[r * SUB + c] = src[r * FULL + c0 + c];
}

// dst[M][c0 : c0 + SUB] = src[M][SUB]
template <uint32_t M, uint32_t FULL, uint32_t SUB>
static __attribute__((noinline)) void scatter_cols(
        const uint16_t *src, uint16_t *dst, uint32_t c0, uint32_t tid, uint32_t thr) {
    for (uint32_t r = tid; r < M; r += thr)
        for (uint32_t c = 0; c < SUB; c++)
            dst[r * FULL + c0 + c] = src[r * SUB + c];
}

// ---------------------------------------------------------------------------------------------
// Epilogues (one thread per row, bf16 only)
// ---------------------------------------------------------------------------------------------

// C[M][N] = A[M][K] @ B[K][N] in bf16 on the SIMT cores (no Gemmini). Each output row is one
// thread's; the dot product accumulates as a linear chain in k order. Used for the attention
// S @ V_h product: on the FPGA the Gemmini S @ V GEMM (whose A operand is ~96% zero after fp8
// quantization) was followed by a Gemmini that never reported not-busy (2026-09-12 C, I5, I7).
template <uint32_t M, uint32_t K, uint32_t N>
static __attribute__((noinline)) void matmul_g(
        const uint16_t *A, const uint16_t *B, uint16_t *C, uint32_t tid, uint32_t thr) {
    static_assert(N % 2 == 0);
    for (uint32_t r = tid; r < M; r += thr)
        for (uint32_t c = 0; c < N; c++) {
            _Float16 acc = as_bf16((uint16_t)0);
            for (uint32_t k = 0; k < K; k++)
                acc = h_add(acc, h_mul(as_bf16(A[r * K + k]), as_bf16(B[k * N + c])));
            C[r * N + c] = h_bits(acc);
        }
}

// tile[r][c] += bias[c]
template <uint32_t M, uint32_t N>
// PAIRED 32-BIT STORES (2026-09-22). Every bf16 epilogue used to write uint16_t, i.e. a SUB-WORD
// store per element. Sub-word stores leave partial-line dirty state in the L0 D-cache, which
// accumulates until the GPU wedges -- the same defect fixed in tf_store_c, which was only half the
// problem. Evidence: at H=768 rungs 1-4 pass but rung 5 hangs, and deleting ONLY layernorm_g (its
// 32x768 = 24,576 sub-word stores) makes rung 5 run again. All N here are multiples of 32 and every
// row start is 4-byte aligned, so two bf16 results pack into one aligned 32-bit store.
#define TF_PACK2(lo, hi) ((uint32_t)(lo) | ((uint32_t)(hi) << 16))


static __attribute__((noinline)) void bias_add_g(
        uint16_t *tile, const uint16_t *bias, uint32_t tid, uint32_t thr) {
    static_assert(N % 2 == 0, "paired stores need an even N");
    for (uint32_t r = tid; r < M; r += thr) {
        uint32_t *const row = reinterpret_cast<uint32_t *>(tile + r * N);
        for (uint32_t c = 0; c < N; c += 2)
            row[c / 2] = TF_PACK2(h_bits(h_add(as_bf16(tile[r * N + c]), as_bf16(bias[c]))),
                                  h_bits(h_add(as_bf16(tile[r * N + c + 1]), as_bf16(bias[c + 1]))));
    }
}

// out = a + b. out may alias a or b.
template <uint32_t M, uint32_t N>
static __attribute__((noinline)) void residual_add_g(
        uint16_t *out, const uint16_t *a, const uint16_t *b, uint32_t tid, uint32_t thr) {
    static_assert(N % 2 == 0, "paired stores need an even N");
    for (uint32_t r = tid; r < M; r += thr) {
        uint32_t *const row = reinterpret_cast<uint32_t *>(out + r * N);
        for (uint32_t c = 0; c < N; c += 2)
            row[c / 2] = TF_PACK2(h_bits(h_add(as_bf16(a[r * N + c]), as_bf16(b[r * N + c]))),
                                  h_bits(h_add(as_bf16(a[r * N + c + 1]), as_bf16(b[r * N + c + 1]))));
    }
}

// Exact (erf) GELU, 0.5*x*(1 + erf(x/sqrt2)), erf by Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7,
// far below bf16 resolution). |x|/sqrt2 is clamped at 6 (erf(6) = 1 - 2e-17) to bound exp's input.
static inline _Float16 h_gelu(_Float16 x) {
    const _Float16 one = TFB(1.0f);
    _Float16 z = h_min(h_mul(h_abs(x), TFB(0.70710678f)), TFB(6.0f));
    const _Float16 t = h_div(one, h_add(one, h_mul(TFB(0.3275911f), z)));
    _Float16 poly = TFB(1.061405429f);
    poly = h_add(h_mul(poly, t), TFB(-1.453152027f));
    poly = h_add(h_mul(poly, t), TFB(1.421413741f));
    poly = h_add(h_mul(poly, t), TFB(-0.284496736f));
    poly = h_add(h_mul(poly, t), TFB(0.254829592f));
    poly = h_mul(poly, t);
    const _Float16 ez = mu_fexp(h_neg(h_mul(z, z)));
    const uint16_t eb = h_bits(h_sub(one, h_mul(poly, ez)));
    const _Float16 erf = as_bf16((uint16_t)((eb & 0x7fffu) | (h_bits(x) & 0x8000u)));
    return h_mul(h_mul(TFB(0.5f), x), h_add(one, erf));
}

template <uint32_t M, uint32_t N>
static __attribute__((noinline)) void gelu_g(uint16_t *tile, uint32_t tid, uint32_t thr) {
    static_assert(N % 2 == 0, "paired stores need an even N");
    for (uint32_t r = tid; r < M; r += thr) {
        uint32_t *const row = reinterpret_cast<uint32_t *>(tile + r * N);
        for (uint32_t c = 0; c < N; c += 2)
            row[c / 2] = TF_PACK2(h_bits(h_gelu(as_bf16(tile[r * N + c]))),
                                  h_bits(h_gelu(as_bf16(tile[r * N + c + 1]))));
    }
}

// 1/sqrt(v), v > 0. Seed 2^-(floor(e/2)+1) puts v*y0^2 in [1/4, 1), inside Newton's convergence
// region from below; 6 iterations reach bf16 resolution from the worst seed (rel err 0.5 -> 6e-5).
static inline _Float16 h_rsqrt(_Float16 v) {
    const int32_t e = (int32_t)((h_bits(v) >> 7) & 0xffu) - 127;
    const int32_t k = (e >> 1) + 1;
    _Float16 y = as_bf16((uint16_t)(((uint32_t)(127 - k) & 0xffu) << 7));
    const _Float16 hv = h_mul(TFB(0.5f), v);
    const _Float16 c15 = TFB(1.5f);
    for (uint32_t i = 0; i < 6; i++) y = h_mul(y, h_sub(c15, h_mul(hv, h_mul(y, y))));
    return y;
}

// Row LayerNorm with affine gamma/beta, biased variance. Sums are a linear chain in column order.
template <uint32_t M, uint32_t N>
static __attribute__((noinline)) void layernorm_g(
        uint16_t *tile, const uint16_t *gamma, const uint16_t *beta, uint16_t eps_code,
        uint32_t tid, uint32_t thr) {
    // N need NOT be a power of two. The assert here used to demand it so that 1/N is exact in
    // bf16, but golden_tf.layernorm rounds it the same way (inv_n = _c(1.0/N) vs TFB(1.0f/N))
    // and both then compute mean = sx * invN, so device and golden agree either way. BERT-base's
    // hidden size is 768, which is not a power of two -- this assert blocked H=768 (2026-09-22).
    static_assert(N % 32 == 0, "layernorm expects a whole number of 32-wide blocks");
    const _Float16 invN = TFB(1.0f / (float)N);
    const _Float16 eps = as_bf16(eps_code);
    for (uint32_t r = tid; r < M; r += thr) {
        _Float16 sx = as_bf16((uint16_t)0), sxx = as_bf16((uint16_t)0);
        for (uint32_t c = 0; c < N; c++) {
            const _Float16 v = as_bf16(tile[r * N + c]);
            sx = h_add(sx, v);
            sxx = h_add(sxx, h_mul(v, v));
        }
        const _Float16 mean = h_mul(sx, invN);
        // bf16 rounding can drive a near-constant row's variance below zero; clamp, then add eps.
        const _Float16 var = h_max(h_sub(h_mul(sxx, invN), h_mul(mean, mean)), as_bf16((uint16_t)0));
        const _Float16 inv = h_rsqrt(h_add(var, eps));
        uint32_t *const row = reinterpret_cast<uint32_t *>(tile + r * N);
        for (uint32_t c = 0; c < N; c += 2) {
            _Float16 d0 = h_mul(h_mul(h_sub(as_bf16(tile[r * N + c]), mean), inv), as_bf16(gamma[c]));
            _Float16 d1 = h_mul(h_mul(h_sub(as_bf16(tile[r * N + c + 1]), mean), inv),
                                as_bf16(gamma[c + 1]));
            row[c / 2] = TF_PACK2(h_bits(h_add(d0, as_bf16(beta[c]))),
                                  h_bits(h_add(d1, as_bf16(beta[c + 1]))));
        }
    }
}

// Row softmax of scale*tile, in place. exp is taken relative to the row max so every exp <= 1.
template <uint32_t M, uint32_t N>
static __attribute__((noinline)) void softmax_g(
        uint16_t *tile, uint16_t scale_code, uint32_t tid, uint32_t thr) {
    const _Float16 scale = as_bf16(scale_code);
    for (uint32_t r = tid; r < M; r += thr) {
        _Float16 mx = as_bf16((uint16_t)0xff7f);     // -max bf16
        for (uint32_t c = 0; c < N; c++)
            mx = h_max(mx, h_mul(as_bf16(tile[r * N + c]), scale));
        _Float16 sum = as_bf16((uint16_t)0);
        for (uint32_t c = 0; c < N; c++) {
            // Bound fexp.h's input, as h_gelu does. x - mx reaches about -460 here once the raw
            // scores carry their real scales (|score| ~ 1300, scale 1/sqrt(HD)), and the hardware
            // exp returns non-finite values that far out: rung 3 came back with 54 inf/nan in
            // s0 while sraw was bit-exact 1024/1024 (2026-09-17 rung3real). exp(-80) = 1.8e-35 is
            // still finite in bf16 and, against a sum >= 1, divides to the same zero the exact
            // softmax gives, so clamping changes no representable result.
            const _Float16 d = h_max(h_sub(h_mul(as_bf16(tile[r * N + c]), scale), mx), TFB(-80.0f));
            const _Float16 e = mu_fexp(d);
            tile[r * N + c] = h_bits(e);
            sum = h_add(sum, e);
        }
        const _Float16 inv = h_div(TFB(1.0f), sum);
        for (uint32_t c = 0; c < N; c++)
            tile[r * N + c] = h_bits(h_mul(as_bf16(tile[r * N + c]), inv));
    }
}

// Single-row forms of the epilogues, so kernel.cpp can drive the row loop itself and stamp
// progress per row while the hang is being localised (2026-09-22). Identical arithmetic and
// identical paired stores -- only the loop nesting moves.
template <uint32_t N>
static __attribute__((noinline)) void bias_add_row(uint16_t *row, const uint16_t *bias) {
    uint32_t *const w = reinterpret_cast<uint32_t *>(row);
    for (uint32_t c = 0; c < N; c += 2)
        w[c / 2] = TF_PACK2(h_bits(h_add(as_bf16(row[c]), as_bf16(bias[c]))),
                            h_bits(h_add(as_bf16(row[c + 1]), as_bf16(bias[c + 1]))));
}

template <uint32_t N>
static __attribute__((noinline)) void residual_add_row(uint16_t *out, const uint16_t *a,
                                                       const uint16_t *b) {
    uint32_t *const w = reinterpret_cast<uint32_t *>(out);
    for (uint32_t c = 0; c < N; c += 2)
        w[c / 2] = TF_PACK2(h_bits(h_add(as_bf16(a[c]), as_bf16(b[c]))),
                            h_bits(h_add(as_bf16(a[c + 1]), as_bf16(b[c + 1]))));
}

template <uint32_t N>
static __attribute__((noinline)) void layernorm_row(uint16_t *row, const uint16_t *gamma,
                                                    const uint16_t *beta, uint16_t eps_code) {
    const _Float16 invN = TFB(1.0f / (float)N);
    const _Float16 eps = as_bf16(eps_code);
    _Float16 sx = as_bf16((uint16_t)0), sxx = as_bf16((uint16_t)0);
    for (uint32_t c = 0; c < N; c++) {
        const _Float16 v = as_bf16(row[c]);
        sx = h_add(sx, v);
        sxx = h_add(sxx, h_mul(v, v));
    }
    const _Float16 mean = h_mul(sx, invN);
    const _Float16 var = h_max(h_sub(h_mul(sxx, invN), h_mul(mean, mean)), as_bf16((uint16_t)0));
    const _Float16 inv = h_rsqrt(h_add(var, eps));
    uint32_t *const w = reinterpret_cast<uint32_t *>(row);
    for (uint32_t c = 0; c < N; c += 2) {
        _Float16 d0 = h_mul(h_mul(h_sub(as_bf16(row[c]), mean), inv), as_bf16(gamma[c]));
        _Float16 d1 = h_mul(h_mul(h_sub(as_bf16(row[c + 1]), mean), inv), as_bf16(gamma[c + 1]));
        w[c / 2] = TF_PACK2(h_bits(h_add(d0, as_bf16(beta[c]))), h_bits(h_add(d1, as_bf16(beta[c + 1]))));
    }
}

#endif
