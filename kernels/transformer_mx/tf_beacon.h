// Shared by kernel.cpp (device) and host.cpp (rv64 host): fixed GMEM addresses and the stage
// names of the progress beacon. One table, so the two sides cannot drift out of sync.
#ifndef TF_BEACON_H
#define TF_BEACON_H
#include <stdint.h>

// [0..5] mcycle at each phase boundary (tid 0).
#define TF_MARK_GMEM  0x40f00000u
#define TF_NUM_PHASES 6u
// [0] stage id + 1 (0 = nothing yet)  [1] mcycle at stage entry  [2] head index
// [3] seq  [4] hidden  [5] TF_DONE_MAGIC once the output below is complete
#define TF_WHERE_GMEM 0x40f00100u
#define TF_DONE_MAGIC 0x7F5AB1E5u
// Layer output [seq][hidden], bf16 codes, row-major. Fixed so the host can read and check it.
#define TF_OUT_GMEM   0x40f10000u

#define TF_PHASE_NAMES_INIT {                   \
    "attn start (Q/K/V proj)",                  \
    "post-QKV (per-head attention)",            \
    "post-attention (Wo + LN1 + residual)",     \
    "pre-FFN (FF1 + GELU)",                     \
    "post-FF1 (FF2 + bias)",                    \
    "layer done (LN2 + residual)",              \
}

#define TF_STAGES(X)                                                    \
    X(SETUP,        "gemmini flush + config_ex")                        \
    X(QUANT_X,      "mxquant_A(X)")                                     \
    X(GEMM_Q,       "GEMM Wq")         X(BIAS_Q,  "bias q")             \
    X(GEMM_K,       "GEMM Wk")         X(BIAS_K,  "bias k")             \
    X(GEMM_V,       "GEMM Wv")         X(BIAS_V,  "bias v")             \
    X(HD_GATHER,    "head: gather q/k/v")                               \
    X(HD_TRANSPOSE, "head: transpose k")                                \
    X(HD_QUANT_QK,  "head: mxquant q, k^T")                             \
    X(HD_GEMM_QK,   "head: GEMM q@k^T")                                 \
    X(HD_SOFTMAX,   "head: softmax")                                    \
    X(HD_QUANT_SV,  "head: mxquant s, v")                               \
    X(HD_GEMM_SV,   "head: GEMM s@v")                                   \
    X(HD_SCATTER,   "head: scatter to ctx")                             \
    X(QUANT_CTX,    "mxquant_A(ctx)")                                   \
    X(GEMM_WO,      "GEMM Wo")         X(BIAS_WO, "bias o")             \
    X(LN1,          "layernorm1")      X(RESID1,  "residual1")          \
    X(QUANT_RES,    "mxquant_A(res)")                                   \
    X(GEMM_FF1,     "GEMM FF1")        X(BIAS_FF1, "bias ff1")          \
    X(GELU,         "GELU")                                             \
    X(QUANT_FF,     "mxquant_A(ff)")                                    \
    X(GEMM_FF2,     "GEMM FF2")        X(BIAS_FF2, "bias ff2")          \
    X(LN2,          "layernorm2")      X(RESID2,  "residual2")          \
    X(DONE,         "DONE")

#define TF_STAGE_ENUM(n, s) TFS_##n,
enum { TF_STAGES(TF_STAGE_ENUM) TFS_COUNT };
#undef TF_STAGE_ENUM

#endif // TF_BEACON_H
