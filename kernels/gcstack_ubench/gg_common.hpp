#pragma once

// Shared setup for the bench_gg_* benches (Gemmini "G2Stack"/ggstack states).
// Include AFTER "ubench.hpp".
//
// The mxgemm library reads its operands from globals by name (A_in, B_in, the scale
// factors and the LUTs), so a data header must be included before it. The values do not
// matter for these benches -- only which Gemmini units are busy -- so any FP8 128x128x128
// data set works. GG is the same feasible tile config as
// ../gemm_mxgemmini/mxgemm.fp8.m128n128k128.tm128tn128tk128.fullout.cpp.

#include "../gemm_mxgemmini/mxgemm.data.fp8.m128n128k128.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_lib.hpp"

constexpr GemmConfig GG{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

// One raw Gemmini instruction word (same encoding ROCC_INSTRUCTION_RS1_RS2 writes).
constexpr uint32_t gg_inst(uint32_t funct) {
  return 0x7Bu | (0u << 7) | (3u << 12) | (1u << 15) | (2u << 20) | (funct << 25);
}
