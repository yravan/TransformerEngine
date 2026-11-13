/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_bidim_nvfp4.cuh
 *  \brief Register-resident NVFP4 cast: rowwise 1D blocks plus the RHT'd colwise (transposed)
 *         copy, in one pass over the BF16 input, optionally fused with an activation, its
 *         gradient and the bias gradient.
 */

#ifndef TRANSFORMER_ENGINE_NVFP4_SPECIALIZED_CAST_BIDIM_NVFP4_CUH_
#define TRANSFORMER_ENGINE_NVFP4_SPECIALIZED_CAST_BIDIM_NVFP4_CUH_

#include <cuda_runtime.h>

#include "../../../common.h"
#include "../../../util/activation_x2.cuh"
#include "../../../util/curanddx.hpp"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../util/ptx_arch_spec.cuh"
#include "../../../util/system.h"
#include "../../../utils.cuh"
#include "../core_nvfp4.cuh"
#include "../swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace bidim {

#if FP4_TYPE_SUPPORTED

namespace ptx = transformer_engine::ptx;

// NVFP4 scales 16 consecutive elements together.  The colwise copy is the transpose of the
// input, quantized along what were the input's columns, after a 16x16 randomized Hadamard
// transform of every 16 consecutive rows.  Both the rowwise and the colwise block therefore
// close inside 16 rows, so a lane that owns 16 consecutive rows of a few columns holds every
// element both of its colwise blocks and (with one or two partner lanes) of its rowwise blocks
// need: the tile is read from global memory once, into registers, and never staged.
//
// The global (per-tensor) encode scales are taken from amax values the caller provides.  The
// kernel also emits the amaxes of what it quantized (pre-RHT for the rowwise copy, post-RHT for
// the colwise one) so the caller can carry them to the next call.

constexpr int32_t kBlock = 16;
constexpr int32_t kRowsPerLane = kBlock;
constexpr int32_t kWarpsPerCta = 4;
constexpr int32_t kThreadsPerCta = kWarpsPerCta * THREADS_PER_WARP;
constexpr int32_t kRowsPerCta = kWarpsPerCta * kRowsPerLane;
constexpr int32_t kElemsPerWord = 2;
constexpr float kHadamardScale = 0.25f;

__device__ __forceinline__ float bf16_pair_amax(uint32_t pair) {
  ptx::bf16x2 lo, hi, folded;
  reinterpret_cast<uint32_t &>(lo) = pair;
  reinterpret_cast<uint32_t &>(hi) = __byte_perm(pair, pair, 0x1032);
  ptx::abs_max_2x(folded, lo, hi);
  const uint32_t bits = (reinterpret_cast<const uint32_t &>(folded) & 0x7FFFu) << 16;
  return __uint_as_float(bits);
}

/*! \brief 1 / x: IEEE division, or the approximate reciprocal (1 ulp) under fast math. */
__device__ __forceinline__ float reciprocal(const float x, const bool fast_math) {
  if (fast_math) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
  }
  return 1.0f / x;
}

__device__ __forceinline__ float2 word_to_float2(uint32_t w) {
  // A BF16 is the top half of an FP32: two ALU ops instead of two converts.
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u));
}

/*! \brief Scale two BF16 words (4 values) by one FP32 factor and pack them as 4 E2M1 nibbles. */
template <bool SR>
__device__ __forceinline__ uint16_t scale_cvt_fp4x4(const uint32_t w01, const uint32_t w23,
                                                    const float2 scale2, const uint32_t rbits) {
  const float2 v01 = __fmul2_rn(word_to_float2(w01), scale2);
  const float2 v23 = __fmul2_rn(word_to_float2(w23), scale2);
  uint16_t out;
  if constexpr (SR) {
    asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%4, %3, %2, %1}, %5;\n"
                 : "=h"(out)
                 : "f"(v01.x), "f"(v01.y), "f"(v23.x), "f"(v23.y), "r"(rbits));
  } else {
    asm volatile(
        "{\n"
        ".reg.b8 f0, f1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 f0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 f1, %4, %3;\n"
        "mov.b16 %0, {f0, f1};\n"
        "}"
        : "=h"(out)
        : "f"(v01.x), "f"(v01.y), "f"(v23.x), "f"(v23.y));
  }
  return out;
}

/*! \brief Four scaled FP32 values to packed E2M1, round-to-nearest or stochastic. */
template <bool SR>
__device__ __forceinline__ uint16_t cvt_fp4x4(const float4 v, const uint32_t rbits) {
  uint16_t out;
  if constexpr (SR) {
    // Operands are listed from the highest nibble down, as in mul_cvt_bf16_to_fp4_4x.
    asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%4, %3, %2, %1}, %5;\n"
                 : "=h"(out)
                 : "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbits));
  } else {
    asm volatile(
        "{\n"
        ".reg.b8 f0, f1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 f0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 f1, %4, %3;\n"
        "mov.b16 %0, {f0, f1};\n"
        "}"
        : "=h"(out)
        : "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
  }
  return out;
}

struct Rng {
  transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS> state;
  uint4 buf;
  int idx;
  __device__ __forceinline__ void init(const size_t *rng_state, const size_t seq_base) {
    const size_t sequence = threadIdx.x + blockIdx.x * kThreadsPerCta +
                            blockIdx.y * gridDim.x * kThreadsPerCta + seq_base;
    state.init(rng_state != nullptr ? rng_state[0] : 0, sequence,
               rng_state != nullptr ? rng_state[1] : 0);
    buf = state.generate4();
    idx = 0;
  }
  __device__ __forceinline__ uint32_t next() {
    if (idx == 4) {
      idx = 0;
      buf = state.generate4();
    }
    return reinterpret_cast<const uint32_t *>(&buf)[idx++];
  }
};

/*! \brief One CTA: 64 rows x (32 * COLS_PER_LANE) columns.
 *
 * \tparam COLS_PER_LANE  8 (128-bit loads; column count a multiple of 256) or 2 (32-bit loads;
 *                        column count a multiple of 64).
 */
template <int32_t COLS_PER_LANE, int32_t MIN_BLOCKS_PER_SM, bool SWIZZLED_SCALES, bool SR,
          bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
__global__ __launch_bounds__(kThreadsPerCta, MIN_BLOCKS_PER_SM) void cast_bidim_nvfp4_kernel(
    const uint8_t *__restrict__ input, const uint8_t *__restrict__ act_input,
    uint8_t *__restrict__ out_rowwise, uint8_t *__restrict__ scales_rowwise,
    uint8_t *__restrict__ out_colwise, uint8_t *__restrict__ scales_colwise,
    const float *__restrict__ amax_rowwise_in, const float *__restrict__ amax_colwise_in,
    float *__restrict__ amax_rowwise_out, float *__restrict__ amax_colwise_out,
    float *__restrict__ dbias_workspace, const size_t *__restrict__ rng_state,
    const uint32_t sign_mask, const int32_t rows, const int32_t cols, const int32_t col_base,
    const int32_t scale_stride_rowwise, const int32_t scale_stride_colwise, const bool fast_math) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int32_t kWords = COLS_PER_LANE / kElemsPerWord;
  constexpr int32_t kColsPerCta = THREADS_PER_WARP * COLS_PER_LANE;
  constexpr int32_t kLanesPerRowBlock = kBlock / COLS_PER_LANE;
  constexpr int32_t kColBytesPerCta = kRowsPerCta / 2;
  // Column stride of 9 words: coprime with the 32 banks, so a warp's 32 lanes writing 32
  // different columns at the same row offset hit 32 different banks.
  constexpr int32_t kColWordsPadded = kColBytesPerCta / 4 + 1;

  __shared__ uint32_t s_col[kColsPerCta][kColWordsPadded];
  __shared__ float s_dbias[IS_DBIAS ? kWarpsPerCta : 1][IS_DBIAS ? kColsPerCta : 1];
  __shared__ float s_amax[2][kWarpsPerCta];

  const int32_t tid = threadIdx.x;
  const int32_t warp = tid / THREADS_PER_WARP;
  const int32_t lane = tid % THREADS_PER_WARP;
  const int32_t row0 = blockIdx.y * kRowsPerCta + warp * kRowsPerLane;
  // The launch covers columns [col_base, col_base + gridDim.x * kColsPerCta) of a `cols`-wide tensor.
  const int32_t cta_col0 = col_base + blockIdx.x * kColsPerCta;
  const int32_t col0 = cta_col0 + lane * COLS_PER_LANE;
  const int32_t row_block = blockIdx.y * kWarpsPerCta + warp;

  const uint64_t policy = ptx::create_l2_policy_evict_last();
  const size_t row_stride_bytes = static_cast<size_t>(cols) * sizeof(bf16);
  const uint8_t *tile_in = input + (static_cast<size_t>(row0) * cols + col0) * sizeof(bf16);

  uint32_t tile[kRowsPerLane][kWords];
#pragma unroll
  for (int32_t i = 0; i < kRowsPerLane; ++i) {
    const void *src = tile_in + i * row_stride_bytes;
    if constexpr (kWords == 4) {
      ptx::ld_global_nc_b32x4(tile[i], src, policy);
    } else if constexpr (kWords == 2) {
      const uint2 v = __ldg(reinterpret_cast<const uint2 *>(src));
      tile[i][0] = v.x;
      tile[i][1] = v.y;
    } else {
      tile[i][0] = __ldg(reinterpret_cast<const uint32_t *>(src));
    }
  }

  Rng rng;
  if constexpr (SR) {
    rng.init(rng_state, static_cast<size_t>(col_base) * rows);
  }

  if constexpr (IS_ACT || IS_DACT || IS_DBIAS) {
    const uint8_t *act_in = act_input + (static_cast<size_t>(row0) * cols + col0) * sizeof(bf16);
    float2 column_sum[IS_DBIAS ? kWords : 1];
    if constexpr (IS_DBIAS) {
#pragma unroll
      for (int32_t k = 0; k < kWords; ++k) column_sum[k] = make_float2(0.f, 0.f);
    }
#pragma unroll 2
    for (int32_t i = 0; i < kRowsPerLane; ++i) {
      // The activation input is loaded one row at a time: holding the whole tile spills.
      uint32_t act_row[kWords];
      if constexpr (IS_DACT) {
        const void *src = act_in + i * row_stride_bytes;
        if constexpr (kWords == 4) {
          ptx::ld_global_nc_b32x4(act_row, src, policy);
        } else if constexpr (kWords == 2) {
          const uint2 v = __ldg(reinterpret_cast<const uint2 *>(src));
          act_row[0] = v.x;
          act_row[1] = v.y;
        } else {
          act_row[0] = __ldg(reinterpret_cast<const uint32_t *>(src));
        }
      }
#pragma unroll
      for (int32_t k = 0; k < kWords; ++k) {
        float2 value = word_to_float2(tile[i][k]);
        if constexpr (IS_ACT || IS_DACT) {
          float2 act_in_v = make_float2(0.f, 0.f);
          if constexpr (IS_DACT) act_in_v = word_to_float2(act_row[k]);
          value = activate_x2<IS_ACT, IS_DACT, ParamOP, OP>(value, act_in_v);
          const __nv_bfloat162 rounded = __floats2bfloat162_rn(value.x, value.y);
          tile[i][k] = reinterpret_cast<const uint32_t &>(rounded);
        }
        if constexpr (IS_DBIAS) column_sum[k] = __fadd2_rn(column_sum[k], value);
      }
    }
    if constexpr (IS_DBIAS) {
#pragma unroll
      for (int32_t k = 0; k < kWords; ++k) {
        s_dbias[warp][lane * COLS_PER_LANE + 2 * k] = column_sum[k].x;
        s_dbias[warp][lane * COLS_PER_LANE + 2 * k + 1] = column_sum[k].y;
      }
    }
  }

  // The pre-RHT amax of the tile falls out of the rowwise block amaxes below.
  float tile_amax = 0.f;

  const float S_enc_row = core::compute_global_encode_scaling_factor_FP4<fp8e4m3>(*amax_rowwise_in);
  const float S_dec_row = 1.0f / S_enc_row;
  const float S_enc_col = core::compute_global_encode_scaling_factor_FP4<fp8e4m3>(*amax_colwise_in);
  const float S_dec_col = 1.0f / S_enc_col;

  // ---- rowwise ------------------------------------------------------------------------------
  {
    uint8_t *out_row = out_rowwise + (static_cast<size_t>(row0) * cols + col0) / 2;
    const bool owns_scale = (lane % kLanesPerRowBlock) == 0;
    const int32_t scale_col = col0 / kBlock;
#pragma unroll
    for (int32_t i = 0; i < kRowsPerLane; ++i) {
      ptx::bf16x2 a;
      reinterpret_cast<uint32_t &>(a) = tile[i][0];
#pragma unroll
      for (int32_t k = 1; k < kWords; ++k) {
        ptx::abs_max_2x(a, a, reinterpret_cast<const ptx::bf16x2 &>(tile[i][k]));
      }
#pragma unroll
      for (int32_t d = 1; d < kLanesPerRowBlock; d <<= 1) {
        ptx::bf16x2 p;
        reinterpret_cast<uint32_t &>(p) =
            __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(a), d);
        ptx::abs_max_2x(a, a, p);
      }
      const float block_amax = bf16_pair_amax(reinterpret_cast<const uint32_t &>(a));
      tile_amax = fmaxf(tile_amax, block_amax);
      const fp8e4m3 S_dec_b = core::compute_decoding_scaling_factor<fp8e4m3>(block_amax, S_enc_row);
      const float scale_inv = fminf(reciprocal(static_cast<float>(S_dec_b) * S_dec_row, fast_math),
                                    detail::TypeExtrema<float>::max);
      const float2 scale_inv_2{scale_inv, scale_inv};
      if constexpr (kWords == 4) {
        const uint32_t lo = scale_cvt_fp4x4<SR>(tile[i][0], tile[i][1], scale_inv_2, SR ? rng.next() : 0u);
        const uint32_t hi = scale_cvt_fp4x4<SR>(tile[i][2], tile[i][3], scale_inv_2, SR ? rng.next() : 0u);
        *reinterpret_cast<uint32_t *>(out_row + static_cast<size_t>(i) * cols / 2) = lo | (hi << 16);
      } else if constexpr (kWords == 2) {
        *reinterpret_cast<uint16_t *>(out_row + static_cast<size_t>(i) * cols / 2) =
            scale_cvt_fp4x4<SR>(tile[i][0], tile[i][1], scale_inv_2, SR ? rng.next() : 0u);
      } else {
        out_row[static_cast<size_t>(i) * cols / 2] =
            static_cast<uint8_t>(scale_cvt_fp4x4<SR>(tile[i][0], 0u, scale_inv_2, SR ? rng.next() : 0u));
      }
      if (owns_scale) {
        const size_t idx = SWIZZLED_SCALES
                               ? swizzle::gemm_swizzled_scale_idx(row0 + i, scale_col, cols / kBlock)
                               : static_cast<size_t>(row0 + i) * scale_stride_rowwise + scale_col;
        scales_rowwise[idx] = *reinterpret_cast<const uint8_t *>(&S_dec_b);
      }
    }
  }

  // ---- colwise: randomized Hadamard over the lane's 16 rows, then 1D blocks down each column --
  float post_amax = 0.f;
#pragma unroll
  for (int32_t k = 0; k < kWords; ++k) {
    float2 v[kBlock];
#pragma unroll
    for (int32_t r = 0; r < kBlock; ++r) {
      uint32_t w = tile[r][k];
      // The random sign flips the input element r (both columns of the word at once).
      w ^= ((sign_mask >> r) & 1u) ? 0x80008000u : 0u;
      v[r] = word_to_float2(w);
    }
    // Fast Walsh-Hadamard transform in natural (Sylvester) order: y[c] = sum_r (-1)^popc(r&c) x[r].
#pragma unroll
    for (int32_t h = 1; h < kBlock; h <<= 1) {
#pragma unroll
      for (int32_t r = 0; r < kBlock; ++r) {
        if ((r & h) == 0) {
          const float2 a = v[r], b = v[r + h];
          v[r] = __fadd2_rn(a, b);
          v[r + h] = __fadd2_rn(a, make_float2(-b.x, -b.y));
        }
      }
    }
    float2 a = make_float2(0.f, 0.f);
#pragma unroll
    for (int32_t r = 0; r < kBlock; ++r) {
      // The reference RHT is a bf16 mma: its output is rounded to bf16 before the amax and the cast.
      v[r] = __bfloat1622float2(__float22bfloat162_rn(v[r]));
      a = make_float2(fmaxf(a.x, fabsf(v[r].x)), fmaxf(a.y, fabsf(v[r].y)));
    }
    a = __fmul2_rn(a, make_float2(kHadamardScale, kHadamardScale));
    post_amax = fmaxf(post_amax, fmaxf(a.x, a.y));
    const fp8e4m3 S_dec_b0 = core::compute_decoding_scaling_factor<fp8e4m3>(a.x, S_enc_col);
    const fp8e4m3 S_dec_b1 = core::compute_decoding_scaling_factor<fp8e4m3>(a.y, S_enc_col);
    const float si0 = fminf(reciprocal(static_cast<float>(S_dec_b0) * S_dec_col, fast_math),
                            detail::TypeExtrema<float>::max) * kHadamardScale;
    const float si1 = fminf(reciprocal(static_cast<float>(S_dec_b1) * S_dec_col, fast_math),
                            detail::TypeExtrema<float>::max) * kHadamardScale;
    uint16_t q0[4], q1[4];
    const float2 si2 = make_float2(si0, si1);
#pragma unroll
    for (int32_t g = 0; g < 4; ++g) {
      const float2 p0 = __fmul2_rn(v[4 * g], si2), p1 = __fmul2_rn(v[4 * g + 1], si2);
      const float2 p2 = __fmul2_rn(v[4 * g + 2], si2), p3 = __fmul2_rn(v[4 * g + 3], si2);
      q0[g] = cvt_fp4x4<SR>(make_float4(p0.x, p1.x, p2.x, p3.x), SR ? rng.next() : 0u);
      q1[g] = cvt_fp4x4<SR>(make_float4(p0.y, p1.y, p2.y, p3.y), SR ? rng.next() : 0u);
    }
    const int32_t c0 = lane * COLS_PER_LANE + 2 * k;
    s_col[c0][2 * warp] = q0[0] | (q0[1] << 16);
    s_col[c0][2 * warp + 1] = q0[2] | (q0[3] << 16);
    s_col[c0 + 1][2 * warp] = q1[0] | (q1[1] << 16);
    s_col[c0 + 1][2 * warp + 1] = q1[2] | (q1[3] << 16);
    const int32_t gc0 = col0 + 2 * k;
    const size_t sidx0 = SWIZZLED_SCALES
                             ? swizzle::gemm_swizzled_scale_idx(gc0, row_block, rows / kBlock)
                             : static_cast<size_t>(gc0) * scale_stride_colwise + row_block;
    const size_t sidx1 = SWIZZLED_SCALES
                             ? swizzle::gemm_swizzled_scale_idx(gc0 + 1, row_block, rows / kBlock)
                             : static_cast<size_t>(gc0 + 1) * scale_stride_colwise + row_block;
    scales_colwise[sidx0] = *reinterpret_cast<const uint8_t *>(&S_dec_b0);
    scales_colwise[sidx1] = *reinterpret_cast<const uint8_t *>(&S_dec_b1);
  }

  // ---- amax out, dbias, colwise store ------------------------------------------------------
#pragma unroll
  for (int32_t d = 16; d > 0; d >>= 1) {
    tile_amax = fmaxf(tile_amax, __shfl_xor_sync(0xFFFFFFFFu, tile_amax, d));
    post_amax = fmaxf(post_amax, __shfl_xor_sync(0xFFFFFFFFu, post_amax, d));
  }
  if (lane == 0) {
    s_amax[0][warp] = tile_amax;
    s_amax[1][warp] = post_amax;
  }
  __syncthreads();
  if (tid == 0) {
    float r = s_amax[0][0], c = s_amax[1][0];
#pragma unroll
    for (int32_t w = 1; w < kWarpsPerCta; ++w) {
      r = fmaxf(r, s_amax[0][w]);
      c = fmaxf(c, s_amax[1][w]);
    }
    if (amax_rowwise_out != nullptr) atomicMaxFloat(amax_rowwise_out, r);
    if (amax_colwise_out != nullptr) atomicMaxFloat(amax_colwise_out, c);
  }
  if constexpr (IS_DBIAS) {
    for (int32_t c = tid; c < kColsPerCta; c += kThreadsPerCta) {
      float sum = s_dbias[0][c];
#pragma unroll
      for (int32_t w = 1; w < kWarpsPerCta; ++w) sum += s_dbias[w][c];
      dbias_workspace[static_cast<size_t>(blockIdx.y) * cols + cta_col0 + c] = sum;
    }
  }
  // Each column of the CTA is 32 contiguous bytes of the transposed output.
  constexpr int32_t kColsPerThread = (kColsPerCta + kThreadsPerCta - 1) / kThreadsPerCta;
#pragma unroll
  for (int32_t j = 0; j < kColsPerThread; ++j) {
    const int32_t c = tid * kColsPerThread + j;
    if (kColsPerCta % kThreadsPerCta == 0 || c < kColsPerCta) {
      uint8_t *dst = out_colwise + (static_cast<size_t>(cta_col0 + c) * rows +
                                    static_cast<size_t>(blockIdx.y) * kRowsPerCta) / 2;
      uint32_t w[8];
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) w[i] = s_col[c][i];
      ptx::st_global_b32x4(dst, reinterpret_cast<const uint32_t (&)[4]>(w[0]), policy);
      ptx::st_global_b32x4(dst + 16, reinterpret_cast<const uint32_t (&)[4]>(w[4]), policy);
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <int32_t COLS_PER_LANE, int32_t MIN_BLOCKS_PER_SM, bool SWIZZLED_SCALES, bool SR,
          bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
inline void launch_cast_bidim_nvfp4_cpl(const void *input, const void *act_input,
                                        void *out_rowwise, void *scales_rowwise,
                                        void *out_colwise, void *scales_colwise,
                                        const float *amax_rowwise_in,
                                        const float *amax_colwise_in, float *amax_rowwise_out,
                                        float *amax_colwise_out, float *dbias_workspace,
                                        const size_t *rng_state, uint32_t sign_mask, int rows,
                                        int cols, int col_base, int chunk_cols,
                                        int scale_stride_rowwise, int scale_stride_colwise,
                                        bool fast_math, cudaStream_t stream) {
  constexpr int32_t kColsPerCta = THREADS_PER_WARP * COLS_PER_LANE;
  NVTE_CHECK(rows % kRowsPerCta == 0, "NVFP4 bidim cast needs rows (", rows, ") % ", kRowsPerCta,
             " == 0.");
  NVTE_CHECK(chunk_cols % kColsPerCta == 0, "NVFP4 bidim cast chunk (", chunk_cols, ") % ",
             kColsPerCta, " == 0.");
  const dim3 grid(chunk_cols / kColsPerCta, rows / kRowsPerCta);
  cast_bidim_nvfp4_kernel<COLS_PER_LANE, MIN_BLOCKS_PER_SM, SWIZZLED_SCALES, SR, IS_DBIAS, IS_DACT,
                          IS_ACT, ParamOP, OP><<<grid, kThreadsPerCta, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(input), reinterpret_cast<const uint8_t *>(act_input),
      reinterpret_cast<uint8_t *>(out_rowwise), reinterpret_cast<uint8_t *>(scales_rowwise),
      reinterpret_cast<uint8_t *>(out_colwise), reinterpret_cast<uint8_t *>(scales_colwise),
      amax_rowwise_in, amax_colwise_in, amax_rowwise_out, amax_colwise_out, dbias_workspace,
      rng_state, sign_mask, rows, cols, col_base, scale_stride_rowwise, scale_stride_colwise,
      fast_math);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

constexpr int32_t kWideMinBlocksPerSm = 4;
constexpr int32_t kMidMinBlocksPerSm = 6;
constexpr int32_t kNarrowMinBlocksPerSm = 8;

template <bool SWIZZLED_SCALES, bool SR, bool IS_DBIAS = false, bool IS_DACT = false,
          bool IS_ACT = false, typename ParamOP = Empty,
          float (*OP)(float, const ParamOP &) = nullptr>
inline void launch_cast_bidim_nvfp4(const void *input, const void *act_input, void *out_rowwise,
                                    void *scales_rowwise, void *out_colwise, void *scales_colwise,
                                    const float *amax_rowwise_in, const float *amax_colwise_in,
                                    float *amax_rowwise_out, float *amax_colwise_out,
                                    float *dbias_workspace, const size_t *rng_state,
                                    uint32_t sign_mask, int rows, int cols,
                                    int scale_stride_rowwise, int scale_stride_colwise,
                                    bool fast_math, cudaStream_t stream) {
  NVTE_CHECK(rows % kRowsPerCta == 0, "NVFP4 bidim cast needs rows (", rows, ") % ", kRowsPerCta,
             " == 0.");
  NVTE_CHECK(cols % 64 == 0, "NVFP4 bidim cast needs cols (", cols, ") % 64 == 0.");
#define NVTE_NVFP4_BIDIM_ARGS(base, chunk)                                                     \
  input, act_input, out_rowwise, scales_rowwise, out_colwise, scales_colwise, amax_rowwise_in,  \
      amax_colwise_in, amax_rowwise_out, amax_colwise_out, dbias_workspace, rng_state,          \
      sign_mask, rows, cols, base, chunk, scale_stride_rowwise, scale_stride_colwise, fast_math, \
      stream
  // The widest tile over as many columns as it covers, then one launch per narrower tail, so a
  // 1856-wide tensor runs 1792 columns at the 256-column rate and only 64 at the 64-column one.
  // NVTE_NVFP4_BIDIM_MAX_CPL (8 / 4 / 2) caps the tile width: the best tile depends on the shape.
  static const int max_cpl = [] {
    const int v = transformer_engine::getenv<int>("NVTE_NVFP4_BIDIM_MAX_CPL", 8);
    return v == 4 ? 4 : (v == 2 ? 2 : 8);
  }();
  int base = 0;
  const int wide = max_cpl >= 8 ? cols / 256 * 256 : 0;
  if (wide > 0) {
    launch_cast_bidim_nvfp4_cpl<8, kWideMinBlocksPerSm, SWIZZLED_SCALES, SR, IS_DBIAS, IS_DACT,
                                IS_ACT, ParamOP, OP>(NVTE_NVFP4_BIDIM_ARGS(base, wide));
    base += wide;
  }
  const int mid = max_cpl >= 4 ? (cols - base) / 128 * 128 : 0;
  if (mid > 0) {
    launch_cast_bidim_nvfp4_cpl<4, kMidMinBlocksPerSm, SWIZZLED_SCALES, SR, IS_DBIAS, IS_DACT,
                                IS_ACT, ParamOP, OP>(NVTE_NVFP4_BIDIM_ARGS(base, mid));
    base += mid;
  }
  if (cols - base > 0) {
    launch_cast_bidim_nvfp4_cpl<2, kNarrowMinBlocksPerSm, SWIZZLED_SCALES, SR, IS_DBIAS, IS_DACT,
                                IS_ACT, ParamOP, OP>(NVTE_NVFP4_BIDIM_ARGS(base, cols - base));
    base = cols;
  }
  NVTE_CHECK(base == cols, "NVFP4 bidim cast covered ", base, " of ", cols, " columns.");
#undef NVTE_NVFP4_BIDIM_ARGS
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace bidim
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_NVFP4_SPECIALIZED_CAST_BIDIM_NVFP4_CUH_
