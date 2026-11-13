/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_bidim_gated.cuh
 *  \brief Register-resident MXFP8 cast fused with a gated activation (SwiGLU) or its gradient.
 */

#ifndef TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_GATED_CUH_
#define TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_GATED_CUH_

#include <cuda_runtime.h>

#include "cast_bidim.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {
namespace bidim_detail {

template <typename OType, int32_t COLS_PER_LANE, int32_t MIN_BLOCKS_PER_SM, bool SWIZZLED_SCALES,
          bool IS_BWD, bool FAST_MATH, float (*ActOP)(float, const Empty &)>
__global__ __launch_bounds__(kThreadsPerCta, MIN_BLOCKS_PER_SM) void quantize_gated_bidim_kernel(
    const uint8_t *__restrict__ gated_input, const uint8_t *__restrict__ grad,
    uint8_t *__restrict__ output_rowwise, uint8_t *__restrict__ scales_rowwise,
    uint8_t *__restrict__ output_colwise, uint8_t *__restrict__ scales_colwise, int32_t rows,
    int32_t cols, int32_t scale_stride_rowwise, int32_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int32_t kPlanes = IS_BWD ? 2 : 1;
  constexpr int32_t kWordsPerLane = COLS_PER_LANE / kElemsPerWord;
  constexpr int32_t kOutWordsPerLane = kWordsPerLane / 2;
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * COLS_PER_LANE;
  constexpr int32_t kLanesPerRowBlock = kBlockElems / COLS_PER_LANE;
  constexpr int32_t kColumnSlots = kColsPerTile / kElemsPerWord;
  constexpr int32_t kSlotsPerLaneHalf = kWordsPerLane / 2;
  constexpr int32_t kHalfSlots = kColumnSlots / 2;

  __shared__ uint32_t s_column_partial[kPlanes][kWarpsPerCta][kColumnSlots];
  __shared__ uint32_t s_column_scale[kPlanes][kColumnSlots];

  const int32_t tid = threadIdx.x;
  const int32_t warp = tid / THREADS_PER_WARP;
  const int32_t lane = tid % THREADS_PER_WARP;

  const int32_t row0 = blockIdx.y * kRowsPerTile + warp * kRowsPerWarp;
  const int32_t col0 = blockIdx.x * kColsPerTile;
  const int32_t out_cols = kPlanes * cols;

  const uint64_t policy = ptx::create_l2_policy_evict_last();

  const size_t in_stride_bytes = static_cast<size_t>(2) * cols * sizeof(bf16);
  const size_t in_offset = static_cast<size_t>(row0) * 2 * cols + col0 + COLS_PER_LANE * lane;
  const uint8_t *act_in = gated_input + in_offset * sizeof(bf16);
  const uint8_t *gate_in = act_in + static_cast<size_t>(cols) * sizeof(bf16);

  uint32_t tile[kPlanes][kRowsPerWarp][kWordsPerLane];

  {
    uint32_t act_tile[kRowsPerWarp][kWordsPerLane];
    uint32_t gate_tile[kRowsPerWarp][kWordsPerLane];
    uint32_t grad_tile[kRowsPerWarp][IS_BWD ? kWordsPerLane : 1];
#pragma unroll
    for (int32_t i = 0; i < kRowsPerWarp; ++i) {
      ptx::ld_global_nc_b32x4(act_tile[i], act_in + i * in_stride_bytes, policy);
      ptx::ld_global_nc_b32x4(gate_tile[i], gate_in + i * in_stride_bytes, policy);
    }
    if constexpr (IS_BWD) {
      const size_t grad_offset = static_cast<size_t>(row0) * cols + col0 + COLS_PER_LANE * lane;
      const uint8_t *grad_in = grad + grad_offset * sizeof(bf16);
#pragma unroll
      for (int32_t i = 0; i < kRowsPerWarp; ++i) {
        ptx::ld_global_nc_b32x4(grad_tile[i], grad_in + i * static_cast<size_t>(cols) * sizeof(bf16),
                                policy);
      }
    }
#pragma unroll
    for (int32_t i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
      for (int32_t k = 0; k < kWordsPerLane; ++k) {
        const float2 a = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 &>(act_tile[i][k]));
        const float2 g =
            __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 &>(gate_tile[i][k]));
        if constexpr (IS_BWD) {
          const float2 d =
              __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 &>(grad_tile[i][k]));
          const float2 sg = sigmoid_x2<FAST_MATH>(a);
          const float2 act = __fmul2_rn(a, sg);
          const float2 dact = __ffma2_rn(act, __ffma2_rn(sg, splat2(-1.f), splat2(1.f)), sg);
          const float2 after_act = __fmul2_rn(__fmul2_rn(dact, d), g);
          const float2 after_gate = __fmul2_rn(act, d);
          const __nv_bfloat162 ra = __floats2bfloat162_rn(after_act.x, after_act.y);
          const __nv_bfloat162 rg = __floats2bfloat162_rn(after_gate.x, after_gate.y);
          tile[0][i][k] = reinterpret_cast<const uint32_t &>(ra);
          tile[1][i][k] = reinterpret_cast<const uint32_t &>(rg);
        } else {
          const float2 act = __fmul2_rn(__fmul2_rn(a, sigmoid_x2<FAST_MATH>(a)), g);
          const __nv_bfloat162 r = __floats2bfloat162_rn(act.x, act.y);
          tile[0][i][k] = reinterpret_cast<const uint32_t &>(r);
        }
      }
    }
  }

  uint8_t *out_row = output_rowwise + static_cast<size_t>(row0) * out_cols + col0 +
                     COLS_PER_LANE * lane;
  uint8_t *out_col = output_colwise + static_cast<size_t>(row0) * out_cols + col0 +
                     COLS_PER_LANE * lane;
  const bool owns_row_scale = (lane % kLanesPerRowBlock) == 0;
  const int32_t row_scale_col = col0 / kBlockElems + (lane / kLanesPerRowBlock);

#pragma unroll
  for (int32_t pl = 0; pl < kPlanes; ++pl) {
    // A warp's rows lie in one 32-row group, so in the swizzled layout the slot of row i is the
    // first row's plus a fixed stride: the index arithmetic is done once per plane.
    const int32_t plane_scale_col = row_scale_col + pl * (cols / kBlockElems);
    uint8_t *const scale_row =
        SWIZZLED_SCALES
            ? scales_rowwise +
                  swizzle::gemm_swizzled_scale_idx(row0, plane_scale_col, DIVUP(out_cols, 128))
            : scales_rowwise + static_cast<size_t>(row0) * scale_stride_rowwise + plane_scale_col;
    const int32_t scale_row_stride =
        SWIZZLED_SCALES ? swizzle::kSwizzledRowStride : scale_stride_rowwise;
#pragma unroll
    for (int32_t i = 0; i < kRowsPerWarp; ++i) {
      ptx::bf16x2 amax;
      reinterpret_cast<uint32_t &>(amax) = tile[pl][i][0];
#pragma unroll
      for (int32_t k = 1; k < kWordsPerLane; ++k) {
        ptx::abs_max_2x(amax, amax, reinterpret_cast<const ptx::bf16x2 &>(tile[pl][i][k]));
      }
#pragma unroll
      for (int32_t d = 1; d < kLanesPerRowBlock; d <<= 1) {
        ptx::bf16x2 partner;
        reinterpret_cast<uint32_t &>(partner) =
            __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(amax), d);
        ptx::abs_max_2x(amax, amax, partner);
      }

      const uint32_t reciprocal =
          mx_scale_reciprocal<OType>(fold_pair_magnitude(reinterpret_cast<const uint32_t &>(amax)));
      const ptx::bf16x2 broadcast = broadcast_pair(reciprocal);
      ptx::bf16x2 scales[kWordsPerLane];
#pragma unroll
      for (int32_t k = 0; k < kWordsPerLane; ++k) {
        scales[k] = broadcast;
      }
      uint32_t out_words[kOutWordsPerLane];
      scale_and_convert<OType>(tile[pl][i], scales, out_words);
      store_words(out_row + static_cast<size_t>(i) * out_cols + pl * cols, out_words, policy);

      if (owns_row_scale) {
        ptx::st_global_b8(scale_row + i * scale_row_stride, mx_scale_byte(reciprocal), policy);
      }
    }
  }

#pragma unroll
  for (int32_t pl = 0; pl < kPlanes; ++pl) {
    uint32_t column_partial[kWordsPerLane];
#pragma unroll
    for (int32_t k = 0; k < kWordsPerLane; ++k) {
      ptx::bf16x2 acc;
      reinterpret_cast<uint32_t &>(acc) = tile[pl][0][k];
#pragma unroll
      for (int32_t i = 1; i < kRowsPerWarp; ++i) {
        ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(tile[pl][i][k]));
      }
      column_partial[k] = reinterpret_cast<const uint32_t &>(acc);
    }
    store_shared_words<kSlotsPerLaneHalf>(&s_column_partial[pl][warp][kSlotsPerLaneHalf * lane],
                                          &column_partial[0]);
    store_shared_words<kSlotsPerLaneHalf>(
        &s_column_partial[pl][warp][kHalfSlots + kSlotsPerLaneHalf * lane],
        &column_partial[kSlotsPerLaneHalf]);
  }
  __syncthreads();

  for (int32_t idx = tid; idx < kPlanes * kColumnSlots; idx += kThreadsPerCta) {
    const int32_t pl = idx / kColumnSlots;
    const int32_t slot = idx % kColumnSlots;
    const uint32_t *column = &s_column_partial[pl][0][slot];
    ptx::bf16x2 acc;
    reinterpret_cast<uint32_t &>(acc) = column[0];
#pragma unroll
    for (int32_t w = 1; w < kWarpsPerCta; ++w) {
      ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(column[w * kColumnSlots]));
    }
    const uint32_t amax_pair = reinterpret_cast<const uint32_t &>(acc);

    uint32_t reciprocal_pair;
    const uint32_t exceptional = ((amax_pair & kMagnitudeMaskPair) + kInfProbePair) & 0x80008000u;
    if (__builtin_expect(exceptional == 0u, 1)) {
      reciprocal_pair = mx_scale_reciprocal_x2<OType>(amax_pair);
    } else {
      const uint32_t lo = mx_scale_reciprocal<OType>(amax_pair & kBf16MagnitudeMask);
      const uint32_t hi = mx_scale_reciprocal<OType>((amax_pair >> 16) & kBf16MagnitudeMask);
      reciprocal_pair = lo | (hi << 16);
    }
    s_column_scale[pl][slot] = reciprocal_pair;
  }
  __syncthreads();

#pragma unroll
  for (int32_t pl = 0; pl < kPlanes; ++pl) {
    uint32_t column_reciprocal[kWordsPerLane];
    load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[0],
                                         &s_column_scale[pl][kSlotsPerLaneHalf * lane]);
    load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[kSlotsPerLaneHalf],
                                         &s_column_scale[pl][kHalfSlots + kSlotsPerLaneHalf * lane]);

    if (warp == pl) {
      const int32_t plane_col0 = col0 + pl * cols + COLS_PER_LANE * lane;
      if constexpr (SWIZZLED_SCALES) {
        // The lane's columns are consecutive within one 32-column group: one stride apart.
        uint8_t *const scale_col =
            scales_colwise +
            swizzle::gemm_swizzled_scale_idx(plane_col0, blockIdx.y, DIVUP(rows, 128));
#pragma unroll
        for (int32_t c = 0; c < COLS_PER_LANE; c += 2) {
          const uint32_t pair_bytes = mx_scale_byte_x2(column_reciprocal[c / 2]);
#pragma unroll
          for (int32_t h = 0; h < 2; ++h) {
            ptx::st_global_b8(scale_col + (c + h) * swizzle::kSwizzledRowStride,
                              (pair_bytes >> (8 * h)) & 0xFFu, policy);
          }
        }
      } else {
        uint32_t packed[kOutWordsPerLane];
#pragma unroll
        for (int32_t j = 0; j < kOutWordsPerLane; ++j) {
          packed[j] = mx_scale_byte_x2(column_reciprocal[2 * j]) |
                      (mx_scale_byte_x2(column_reciprocal[2 * j + 1]) << 16);
        }
        store_words(scales_colwise + static_cast<size_t>(blockIdx.y) * scale_stride_colwise +
                        plane_col0,
                    packed, policy);
      }
    }

    ptx::bf16x2 colwise_scales[kWordsPerLane];
#pragma unroll
    for (int32_t k = 0; k < kWordsPerLane; ++k) {
      reinterpret_cast<uint32_t &>(colwise_scales[k]) = column_reciprocal[k];
    }
#pragma unroll
    for (int32_t i = 0; i < kRowsPerWarp; ++i) {
      uint32_t out_words[kOutWordsPerLane];
      scale_and_convert<OType>(tile[pl][i], colwise_scales, out_words);
      store_words(out_col + static_cast<size_t>(i) * out_cols + pl * cols, out_words, policy);
    }
  }
#endif
}

constexpr int32_t kGatedColsPerLane = 8;
constexpr int32_t kGatedMinBlocksPerSm = 4;
constexpr int32_t kGatedCluster = 1;

/*! \brief Cast a gated activation (or its gradient) to MXFP8, rowwise and colwise, in one pass.
 *
 * \param gated_input  BF16 [rows, 2 * cols]: activation half, then gate half.
 * \param grad         BF16 [rows, cols], read only when IS_BWD.
 * \param output_*     FP8 [rows, cols], or [rows, 2 * cols] when IS_BWD (activation, then gate).
 */
template <typename OType, bool SWIZZLED_SCALES, bool IS_BWD, bool FAST_MATH,
          float (*ActOP)(float, const Empty &)>
inline void launch_cast_bidim_gated(const void *gated_input, const void *grad,
                                    void *output_rowwise, void *scales_rowwise,
                                    void *output_colwise, void *scales_colwise, int rows, int cols,
                                    int scale_stride_rowwise, int scale_stride_colwise,
                                    cudaStream_t stream) {
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * kGatedColsPerLane;
  NVTE_CHECK(rows % kRowsPerTile == 0, "Gated bidimensional MXFP8 requires rows (", rows,
             ") to be a multiple of ", kRowsPerTile, ".");
  NVTE_CHECK(cols % kColsPerTile == 0, "Gated bidimensional MXFP8 requires cols (", cols,
             ") to be a multiple of ", kColsPerTile, ".");

  const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);
  int32_t cluster_x = kGatedCluster;
  while (cluster_x > 1 && static_cast<int32_t>(grid.x) % cluster_x != 0) {
    --cluster_x;
  }

  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_x;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreadsPerCta);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  auto kernel = quantize_gated_bidim_kernel<OType, kGatedColsPerLane, kGatedMinBlocksPerSm,
                                            SWIZZLED_SCALES, IS_BWD, FAST_MATH, ActOP>;
  NVTE_CHECK_CUDA(cudaLaunchKernelEx(
      &config, kernel, reinterpret_cast<const uint8_t *>(gated_input),
      reinterpret_cast<const uint8_t *>(grad), reinterpret_cast<uint8_t *>(output_rowwise),
      reinterpret_cast<uint8_t *>(scales_rowwise), reinterpret_cast<uint8_t *>(output_colwise),
      reinterpret_cast<uint8_t *>(scales_colwise), rows, cols, scale_stride_rowwise,
      scale_stride_colwise));
}

}

constexpr size_t kGatedBidimRows = bidim_detail::kRowsPerTile;
constexpr size_t kGatedBidimCols = THREADS_PER_WARP * bidim_detail::kGatedColsPerLane;

using bidim_detail::launch_cast_bidim_gated;

}
}
}
}
}

#endif
