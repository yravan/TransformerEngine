/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.cuh
 *  \brief Helper function for GEMM-swizzled scales
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_

#include <cstddef>
#include <cstdint>

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace swizzle {

constexpr size_t GEMM_SWIZZLED_SCALE_TILE_DIM_X = 4;
constexpr size_t GEMM_SWIZZLED_SCALE_TILE_DIM_Y = 128;
// Distance between the slots of rows i and i + 1 of one tile while both lie in the same 32-row
// group (see gemm_swizzled_scale_idx below).
constexpr int32_t kSwizzledRowStride = 16;

/*! \brief Convert compact scale indices into GEMM swizzled scale index
 *
 *  MXFP8 GEMM expects scaling factors to be in a "swizzled" order
 *  (https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout).
 *  This function converts indices from "compact" order (i.e. matching
 *  the FP8 data) to swizzled order.
 *
 */
__device__ __forceinline__ size_t gemm_swizzled_scale_idx(size_t i, size_t j, size_t num_tiles_X) {
  constexpr size_t TILE_DIM_X = GEMM_SWIZZLED_SCALE_TILE_DIM_X;
  constexpr size_t TILE_DIM_Y = GEMM_SWIZZLED_SCALE_TILE_DIM_Y;
  constexpr size_t TILE_SIZE = TILE_DIM_X * TILE_DIM_Y;
  const size_t tile_idx_X = j / TILE_DIM_X;
  const size_t tile_idx_Y = i / TILE_DIM_Y;
  const size_t idx_in_tile_X = j % TILE_DIM_X;
  const size_t idx_in_tile_Y = i % TILE_DIM_Y;
  size_t idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE;
  idx += (idx_in_tile_Y % 32) * 16 + (idx_in_tile_Y / 32) * 4 + idx_in_tile_X;
  return idx;
}

__device__ __forceinline__ void zero_gemm_swizzled_scale_padding(uint8_t *const scales,
                                                                 const size_t dim_i,
                                                                 const size_t dim_j,
                                                                 const size_t worker,
                                                                 const size_t num_workers) {
  constexpr size_t TILE_DIM_X = GEMM_SWIZZLED_SCALE_TILE_DIM_X;
  constexpr size_t TILE_DIM_Y = GEMM_SWIZZLED_SCALE_TILE_DIM_Y;
  const size_t pad_i = (dim_i + TILE_DIM_Y - 1) / TILE_DIM_Y * TILE_DIM_Y;
  const size_t pad_j = (dim_j + TILE_DIM_X - 1) / TILE_DIM_X * TILE_DIM_X;
  const size_t tail_cols = pad_j - dim_j;
  const size_t tail_row_elems = (pad_i - dim_i) * pad_j;
  const size_t total = tail_row_elems + dim_i * tail_cols;
  for (size_t p = worker; p < total; p += num_workers) {
    const bool in_tail_rows = p < tail_row_elems;
    const size_t q = in_tail_rows ? p : p - tail_row_elems;
    const size_t i = in_tail_rows ? dim_i + q / pad_j : q / tail_cols;
    const size_t j = in_tail_rows ? q % pad_j : dim_j + q % tail_cols;
    scales[gemm_swizzled_scale_idx(i, j, pad_j / TILE_DIM_X)] = 0;
  }
}

}  // namespace swizzle
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_
