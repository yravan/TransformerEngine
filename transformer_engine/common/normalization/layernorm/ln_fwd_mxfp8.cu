/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ln_fwd_mxfp8.cu
 *  \brief LayerNorm forward fused with MXFP8 quantization, bit-exact with the two-kernel path.
 *
 *  Reproduces the single-warp-per-row configurations of ln_fwd_tuned_kernel (BF16 input and
 *  weights, 768 and 1024 columns): the same statistics reduction (Stats<float, 1, WARPS_M, 1>:
 *  sequential sums, xor-shuffle allreduce, two-pass variance), the same FP32 arithmetic and the
 *  same rounding of the normalized value to BF16. The BF16 values are then quantized the way the
 *  row-wise + column-wise MXFP8 cast (quantize_mxfp8_kernel_cast_only) quantizes a BF16 tensor, so
 *  the output is byte-identical to nvte_layernorm_fwd into a BF16 tensor followed by
 *  nvte_quantize, in either scale layout, while the input is read once and the BF16 intermediate
 *  never written.
 *
 *  Persistent CTAs walk tiles of 32 rows (the height of one column-wise scaling block) x COLS.
 *  A tile arrives in shared memory by one bulk copy, issued while the previous tile is being
 *  processed. Each warp normalizes its two rows together, quantizes them row-wise from registers
 *  and overwrites the tile with the BF16 values in place; the CTA then quantizes the tile
 *  column-wise from shared memory. The grid is one CTA per SM the caller leaves available.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

#include "../../cast/mxfp8/swizzle.cuh"
#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../common.h"

namespace transformer_engine {
namespace normalization {
namespace {

constexpr int kScaleBlock = 32;  // MXFP8 scaling block size.
constexpr int kTileRows = 32;    // One column-wise scaling block.
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * THREADS_PER_WARP;
constexpr int kRowsPerWarp = kTileRows / kWarps;
constexpr int kBuffers = 2;
constexpr int kBytesPerLdg = 16;
constexpr int kEltsPerLdg = kBytesPerLdg / sizeof(bf16);
constexpr int kColsPerLdg = THREADS_PER_WARP * kEltsPerLdg;

static_assert(kRowsPerWarp == 2);

// The column counts whose ln_fwd_tuned_kernel runs one warp per row with BF16 input and weights,
// and whose two tiles fit in shared memory.
constexpr bool is_supported_cols(const size_t cols) {
  return cols == 768 || cols == 1024;
}

template <int COLS>
constexpr uint32_t tile_bytes() {
  return kTileRows * COLS * sizeof(bf16);
}

template <int COLS>
constexpr size_t smem_bytes() {
  return kBuffers * tile_bytes<COLS>() + kTileRows * (COLS / kScaleBlock) * sizeof(e8m0_t);
}

struct QuantizeArgs {
  void *rowwise_data;
  e8m0_t *rowwise_scale_inv;
  void *colwise_data;
  e8m0_t *colwise_scale_inv;
  // Compact layouts: row strides of the scaling-factor tensors. Swizzled layouts: number of
  // 4-wide scaling-factor tiles along the blocked dimension.
  size_t rowwise_scale_stride;
  size_t colwise_scale_stride;
};

// The cast's scaling factor of a block with this absolute maximum.
template <typename OType>
__device__ __forceinline__ e8m0_t amax_to_e8m0(const float amax) {
  return ptx::float_to_e8m0(__fmaf_ieee_rn(amax, Quantized_Limits<OType>::max_norm_rcp, 0.0f));
}

__device__ __forceinline__ bf16 shfl_xor(const bf16 value, const int mask) {
  const uint16_t bits = __shfl_xor_sync(0xffffffff, reinterpret_cast<const uint16_t &>(value), mask);
  return reinterpret_cast<const bf16 &>(bits);
}

// Warp w normalizes rows 2w and 2w+1 of each tile, lane l columns 256*it + 8*l + jt of each, as in
// ln_fwd_tuned_kernel with CTAS_PER_ROW = 1 and WARPS_N = 1. The two rows' statistics are
// computed together: each row's chain of operations is the tuned kernel's, interleaved with the
// other row's.
template <int COLS, typename OType, bool kSwizzled>
__global__ void __launch_bounds__(kThreads, 1)
    ln_mxfp8_kernel(const bf16 *const x, const bf16 *const gamma, const bf16 *const beta,
                    const float epsilon, float *const mu_out, float *const rsigma_out,
                    const QuantizeArgs args, const int num_tiles,
                    const bool zero_centered_gamma) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  using compute_t = float;
  using Ivec = Vec<bf16, kEltsPerLdg>;
  using OTypex4 = ptx::FPx4<OType>;
  constexpr int LDGS = COLS / kColsPerLdg;
  constexpr int kScaleCols = COLS / kScaleBlock;
  constexpr int R = kRowsPerWarp;

  extern __shared__ __align__(128) char smem_[];
  __shared__ uint64_t mbar[kBuffers];
  bf16 *const tiles = reinterpret_cast<bf16 *>(smem_);
  e8m0_t *const rowwise_scales = reinterpret_cast<e8m0_t *>(tiles + kBuffers * kTileRows * COLS);

  const int tidx = threadIdx.x;
  const int warp = tidx / THREADS_PER_WARP;
  const int lane = tidx % THREADS_PER_WARP;

  const auto fetch = [&](const int tile, const int buffer) {
    ptx::mbarrier_arrive_expect_tx(&mbar[buffer], tile_bytes<COLS>());
    ptx::cp_async_bulk_tensor_1d_global_to_shared(
        reinterpret_cast<uint64_t *>(tiles + buffer * kTileRows * COLS),
        reinterpret_cast<const uint64_t *>(x + static_cast<size_t>(tile) * kTileRows * COLS),
        tile_bytes<COLS>(), &mbar[buffer]);
  };
  if (tidx == 0) {
#pragma unroll
    for (int b = 0; b < kBuffers; ++b) {
      ptx::mbarrier_init(&mbar[b], 1);
    }
    ptx::fence_mbarrier_init_release_cluster();
    fetch(blockIdx.x, 0);
  }

  Ivec gamma_v[LDGS];
  Ivec beta_v[LDGS];
  {
    int idx = lane;
#pragma unroll
    for (int it = 0; it < LDGS; ++it) {
      gamma_v[it].load_from(gamma, idx);
      beta_v[it].load_from(beta, idx);
      idx += THREADS_PER_WARP;
    }
  }

  constexpr compute_t rn = 1.f / compute_t(COLS);
  __syncthreads();

  for (int tile = blockIdx.x, k = 0; tile < num_tiles; tile += gridDim.x, ++k) {
    const int buffer = k % kBuffers;
    // The other buffer's tile was fully consumed before the barrier that ended the previous
    // iteration, so the next tile can land in it now.
    if (tidx == 0 && tile + static_cast<int>(gridDim.x) < num_tiles) {
      fetch(tile + gridDim.x, (k + 1) % kBuffers);
    }
    bf16 *const tile_smem = tiles + buffer * kTileRows * COLS;
    const int row_base = tile * kTileRows;
    ptx::mbarrier_wait_parity(&mbar[buffer], (k / kBuffers) % 2);

    bf16 *tile_row[R];
    Ivec xv[R][LDGS];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      tile_row[r] = tile_smem + (warp * R + r) * COLS;
      int idx = lane;
#pragma unroll
      for (int it = 0; it < LDGS; ++it) {
        xv[r][it].load_from(tile_row[r], idx);
        idx += THREADS_PER_WARP;
      }
    }

    // Stats<compute_t, 1, WARPS_M, 1>::compute: the lane's sequential sum, the warp's xor
    // allreduce, then the same for the squared deviations.
    compute_t m[R];
    compute_t m2[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      m[r] = 0.f;
    }
#pragma unroll
    for (int it = 0; it < LDGS; ++it) {
#pragma unroll
      for (int jt = 0; jt < kEltsPerLdg; ++jt) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          m[r] += compute_t(xv[r][it].data.elt[jt]);
        }
      }
    }
#pragma unroll
    for (int step = 1; step < THREADS_PER_WARP; step *= 2) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        m[r] = m[r] + warp_shuffle_xor(m[r], step);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      m[r] *= rn;
      m2[r] = 0.f;
    }
#pragma unroll
    for (int it = 0; it < LDGS; ++it) {
#pragma unroll
      for (int jt = 0; jt < kEltsPerLdg; ++jt) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const compute_t diff = (compute_t(xv[r][it].data.elt[jt]) - m[r]);
          m2[r] += diff * diff;
        }
      }
    }
#pragma unroll
    for (int step = 1; step < THREADS_PER_WARP; step *= 2) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        m2[r] = m2[r] + warp_shuffle_xor(m2[r], step);
      }
    }
    compute_t rs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      rs[r] = rsqrtf(rn * m2[r] + epsilon);
      if (lane == 0) {
        mu_out[row_base + warp * R + r] = m[r];
        rsigma_out[row_base + warp * R + r] = rs[r];
      }
    }

#pragma unroll
    for (int it = 0; it < LDGS; ++it) {
      const int col = it * kColsPerLdg + lane * kEltsPerLdg;
      Ivec z[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int jt = 0; jt < kEltsPerLdg; ++jt) {
          const compute_t y_ij = rs[r] * (compute_t(xv[r][it].data.elt[jt]) - m[r]);
          compute_t g_ij = gamma_v[it].data.elt[jt];
          if (zero_centered_gamma) {
            g_ij += 1;
          }
          const compute_t b_ij = beta_v[it].data.elt[jt];
          const compute_t temp_output = g_ij * y_ij + b_ij;
          z[r].data.elt[jt] = bf16(temp_output);
        }
        z[r].store_to(tile_row[r], col / kEltsPerLdg);
      }

      // A row-wise scaling block is this lane's 8 values and those of the 3 lanes sharing
      // lane / 4. The reduction is the cast's: max.xorsign.abs over the BF16 values.
      bf16 amax[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const ptx::bf16x2 *const z2 = reinterpret_cast<const ptx::bf16x2 *>(&z[r]);
        ptx::bf16x2 amax2{bf16(0.f), bf16(0.f)};
#pragma unroll
        for (int p = 0; p < kEltsPerLdg / 2; ++p) {
          ptx::abs_max_2x(amax2, amax2, z2[p]);
        }
        amax[r] = ptx::get_amax(amax2.x, amax2.y);
      }
#pragma unroll
      for (int mask = 1; mask <= 2; mask *= 2) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          amax[r] = ptx::get_amax(amax[r], shfl_xor(amax[r], mask));
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int row_in_tile = warp * R + r;
        const e8m0_t biased_exponent = amax_to_e8m0<OType>(static_cast<float>(amax[r]));
        const float scale_inv = ptx::exp2f_rcp<float>(biased_exponent);
        const ptx::floatx2 scale_inv_2{scale_inv, scale_inv};
        const ptx::bf16x4 *const z4 = reinterpret_cast<const ptx::bf16x4 *>(&z[r]);
        OTypex4 quantized[2];
        ptx::mul_cvt_4x(quantized[0], ptx::up_cast(z4[0]), scale_inv_2);
        ptx::mul_cvt_4x(quantized[1], ptx::up_cast(z4[1]), scale_inv_2);
        OType *const rowwise_row = reinterpret_cast<OType *>(args.rowwise_data) +
                                   static_cast<size_t>(row_base + row_in_tile) * COLS;
        *reinterpret_cast<uint2 *>(rowwise_row + col) =
            *reinterpret_cast<const uint2 *>(quantized);
        if (lane % 4 == 0) {
          rowwise_scales[row_in_tile * kScaleCols + col / kScaleBlock] = biased_exponent;
        }
      }
    }
    __syncthreads();

    // Column-wise: 2 columns per thread over the tile's 32 rows, two rows per conversion. The
    // amax starts from the first row so that, as with the cast's redux.sync.max.abs, a block of
    // NaNs scales as NaN while a NaN among finite values is ignored.
    for (int g = tidx; g < 0; g += kThreads) {
      const int col = g * 2;
      ptx::bf16x2 v[kTileRows];
      float amax[2];
#pragma unroll
      for (int r = 0; r < kTileRows; ++r) {
        v[r] = *reinterpret_cast<const ptx::bf16x2 *>(tile_smem + r * COLS + col);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float value = fabsf(static_cast<float>(j == 0 ? v[r].x : v[r].y));
          amax[j] = r == 0 ? value : fmaxf(amax[j], value);
        }
      }
      e8m0_t biased_exponents[2];
      float scale_inv[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        biased_exponents[j] = amax_to_e8m0<OType>(amax[j]);
        scale_inv[j] = ptx::exp2f_rcp<float>(biased_exponents[j]);
      }
      const ptx::floatx4 scale_inv_4{scale_inv[0], scale_inv[1], scale_inv[0], scale_inv[1]};
      OType *const colwise = reinterpret_cast<OType *>(args.colwise_data) +
                             static_cast<size_t>(row_base) * COLS + col;
#pragma unroll
      for (int r = 0; r < kTileRows; r += 2) {
        const ptx::bf16x4 pair{v[r].x, v[r].y, v[r + 1].x, v[r + 1].y};
        OTypex4 quantized;
        ptx::mul_cvt_4x(quantized, ptx::up_cast(pair), scale_inv_4);
        const uint32_t bits = reinterpret_cast<const uint32_t &>(quantized);
        *reinterpret_cast<uint16_t *>(colwise + r * COLS) = static_cast<uint16_t>(bits);
        *reinterpret_cast<uint16_t *>(colwise + (r + 1) * COLS) = static_cast<uint16_t>(bits >> 16);
      }
      const size_t scale_row = row_base / kScaleBlock;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const size_t idx =
            kSwizzled ? gemm_swizzled_scale_idx(col + j, scale_row, args.colwise_scale_stride)
                      : scale_row * args.colwise_scale_stride + col + j;
        args.colwise_scale_inv[idx] = biased_exponents[j];
      }
    }

    // Row-wise scales: the 4 of a 128-column stretch are contiguous in both layouts.
    for (int i = tidx; i < kTileRows * (kScaleCols / 4); i += kThreads) {
      const int r = i / (kScaleCols / 4);
      const int scale_col = 4 * (i % (kScaleCols / 4));
      const uint32_t scales =
          *reinterpret_cast<const uint32_t *>(rowwise_scales + r * kScaleCols + scale_col);
      const size_t idx =
          kSwizzled ? gemm_swizzled_scale_idx(row_base + r, scale_col, args.rowwise_scale_stride)
                    : static_cast<size_t>(row_base + r) * args.rowwise_scale_stride + scale_col;
      *reinterpret_cast<uint32_t *>(args.rowwise_scale_inv + idx) = scales;
    }
    __syncthreads();
  }
#else
  NVTE_DEVICE_THREAD0_ERROR("LayerNorm forward with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <int COLS, typename OType>
void launch_layernorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                                const float epsilon, Tensor *z, Tensor *mu, Tensor *rsigma,
                                const int num_sms, const bool zero_centered_gamma,
                                cudaStream_t stream) {
  const int rows = static_cast<int>(x.flat_2d_dims()[0]);
  const bool swizzled = z->with_gemm_swizzled_scales;

  QuantizeArgs args{};
  args.rowwise_data = z->data.dptr;
  args.rowwise_scale_inv = reinterpret_cast<e8m0_t *>(z->scale_inv.dptr);
  const size_t scale_cols = z->scale_inv.shape.back();
  args.rowwise_scale_stride = swizzled ? scale_cols / 4 : scale_cols;
  args.colwise_data = z->columnwise_data.dptr;
  args.colwise_scale_inv = reinterpret_cast<e8m0_t *>(z->columnwise_scale_inv.dptr);
  args.colwise_scale_stride =
      swizzled ? z->columnwise_scale_inv.shape.front() / 4 : z->columnwise_scale_inv.shape.back();

  const int num_tiles = rows / kTileRows;
  const int sms = num_sms > 0 && num_sms < cuda::sm_count() ? num_sms : cuda::sm_count();
  const int grid = std::min(num_tiles, sms);
  constexpr size_t smem = smem_bytes<COLS>();
  TRANSFORMER_ENGINE_SWITCH_CONDITION(swizzled, kSwizzled, {
    const auto kernel = ln_mxfp8_kernel<COLS, OType, kSwizzled>;
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<grid, kThreads, smem, stream>>>(
        reinterpret_cast<const bf16 *>(x.data.dptr), reinterpret_cast<const bf16 *>(gamma.data.dptr),
        reinterpret_cast<const bf16 *>(beta.data.dptr), epsilon,
        reinterpret_cast<float *>(mu->data.dptr), reinterpret_cast<float *>(rsigma->data.dptr),
        args, num_tiles, zero_centered_gamma);
  });  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

bool is_supported_by_te_layernorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma,
                                            const Tensor &beta, const Tensor &z, const Tensor &mu,
                                            const Tensor &rsigma) {
  if (!is_mxfp8_scaling(z.scaling_mode) || !is_supported_by_CC_100()) {
    return false;
  }
  if (!z.has_data() || !z.has_columnwise_data() || z.scale_inv.shape.size() != 2 ||
      z.columnwise_scale_inv.shape.size() != 2) {
    return false;
  }
  const auto [rows, cols] = x.flat_2d_dims();
  // Full 128x128 tiles, so the scaling factors carry no padding in either layout.
  if (rows == 0 || rows % 128 != 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) || !is_supported_cols(cols)) {
    return false;
  }
  if (x.data.dtype != DType::kBFloat16 || gamma.data.dtype != DType::kBFloat16 ||
      beta.data.dtype != DType::kBFloat16) {
    return false;
  }
  const auto is_supported_output = [](DType t) {
    return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2;
  };
  if (!is_supported_output(z.data.dtype) || z.columnwise_data.dtype != z.data.dtype) {
    return false;
  }
  // Vector accesses: ln_fwd_tuned_kernel's 16-byte loads and stores (the bulk copy of x as
  // well), 8 and 2 bytes of each quantized output, 4 scaling factors at a time.
  return is_aligned_to(x.data.dptr, 16) && is_aligned_to(gamma.data.dptr, 16) &&
         is_aligned_to(beta.data.dptr, 16) && is_aligned_to(mu.data.dptr, 16) &&
         is_aligned_to(rsigma.data.dptr, 16) && is_aligned_to(z.data.dptr, 8) &&
         is_aligned_to(z.columnwise_data.dptr, 2) && is_aligned_to(z.scale_inv.dptr, 4) &&
         is_aligned_to(z.columnwise_scale_inv.dptr, 4);
}

void layernorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                         const float epsilon, Tensor *z, Tensor *mu, Tensor *rsigma,
                         const int multiprocessorCount, const bool zero_centered_gamma,
                         cudaStream_t stream) {
  const size_t cols = x.flat_2d_dims()[1];
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(z->data.dtype, OType, {
    const auto launch = [&](auto cols_tag) {
      launch_layernorm_fwd_mxfp8<decltype(cols_tag)::value, OType>(
          x, gamma, beta, epsilon, z, mu, rsigma, multiprocessorCount, zero_centered_gamma, stream);
    };
    switch (cols) {
      case 768:
        launch(std::integral_constant<int, 768>{});
        break;
      case 1024:
        launch(std::integral_constant<int, 1024>{});
        break;
      default:
        NVTE_ERROR("LayerNorm forward with MXFP8 output supports 768 or 1024 columns, got ",
                   cols, ".");
    }
  });  // NOLINT(*)
}

}  // namespace normalization
}  // namespace transformer_engine
