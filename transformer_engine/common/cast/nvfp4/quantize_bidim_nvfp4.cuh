/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_bidim_nvfp4.cuh
 *  \brief Host side of the single-pass NVFP4 cast (rowwise + RHT'd colwise) with carried amax.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_BIDIM_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_BIDIM_NVFP4_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../core/grouped_layout.cuh"
#include "specialized/cast_bidim_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

/*! \brief Whether the fused single-pass kernel serves this cast.
 *
 * The caller opts in by setting the amax-out tensor in the quantization config, which also means
 * the output tensor's amax fields already hold the global amaxes to encode with.
 */
inline bool bidim_eligible(const Tensor &input, const Tensor *output,
                           const QuantizationConfig &config) {
  if (config.nvfp4_amax_out == nullptr) return false;
  if (input.dtype() != DType::kBFloat16) return false;
  if (!output->has_data() || !output->has_columnwise_data()) return false;
  if (config.nvfp4_2d_quantization || output->row_scaled_nvfp4) return false;
  if (config.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled) return false;
  if (output->scale_inv.dtype != DType::kFloat8E4M3 ||
      output->columnwise_scale_inv.dtype != DType::kFloat8E4M3) {
    return false;
  }
  if (output->amax.dptr == nullptr || output->columnwise_amax.dptr == nullptr) return false;
  const auto [rows, cols] = input.flat_2d_dims();
  if (rows % bidim::kRowsPerCta != 0 || cols % 64 != 0) return false;
  return transformer_engine::cuda::sm_arch() >= 100;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize_bidim(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
                    Tensor *workspace, const QuantizationConfig &config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  const auto [rows, cols] = input.flat_2d_dims();
  const size_t dbias_rows = rows / bidim::kRowsPerCta;
  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == Shape{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }
  if constexpr (IS_DACT) {
    NVTE_CHECK(act_input != nullptr && act_input->dtype() == DType::kBFloat16,
               "NVFP4 fused dact needs a BF16 activation input.");
    NVTE_CHECK(act_input->data.shape == input.data.shape, "dact input/grad shape mismatch.");
  }
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;

  const Tensor &amax_out = *convertNVTETensorCheck(config.nvfp4_amax_out);
  NVTE_CHECK(amax_out.data.dtype == DType::kFloat32 && amax_out.data.numel() >= 2,
             "NVFP4 amax-out tensor must hold 2 floats.");
  float *const amax_out_ptr = reinterpret_cast<float *>(amax_out.data.dptr);

  const size_t *rng_state = nullptr;
  const size_t rng_seed_v = config.rng_by_value ? config.rng_seed_offset[0] : 0;
  const size_t rng_offset_v = config.rng_by_value ? config.rng_seed_offset[1] : 0;
  if (config.stochastic_rounding && !config.rng_by_value) {
    NVTE_CHECK(config.rng_state != nullptr, "Stochastic rounding needs an RNG state.");
    const Tensor &rng = *convertNVTETensorCheck(config.rng_state);
    NVTE_CHECK(rng.dtype() == DType::kInt64 && rng.data.shape == Shape{2},
               "RNG state must be 2 int64 values.");
    rng_state = reinterpret_cast<const size_t *>(rng.data.dptr);
  }

  const bool swizzled = output->with_gemm_swizzled_scales;
  const int scale_stride_rowwise = static_cast<int>(output->scale_inv.shape[1]);
  const int scale_stride_colwise = static_cast<int>(output->columnwise_scale_inv.shape[1]);
  const uint32_t sign_mask = static_cast<uint32_t>(config.nvfp4_rht_sign_mask) & 0xFFFFu;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      swizzled, SWIZZLED,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          config.stochastic_rounding, SR,
          bidim::launch_cast_bidim_nvfp4<SWIZZLED, SR, IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
              input.data.dptr, IS_DACT ? act_input->data.dptr : nullptr, output->data.dptr,
              output->scale_inv.dptr, output->columnwise_data.dptr,
              output->columnwise_scale_inv.dptr,
              reinterpret_cast<const float *>(output->amax.dptr),
              reinterpret_cast<const float *>(output->columnwise_amax.dptr), amax_out_ptr,
              amax_out_ptr + 1, workspace_ptr, rng_state, rng_seed_v, rng_offset_v, sign_mask,
              static_cast<int>(rows),
              static_cast<int>(cols), scale_stride_rowwise, scale_stride_colwise,
              config.use_fast_math, stream);));
  if constexpr (IS_DBIAS) {
    common::reduce_dbias<bf16>(workspace_ptr, dbias, dbias_rows, cols, stream);
  }
#else
  NVTE_ERROR("FP4 type not supported by this CUDA toolkit.");
#endif
}

namespace carry {

static __global__ void __launch_bounds__(32) carry_amax_kernel(
    float *__restrict__ history, float *__restrict__ fresh, int32_t *__restrict__ step,
    const float margin, const int32_t history_len, float *__restrict__ amax_rowwise,
    float *__restrict__ amax_columnwise) {
  const int32_t slot = *step % history_len;
  if (threadIdx.x == 0) {
    history[slot] = fresh[0];
    history[history_len + slot] = fresh[1];
  }
  __syncwarp();
  float r = 0.f, c = 0.f;
  for (int32_t i = threadIdx.x; i < history_len; i += 32) {
    r = fmaxf(r, history[i]);
    c = fmaxf(c, history[history_len + i]);
  }
#pragma unroll
  for (int32_t d = 16; d > 0; d >>= 1) {
    r = fmaxf(r, __shfl_xor_sync(0xFFFFFFFFu, r, d));
    c = fmaxf(c, __shfl_xor_sync(0xFFFFFFFFu, c, d));
  }
  if (threadIdx.x == 0) {
    *amax_rowwise = r * margin;
    *amax_columnwise = c * margin;
    fresh[0] = 0.f;
    fresh[1] = 0.f;
    *step += 1;
  }
}

}  // namespace carry

inline void carry_amax(const Tensor &history, Tensor &fresh, Tensor &step, float margin,
                       Tensor &amax_rowwise, Tensor &amax_columnwise, cudaStream_t stream) {
  NVTE_CHECK(history.data.dtype == DType::kFloat32 && history.data.shape.size() == 2 &&
                 history.data.shape[0] == 2,
             "NVFP4 amax history must be a [2, N] float tensor.");
  NVTE_CHECK(fresh.data.dtype == DType::kFloat32 && fresh.data.numel() == 2,
             "NVFP4 fresh amax must hold 2 floats.");
  NVTE_CHECK(step.data.dtype == DType::kInt32 && step.data.numel() == 1,
             "NVFP4 amax step must be 1 int32.");
  carry::carry_amax_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<float *>(history.data.dptr), reinterpret_cast<float *>(fresh.data.dptr),
      reinterpret_cast<int32_t *>(step.data.dptr), margin,
      static_cast<int32_t>(history.data.shape[1]),
      reinterpret_cast<float *>(amax_rowwise.data.dptr),
      reinterpret_cast<float *>(amax_columnwise.data.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_BIDIM_NVFP4_CUH_
