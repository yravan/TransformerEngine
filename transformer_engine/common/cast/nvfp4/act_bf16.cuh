/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file act_bf16.cuh
 *  \brief Activation / activation-backward (+ dbias) on BF16 at memory bandwidth, for the NVFP4
 *         recipes whose casts run as a separate RHT kernel: y = act(x), or dy * act'(x) with the
 *         column sums of the result, written in BF16 for the cast that follows.
 */

#ifndef TRANSFORMER_ENGINE_CAST_NVFP4_ACT_BF16_CUH_
#define TRANSFORMER_ENGINE_CAST_NVFP4_ACT_BF16_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../common.h"
#include "../../util/activation_x2.cuh"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "../core/grouped_layout.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace actbf16 {

constexpr int kThreads = 128;
constexpr int kVec = 8;  // bf16 per 16-byte vector
constexpr int kRowsPerCta = 64;

__device__ __forceinline__ float2 word_to_float2(const uint32_t w) {
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u));
}

__device__ __forceinline__ float identity(float x, const Empty &) { return x; }

template <bool IS_ACT, bool IS_DACT, bool IS_DBIAS, bool WRITE_OUT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads)
    act_bf16_kernel(const uint4 *__restrict__ in, const uint4 *__restrict__ act_in,
                    uint4 *__restrict__ out, float *__restrict__ dbias_part, const int rows,
                    const int cols) {
  const int cols8 = cols / kVec;
  const int col8 = blockIdx.x * kThreads + threadIdx.x;
  if (col8 >= cols8) return;
  const int row0 = blockIdx.y * kRowsPerCta;
  const int row_end = min(row0 + kRowsPerCta, rows);
  float2 sum[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) sum[k] = make_float2(0.f, 0.f);
#pragma unroll 4
  for (int row = row0; row < row_end; ++row) {
    const size_t idx = static_cast<size_t>(row) * cols8 + col8;
    uint4 v = __ldg(in + idx);
    uint4 a = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (IS_DACT) a = __ldg(act_in + idx);
    uint32_t *vw = reinterpret_cast<uint32_t *>(&v);
    const uint32_t *aw = reinterpret_cast<const uint32_t *>(&a);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 ai = make_float2(0.f, 0.f);
      if constexpr (IS_DACT) ai = word_to_float2(aw[k]);
      const float2 y = activate_x2<IS_ACT, IS_DACT, ParamOP, OP>(word_to_float2(vw[k]), ai);
      if constexpr (IS_DBIAS) sum[k] = __fadd2_rn(sum[k], y);
      if constexpr (WRITE_OUT) {
        const __nv_bfloat162 r = __floats2bfloat162_rn(y.x, y.y);
        vw[k] = reinterpret_cast<const uint32_t &>(r);
      }
    }
    if constexpr (WRITE_OUT) out[idx] = v;
  }
  if constexpr (IS_DBIAS) {
    float4 *dst = reinterpret_cast<float4 *>(dbias_part + static_cast<size_t>(blockIdx.y) * cols +
                                             static_cast<size_t>(col8) * kVec);
    dst[0] = make_float4(sum[0].x, sum[0].y, sum[1].x, sum[1].y);
    dst[1] = make_float4(sum[2].x, sum[2].y, sum[3].x, sum[3].y);
  }
}

}  // namespace actbf16

/*! y = act(input) (IS_ACT), y = input * act'(act_input) (IS_DACT) or y = input, in BF16, with the
 *  column sums of y reduced into dbias (IS_DBIAS) through the two-call workspace protocol.
 *  output may be null when only the dbias is wanted. */
template <bool IS_ACT, bool IS_DACT, bool IS_DBIAS, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void act_bf16(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
              Tensor *workspace, cudaStream_t stream) {
  const auto [rows, cols] = input.flat_2d_dims();
  const size_t dbias_rows = DIVUP(rows, static_cast<size_t>(actbf16::kRowsPerCta));
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
  NVTE_CHECK(input.dtype() == DType::kBFloat16 && cols % actbf16::kVec == 0,
             "act_bf16 needs a BF16 input whose last dim is a multiple of 8.");
  if constexpr (IS_DACT) {
    NVTE_CHECK(act_input != nullptr && act_input->dtype() == DType::kBFloat16 &&
                   act_input->data.shape == input.data.shape,
               "act_bf16 dact needs a BF16 activation input of the input's shape.");
  }
  const bool write_out = output != nullptr && output->data.dptr != nullptr;
  if (write_out) {
    NVTE_CHECK(output->dtype() == DType::kBFloat16 && output->data.shape == input.data.shape,
               "act_bf16 output must be BF16 of the input's shape.");
  }
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  const dim3 grid(DIVUP(cols / actbf16::kVec, static_cast<size_t>(actbf16::kThreads)), dbias_rows);
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      write_out, WRITE_OUT,
      actbf16::act_bf16_kernel<IS_ACT, IS_DACT, IS_DBIAS, WRITE_OUT, ParamOP, OP>
      <<<grid, actbf16::kThreads, 0, stream>>>(
          reinterpret_cast<const uint4 *>(input.data.dptr),
          IS_DACT ? reinterpret_cast<const uint4 *>(act_input->data.dptr) : nullptr,
          write_out ? reinterpret_cast<uint4 *>(output->data.dptr) : nullptr, workspace_ptr,
          static_cast<int>(rows), static_cast<int>(cols)););
  NVTE_CHECK_CUDA(cudaGetLastError());
  if constexpr (IS_DBIAS) {
    common::reduce_dbias<bf16>(workspace_ptr, dbias, dbias_rows, cols, stream);
  }
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_NVFP4_ACT_BF16_CUH_
