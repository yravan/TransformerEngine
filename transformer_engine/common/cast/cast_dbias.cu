/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../common.h"
#include "dispatch/quantize.cuh"

void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, nullptr>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_quantize_dbias_rope(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                              NVTETensor workspace, const NVTETensor rope_cos,
                              const NVTETensor rope_sin, size_t head_dim, size_t rope_cols,
                              cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_rope);
  using namespace transformer_engine;

  const Tensor *input_tensor = convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);
  Tensor *dbias_tensor = convertNVTETensorCheck(dbias);
  Tensor *workspace_tensor = convertNVTETensorCheck(workspace);
  const Tensor *cos_tensor = convertNVTETensorCheck(rope_cos);
  const Tensor *sin_tensor = convertNVTETensorCheck(rope_sin);

  NVTE_CHECK(output_tensor->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "RoPE backward is fused only into the MXFP8 quantize.");
  NVTE_CHECK(cos_tensor->dtype() == DType::kFloat32 && sin_tensor->dtype() == DType::kFloat32,
             "RoPE tables must be FP32.");
  NVTE_CHECK(cos_tensor->data.shape.size() == 2 && cos_tensor->data.shape == sin_tensor->data.shape,
             "RoPE tables must be 2D and of the same shape.");

  dispatch::mxfp8::RopeParams rope;
  rope.cos = reinterpret_cast<const float *>(cos_tensor->data.dptr);
  rope.sin = reinterpret_cast<const float *>(sin_tensor->data.dptr);
  rope.seq_len = static_cast<uint32_t>(cos_tensor->data.shape[0]);
  rope.rot_dim = static_cast<uint32_t>(cos_tensor->data.shape[1]);
  rope.head_dim = static_cast<uint32_t>(head_dim);
  rope.cols = static_cast<uint32_t>(rope_cols);

  Tensor dummy_noop;
  dispatch::mxfp8::quantize</*IS_DBIAS=*/true, /*IS_DACT=*/false, /*IS_ACT=*/false, Empty,
                            nullptr>(*input_tensor, nullptr, &dummy_noop, output_tensor,
                                     dbias_tensor, workspace_tensor,
                                     /*use_2d_quantization=*/false, stream, rope);
}
