/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/multi_stream.h>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/multi_stream.h"
#include "../util/math.h"
#include "../utils.cuh"
#include "nvfp4/act_bf16.cuh"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"
#include "transformer_engine/transpose.h"

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  // Create config with noop tensor
  QuantizationConfig quant_config;
  quant_config.noop_tensor = noop;

  nvte_quantize_v2(input, output, reinterpret_cast<NVTEQuantizationConfig>(&quant_config), stream);
}

void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  dispatch::dequantize_helper(*convertNVTETensorCheck(input), convertNVTETensorCheck(output),
                              stream);
}

void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_configs,
                                const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  const size_t num_streams = nvte_get_num_compute_streams();

  int num_stream_used = std::min(num_streams, num_tensors);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(cudaEventRecord(detail::get_compute_stream_event(0), stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaStreamWaitEvent(detail::get_compute_stream(s), detail::get_compute_stream_event(0)));
  }

  for (int i = 0; i < num_tensors; i++) {
    dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(
        inputs[i], outputs[i], quant_configs, detail::get_compute_stream(i % num_streams));
  }

  // record events on compute streams
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaEventRecord(detail::get_compute_stream_event(s), detail::get_compute_stream(s)));
  }
  // wait for all compute streams to finish
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream, detail::get_compute_stream_event(s)));
  }
}

void nvte_nvfp4_quantize_fused(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               NVTENVFP4FusedAct act, const NVTEQuantizationConfig quant_config,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_quantize_fused);
  using namespace transformer_engine;
  NVTE_CHECK(quant_config != nullptr, "nvte_nvfp4_quantize_fused needs a quantization config.");
  NVTE_CHECK(reinterpret_cast<const QuantizationConfig *>(quant_config)->nvfp4_amax_out != nullptr,
             "nvte_nvfp4_quantize_fused needs the NVFP4 amax-out tensor in the config.");
  const bool with_dbias = dbias != nullptr;
#define NVTE_NVFP4_FUSED_FWD(OP)                                                              \
  dispatch::quantize_fwd_helper</*IS_ACT=*/true, Empty, OP<fp32, fp32>>(input, output,         \
                                                                        quant_config, stream)
#define NVTE_NVFP4_FUSED_BWD(OP)                                                              \
  if (with_dbias) {                                                                           \
    dispatch::quantize_bwd_helper</*IS_DBIAS=*/true, /*IS_DACT=*/true, Empty, OP<fp32, fp32>>( \
        input, act_input, output, dbias, workspace, quant_config, stream);                    \
  } else {                                                                                    \
    dispatch::quantize_bwd_helper</*IS_DBIAS=*/false, /*IS_DACT=*/true, Empty, OP<fp32, fp32>>( \
        input, act_input, output, dbias, workspace, quant_config, stream);                    \
  }
  switch (act) {
    case kNVTENVFP4FusedActNone:
      if (with_dbias) {
        dispatch::quantize_bwd_helper</*IS_DBIAS=*/true, /*IS_DACT=*/false, Empty, nullptr>(
            input, act_input, output, dbias, workspace, quant_config, stream);
      } else {
        dispatch::quantize_fwd_helper</*IS_ACT=*/false, Empty, nullptr>(input, output,
                                                                        quant_config, stream);
      }
      break;
    case kNVTENVFP4FusedActGelu:
      NVTE_NVFP4_FUSED_FWD(gelu);
      break;
    case kNVTENVFP4FusedActGeluErf:
      NVTE_NVFP4_FUSED_FWD(gelu_erf);
      break;
    case kNVTENVFP4FusedActSilu:
      NVTE_NVFP4_FUSED_FWD(silu);
      break;
    case kNVTENVFP4FusedActQGelu:
      NVTE_NVFP4_FUSED_FWD(qgelu);
      break;
    case kNVTENVFP4FusedActRelu:
      NVTE_NVFP4_FUSED_FWD(relu);
      break;
    case kNVTENVFP4FusedActDGelu:
      NVTE_NVFP4_FUSED_BWD(dgelu);
      break;
    case kNVTENVFP4FusedActDGeluErf:
      NVTE_NVFP4_FUSED_BWD(dgelu_erf);
      break;
    case kNVTENVFP4FusedActDSilu:
      NVTE_NVFP4_FUSED_BWD(dsilu);
      break;
    case kNVTENVFP4FusedActDQGelu:
      NVTE_NVFP4_FUSED_BWD(dqgelu);
      break;
    case kNVTENVFP4FusedActDRelu:
      NVTE_NVFP4_FUSED_BWD(drelu);
      break;
    default:
      NVTE_ERROR("Unsupported NVFP4 fused activation (", static_cast<int>(act), ")");
  }
#undef NVTE_NVFP4_FUSED_FWD
#undef NVTE_NVFP4_FUSED_BWD
}

void nvte_nvfp4_act_bf16(const NVTETensor input, const NVTETensor act_input, NVTETensor output,
                         NVTETensor dbias, NVTETensor workspace, NVTENVFP4FusedAct act,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_act_bf16);
  using namespace transformer_engine;
  const Tensor &in = *convertNVTETensorCheck(input);
  const Tensor *act_in = act_input != nullptr ? convertNVTETensorCheck(act_input) : nullptr;
  Tensor *out = output != nullptr ? convertNVTETensor(output) : nullptr;
  Tensor *db = dbias != nullptr ? convertNVTETensor(dbias) : nullptr;
  Tensor *ws = workspace != nullptr ? convertNVTETensor(workspace) : nullptr;
  const bool with_dbias = db != nullptr;
#define NVTE_NVFP4_ACT_FWD(OP)                                                              \
  if (with_dbias) {                                                                         \
    dispatch::nvfp4::act_bf16<true, false, true, Empty, OP<fp32, fp32>>(in, act_in, out, db, ws, \
                                                                        stream);            \
  } else {                                                                                  \
    dispatch::nvfp4::act_bf16<true, false, false, Empty, OP<fp32, fp32>>(in, act_in, out, db, \
                                                                         ws, stream);       \
  }
#define NVTE_NVFP4_ACT_BWD(OP)                                                              \
  if (with_dbias) {                                                                         \
    dispatch::nvfp4::act_bf16<false, true, true, Empty, OP<fp32, fp32>>(in, act_in, out, db, ws, \
                                                                        stream);            \
  } else {                                                                                  \
    dispatch::nvfp4::act_bf16<false, true, false, Empty, OP<fp32, fp32>>(in, act_in, out, db, \
                                                                         ws, stream);       \
  }
  switch (act) {
    case kNVTENVFP4FusedActNone:
      if (with_dbias) {
        dispatch::nvfp4::act_bf16<false, false, true, Empty, dispatch::nvfp4::actbf16::identity>(
            in, act_in, out, db, ws, stream);
      } else {
        dispatch::nvfp4::act_bf16<false, false, false, Empty, dispatch::nvfp4::actbf16::identity>(
            in, act_in, out, db, ws, stream);
      }
      break;
    case kNVTENVFP4FusedActGelu:
      NVTE_NVFP4_ACT_FWD(gelu);
      break;
    case kNVTENVFP4FusedActGeluErf:
      NVTE_NVFP4_ACT_FWD(gelu_erf);
      break;
    case kNVTENVFP4FusedActSilu:
      NVTE_NVFP4_ACT_FWD(silu);
      break;
    case kNVTENVFP4FusedActQGelu:
      NVTE_NVFP4_ACT_FWD(qgelu);
      break;
    case kNVTENVFP4FusedActRelu:
      NVTE_NVFP4_ACT_FWD(relu);
      break;
    case kNVTENVFP4FusedActDGelu:
      NVTE_NVFP4_ACT_BWD(dgelu);
      break;
    case kNVTENVFP4FusedActDGeluErf:
      NVTE_NVFP4_ACT_BWD(dgelu_erf);
      break;
    case kNVTENVFP4FusedActDSilu:
      NVTE_NVFP4_ACT_BWD(dsilu);
      break;
    case kNVTENVFP4FusedActDQGelu:
      NVTE_NVFP4_ACT_BWD(dqgelu);
      break;
    case kNVTENVFP4FusedActDRelu:
      NVTE_NVFP4_ACT_BWD(drelu);
      break;
    default:
      NVTE_ERROR("Unsupported NVFP4 activation (", static_cast<int>(act), ")");
  }
#undef NVTE_NVFP4_ACT_FWD
#undef NVTE_NVFP4_ACT_BWD
}

void nvte_nvfp4_gated_bf16(const NVTETensor input, const NVTETensor grad, NVTETensor output,
                           NVTENVFP4FusedAct act, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_gated_bf16);
  using namespace transformer_engine;
  const Tensor &in = *convertNVTETensorCheck(input);
  const Tensor *g = grad != nullptr ? convertNVTETensorCheck(grad) : nullptr;
  Tensor *out = convertNVTETensorCheck(output);
  switch (act) {
    case kNVTENVFP4FusedActSwiGLU:
      dispatch::nvfp4::gated_bf16<false, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(in, g, out, stream);
      break;
    case kNVTENVFP4FusedActDSwiGLU:
      dispatch::nvfp4::gated_bf16<true, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(in, g, out, stream);
      break;
    default:
      NVTE_ERROR("Unsupported NVFP4 gated activation (", static_cast<int>(act), ")");
  }
}

void nvte_nvfp4_carry_amax(NVTETensor history, NVTETensor fresh, NVTETensor step, float margin,
                           NVTETensor amax_rowwise, NVTETensor amax_columnwise,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_carry_amax);
  using namespace transformer_engine;
  dispatch::nvfp4::carry_amax(*convertNVTETensorCheck(history), *convertNVTETensorCheck(fresh),
                              *convertNVTETensorCheck(step), margin,
                              *convertNVTETensorCheck(amax_rowwise),
                              *convertNVTETensorCheck(amax_columnwise), stream);
}
