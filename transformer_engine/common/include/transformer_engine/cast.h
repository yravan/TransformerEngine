/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Quantize the tensor
 *
 *  The type of quantized tensor in the output depends on the scaling mode of the output
 *  tensor.
 *
 *  Supported formats are:
 *
 *  1) MXFP8 scaling (for compute capability 10.0 or newer)
 *
 *  The MXFP8 implementation is per the microscaling format MXFP8 defined by the OCP specification:
 *  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *
 *
 *  Supported modes of MXFP8 scaling (live scaling) for scaling mode NVTE_MXFP8_1D_SCALING
 *      a) Rowwise scaling (along the dim=0) computes one set of the output data, which includes:
 *          - the scaled output tensor
 *          - the corresponding scaling factors
 *         The scaling factors are computed for blocks of the shape [1,32]
 *         (i.e., each scaling factor spans 32 contiguous elements along rows).
 *
 *      b) Columwise scaling (along the dim=1) computes one set of the output data.
 *         The scaling factors are computed for blocks of the shape [32,1]
 *         (i.e., each scaling factor spans 32 contiguous elements along columns).
 *
 *      c) Both rowwise AND columnwise scaling (along the dim=0 and the dim=1)
 *         computes two sets of the output data: both 1) and 2).
 *
 *  The shape of the MX block must be specified in the 'output' argument,
 *  and can be either [1,32] or [32,1] as no other shapes are currently supported.
 *
 *  To cast the input tensor to the MXFP8, the scaling_mode.delayed_scaling parameter
 *  of the output tensor should be set to 0.
 *
 *  2) NVTE_DELAYED_TENSOR_SCALING that quantize the entire tensor
 *  using a single scaling factor. The absolute maximum value of the tensor should
 *  be precalculated either online (current scaling) or based on a tensor history
 *  (delayed scaling). The calls to nvte_quantize scale based on that data value.
 *  Note the NVTE_DELAYED_TENSOR_SCALING NVTEScalingMode is reused for online
 *  per tensor scaling.
 *
 *
 *  3) FP8 block scaling formats NVTE_BLOCK_SCALING_1D and NVTE_BLOCK_SCALING_2D
 *  for compute capability of at least 9.0. These modes quantize the tensor by blocks
 *  of size 1x128 (with columnwise mode of 128x1) and 128x128 respectively.
 *
 *  The supported modes are:
 *      a) Rowwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with identical shape to the
 *            input tensor.
 *          - Scale factors which are computed for either 1D 1x128 or 2D 128x128 blocks.
 *      b) Columnwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with a shape equivalent to
 *            the transpose of the input tensor.
 *          - Scale factors which are calculated for either 1D 128x1 or 2D 128x128 blocks
 *            of the input tensor.
 *      c) Both: In which both tensors and both scales are calculated.
 *
 *  This quantization mode includes both the calculation of the scaling factors
 *  per-tile and quantization of the row and/or columnwise tiles. No precalculated
 *  absolute max is required. The scaling factors are also rounded to powers of 2.
 */

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8/BlockwiseFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input grouped tensor.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *         NVFP4 output requires row scaling, BF16 input, and dimensions that are multiples of 128.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in,out] output           Output grouped tensor.
 *  \param[in]     quant_config     Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[out]     noop             Noop tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream);

/*! \brief Casts input tensor to quantized output tensor, with advanced quantization options.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[in]      quant_config     Quantization configuration.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts input tensor to MXFP8. Additionally, reduces the input along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream);

/*! \brief Casts input grouped tensor to MXFP8. Additionally, reduces the input along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     quant_config     Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                               NVTEGroupedTensor dbias, NVTETensor workspace,
                               const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Computes backward of GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of the exact (erf) GeLU operation on the input, then casts to
 *         FP8/MXFP8, and computes dbias. Same contract as nvte_quantize_dbias_dgelu.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dgelu_erf(const NVTETensor input, const NVTETensor act_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream);

/*! \brief Computes backward of GeLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the GeLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dgelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of SiLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the SiLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of SiLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the SiLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dsilu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of ReLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the ReLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_drelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                     NVTEGroupedTensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Computes backward of Quick GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Quick GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Computes backward of Quick GeLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Quick GeLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dqgelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                      NVTEGroupedTensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Computes backward of Squared ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Squared ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Computes backward of Squared ReLU operation on the grouped input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Squared ReLU backward along columns.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         Grouped dbias is not yet supported for grouped tensors with a varying last dimension.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input grouped tensor to be cast.
 *  \param[in]     act_input        Activation input grouped tensor.
 *  \param[in,out] output           Output grouped FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_group_quantize_dbias_dsrelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor act_input, NVTEGroupedTensor output,
                                      NVTEGroupedTensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Casts input tensor from reduced to higher precision.
 *         In case of the MXFP8 dequantization, the dequantized values are stored to the rowwise
 *         data of the output tensor, regardless of whether the row- or columnwise scaling is used.
 *
 *  \param[in]     input     Input FP8/MXFP8 tensor to be cast.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input grouped tensor from reduced to higher precision.
 *         In case of the MXFP8 dequantization, the dequantized values are stored to the rowwise
 *         data of the output tensor, regardless of whether the row- or columnwise scaling is used.
 *
 *  \param[in]     input     Input grouped FP8/MXFP8 tensor to be cast.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dequantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           cudaStream_t stream);

/*! \brief Fused grouped requantization. Currently only MXFP8 1D scaling is supported.
 *
 * Makes a rowwise-only grouped MXFP8 tensor GEMM-ready in one kernel:
 * dequantizes, builds the columnwise MXFP8 copy with GEMM-swizzled per-group
 * scales, and re-emits the rowwise scales at their GEMM-swizzled addresses.
 * Optionally also materializes the BF16 dequantized values. Replaces the
 * separate group_dequantize -> group_quantize(columnwise) ->
 * grouped_swizzle(rowwise scales) chain.
 *
 * Requirements: SM100+, MXFP8 1D scaling, the hidden dim and (caller contract,
 * offsets are device-resident) every group's row count multiples of 128. Rows
 * at or past tensor_offsets[num_groups] (capacity-mode / paged-stash tail) are
 * left untouched, matching the unfused chain. On return the output tensor's
 * with_gemm_swizzled_scales metadata is set.
 *
 *  \param[in]     input           Rowwise MXFP8 tensor with compact (unswizzled)
 *                                 E8M0 scales; groups stacked along the first dim.
 *  \param[in,out] output          Destination: columnwise E4M3 data +
 *                                 columnwise_scale_inv (per-group GEMM-swizzled)
 *                                 and scale_inv (the GEMM-swizzled copy of the
 *                                 input's rowwise scales). Rowwise data is not
 *                                 written; the GEMM keeps consuming the input's.
 *  \param[in]     tensor_offsets  Int64 device tensor of num_groups + 1 exclusive
 *                                 ELEMENT offsets, offsets[g] = row offset x cols
 *                                 (the grouped tensor's cached tensor_offsets;
 *                                 offsets[0] = 0, offsets[num_groups] = live rows
 *                                 x cols, which may be less than the allocated
 *                                 rows x cols).
 *  \param[out]    dequantized     Optional BF16 [rows, cols] tensor; pass NULL
 *                                 (or an unallocated tensor) to skip it.
 *  \param[in]     quant_config    Quantization options. `use_fast_math` selects a
 *                                 BF16 intermediate instead of the default FP32
 *                                 path.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_group_requantize(const NVTETensor input, NVTETensor output,
                           const NVTETensor tensor_offsets, NVTETensor dequantized,
                           const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts multiple input tensors to quantized output tensors.
 *
 *  \param[in]      inputs          List of input tensors to be cast.
 *  \param[in,out]  outputs         List of output quantized tensors.
 *  \param[in]      quant_config    (Optional) Quantization configurations.
 *  \param[in]      num_tensors     Number of input and output tensors.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_config, const size_t num_tensors,
                                cudaStream_t stream);

/*! \brief Casts grouped input tensor to quantized output tensors.
 *
 *  \param[in]      input           Input tensor to be cast.
 *  \param[in,out]  outputs         Output quantized tensors.
 *  \param[in]      split_sections  Split sections of the input tensor.
 *  \param[in]      num_tensors     Number of output tensors.
 *  \param[in]      quant_config    (Optional) Quantization configurations.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_group_nvfp4_quantize_with_amax(const NVTETensor input, NVTETensor *outputs,
                                         const size_t *split_sections, size_t num_tensors,
                                         const NVTEQuantizationConfig quant_config,
                                         cudaStream_t stream);

/*! \deprecated Use nvte_group_amax instead. */
void nvte_group_nvfp4_compute_amax(const NVTETensor input, NVTETensor *outputs,
                                   const size_t *split_sections, size_t num_tensors,
                                   cudaStream_t stream);

/*! \enum NVTENVFP4FusedAct
 *  \brief Activation fused into the single-pass NVFP4 cast (nvte_nvfp4_quantize_fused).
 */
enum NVTENVFP4FusedAct {
  kNVTENVFP4FusedActNone = 0,
  kNVTENVFP4FusedActGelu = 1,
  kNVTENVFP4FusedActGeluErf = 2,
  kNVTENVFP4FusedActDGelu = 3,
  kNVTENVFP4FusedActDGeluErf = 4,
  kNVTENVFP4FusedActSilu = 5,
  kNVTENVFP4FusedActDSilu = 6,
  kNVTENVFP4FusedActQGelu = 7,
  kNVTENVFP4FusedActDQGelu = 8,
  kNVTENVFP4FusedActRelu = 9,
  kNVTENVFP4FusedActDRelu = 10
};

/*! \brief Single-pass NVFP4 cast (rowwise + RHT'd columnwise) of a BF16 tensor, optionally fused
 *         with an activation (forward) or an activation gradient and the bias gradient (backward).
 *
 *  The output's amax fields must already hold the global amaxes to encode with (pre-RHT for the
 *  rowwise copy, post-RHT for the columnwise one); the fresh amaxes of what was quantized are
 *  accumulated by atomic max into the 2-float tensor set with kNVTEQuantizationConfigNVFP4AmaxOut,
 *  which must be set (together with kNVTEQuantizationConfigNVFP4RHTSignMask). dbias and workspace
 *  follow the nvte_quantize_dbias protocol and are ignored for forward activations.
 *
 *  \param[in]      input         BF16 input (the gradient for backward activations).
 *  \param[in]      act_input     Activation input for backward activations, NULL otherwise.
 *  \param[in,out]  output        NVFP4 output with rowwise and columnwise data.
 *  \param[out]     dbias         Bias gradient (backward only; NULL for none).
 *  \param[out]     workspace     Workspace for the bias gradient.
 *  \param[in]      act           Fused activation.
 *  \param[in]      quant_config  Quantization configuration.
 *  \param[in]      stream        CUDA stream used for the operation.
 */
void nvte_nvfp4_quantize_fused(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               NVTENVFP4FusedAct act, const NVTEQuantizationConfig quant_config,
                               cudaStream_t stream);

/*! \brief Carry the global amaxes of an NVFP4 quantizer from its previous casts to the next one.
 *
 *  Folds `fresh` (2 floats: rowwise, columnwise) into slot `step % history_len` of `history`
 *  ([2, history_len] floats), increments `step` (1 int32), writes `margin` times the maximum
 *  over the history into `amax_rowwise` / `amax_columnwise` (1 float each) and zeroes `fresh`.
 */
void nvte_nvfp4_carry_amax(NVTETensor history, NVTETensor fresh, NVTETensor step, float margin,
                           NVTETensor amax_rowwise, NVTETensor amax_columnwise,
                           cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
