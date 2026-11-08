# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNorm forward with MXFP8 output from Transformer Engine's fused kernel."""

import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.constants import TE_DType

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

MXFP8_ATTRS = (
    "_rowwise_data",
    "_columnwise_data",
    "_rowwise_scale_inv",
    "_columnwise_scale_inv",
)


def _assert_same_bytes(out, ref):
    for attr in MXFP8_ATTRS:
        torch.testing.assert_close(
            getattr(out, attr).view(torch.uint8),
            getattr(ref, attr).view(torch.uint8),
            atol=0,
            rtol=0,
            msg=attr,
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(
    os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0") == "1",
    reason="Tests Transformer Engine's fused kernel, not cuDNN",
)
@pytest.mark.parametrize("shape", [(128, 768), (256, 1024), (384, 1024)])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
def test_layernorm_fwd_mxfp8(shape, fp8_dtype, zero_centered_gamma, optimize_for_gemm):
    """The fused output is byte-identical to the BF16 forward followed by the MXFP8 cast."""
    torch.manual_seed(1234)
    rows, cols = shape
    eps = 1e-6
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    x[0] = 3.0
    x[1] = 0.0
    if optimize_for_gemm:
        x[2, 7] = float("nan")
        x[3, 9] = float("inf")
        x[32:64, 5] = float("nan")
    weight = 1 + 0.1 * torch.randn(cols, dtype=torch.bfloat16, device="cuda")
    bias = 0.1 * torch.randn(cols, dtype=torch.bfloat16, device="cuda")

    quantizer = MXFP8Quantizer(fp8_dtype)
    quantizer.optimize_for_gemm = optimize_for_gemm
    out, mu, rsigma = tex.layernorm_fwd(
        x, weight, bias, eps, None, quantizer, TE_DType[x.dtype], 0, zero_centered_gamma
    )
    assert out._with_gemm_swizzled_scales == optimize_for_gemm

    y, mu_ref, rsigma_ref = tex.layernorm_fwd(
        x, weight, bias, eps, None, None, TE_DType[x.dtype], 0, zero_centered_gamma
    )
    torch.testing.assert_close(mu, mu_ref, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(rsigma, rsigma_ref, atol=0, rtol=0, equal_nan=True)
    _assert_same_bytes(out, quantizer(y))


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(
    os.getenv("NVTE_NORM_FWD_USE_CUDNN", "0") == "1",
    reason="Tests Transformer Engine's fused kernel, not cuDNN",
)
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
def test_layernorm_fwd_mxfp8_sm_margin(optimize_for_gemm):
    """With all SMs but one reserved, the kernel runs over chunks of rows with the same output."""
    torch.manual_seed(1234)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count

    outputs = []
    for sm_margin in (0, sm_count - 1):
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        quantizer.optimize_for_gemm = optimize_for_gemm
        outputs.append(
            tex.layernorm_fwd(
                x, weight, bias, 1e-6, None, quantizer, TE_DType[x.dtype], sm_margin, False
            )
        )
    (out, mu, rsigma), (out_margin, mu_margin, rsigma_margin) = outputs
    torch.testing.assert_close(mu_margin, mu, atol=0, rtol=0)
    torch.testing.assert_close(rsigma_margin, rsigma, atol=0, rtol=0)
    _assert_same_bytes(out_margin, out)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("quantizer_kwargs", [dict(), dict(with_2d_quantization=True)])
@pytest.mark.parametrize("shape", [(128, 1536), (128, 4096)])
def test_layernorm_fwd_mxfp8_unfused_shapes(quantizer_kwargs, shape):
    """Shapes and quantizers the fused kernel does not take still quantize the BF16 forward."""
    torch.manual_seed(1234)
    rows, cols = shape
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(cols, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(cols, dtype=torch.bfloat16, device="cuda")

    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, **quantizer_kwargs)
    out, _, _ = tex.layernorm_fwd(x, weight, bias, 1e-6, None, quantizer, TE_DType[x.dtype], 0, False)
    y, _, _ = tex.layernorm_fwd(x, weight, bias, 1e-6, None, None, TE_DType[x.dtype], 0, False)
    _assert_same_bytes(out, quantizer(y))
