# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

SHAPES = [(128, 256), (256, 1024), (4096, 1536), (2048, 4096)]
FP8_DTYPES = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]


def misaligned(t: torch.Tensor) -> torch.Tensor:
    flat = torch.empty(t.numel() + 8, dtype=t.dtype, device=t.device)
    out = flat[8:].view(t.shape)
    out.copy_(t)
    return out


def quantizer(dtype):
    q = MXFP8Quantizer(fp8_dtype=dtype, rowwise=True, columnwise=True)
    q.optimize_for_gemm = True
    return q


def parts(y):
    return (
        y._rowwise_data.view(torch.uint8),
        y._rowwise_scale_inv,
        y._columnwise_data.view(torch.uint8),
        y._columnwise_scale_inv,
    )


def assert_same_quantization(a, b):
    for u, v in ((parts(a)[0], parts(b)[0]), (parts(a)[2], parts(b)[2])):
        differ = u != v
        both_zero = ((u | v) == 0x80) & ((u & v) == 0)
        assert not (differ & ~both_zero).any()
    torch.testing.assert_close(parts(a)[1], parts(b)[1], rtol=0, atol=0)
    torch.testing.assert_close(parts(a)[3], parts(b)[3], rtol=0, atol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_gelu_erf_register_resident_matches_staged(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 4
    fused = tex.gelu_erf(x, quantizer(dtype))
    staged = tex.gelu_erf(misaligned(x), quantizer(dtype))
    assert_same_quantization(fused, staged)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_dgelu_erf_register_resident_matches_staged(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 4
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    fused = tex.dgelu_erf(g, x, quantizer(dtype))
    staged = tex.dgelu_erf(misaligned(g), misaligned(x), quantizer(dtype))
    assert_same_quantization(fused, staged)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_dbias_dgelu_erf_register_resident_matches_staged(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 4
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    dbias, fused = tex.dbias_dgelu_erf(g, x, quantizer(dtype))
    dbias_staged, staged = tex.dbias_dgelu_erf(misaligned(g), misaligned(x), quantizer(dtype))
    assert_same_quantization(fused, staged)
    reference = (
        g.float()
        * (
            0.5 * (1.0 + torch.erf(x.float() * 0.7071067811865476))
            + x.float() * torch.exp(-0.5 * x.float() ** 2) * 0.3989422804014327
        )
    ).sum(0)
    tolerance = 2e-2 * reference.abs().max()
    torch.testing.assert_close(dbias.float(), reference, rtol=0, atol=tolerance)
    torch.testing.assert_close(dbias.float(), dbias_staged.float(), rtol=0, atol=tolerance)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_swiglu_register_resident_matches_staged(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn((shape[0], 2 * shape[1]), dtype=torch.bfloat16, device="cuda") * 4
    fused = tex.swiglu(x, quantizer(dtype))
    staged = tex.swiglu(misaligned(x), quantizer(dtype))
    assert_same_quantization(fused, staged)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_dswiglu_register_resident_matches_staged(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn((shape[0], 2 * shape[1]), dtype=torch.bfloat16, device="cuda") * 4
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    fused = tex.dswiglu(g, x, quantizer(dtype))
    staged = tex.dswiglu(misaligned(g), misaligned(x), quantizer(dtype))
    assert_same_quantization(fused, staged)
