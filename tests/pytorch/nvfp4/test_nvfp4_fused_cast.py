# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""The single-pass NVFP4 cast with carried global amax against the reference two-pass path."""

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401  (loads the extension)
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.utils import get_device_compute_capability

pytestmark = pytest.mark.skipif(
    get_device_compute_capability() < (10, 0), reason="NVFP4 needs Blackwell"
)


def _quantizers(stochastic_rounding: bool, margin: float = 1.0):
    kwargs = dict(
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=stochastic_rounding,
        with_random_sign_mask=True,
    )
    ref = NVFP4Quantizer(**kwargs)
    fused = NVFP4Quantizer(**kwargs, delayed_amax=True, amax_margin=margin)
    # Same random sign mask, so the RHT is the same transform.
    fused.rht_matrix_random_sign_mask_t = ref.rht_matrix_random_sign_mask_t
    fused._rebuild_derived_state()
    return ref, fused


def _dq(t):
    return t.dequantize(dtype=torch.float32)


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _col_dq(t):
    """Dequantize the columnwise (transposed, RHT'd) copy: compact E4M3 scales, 16-wide blocks."""
    data = t._columnwise_data.view(torch.uint8)
    rows, half = data.shape
    vals = torch.stack([data & 0xF, data >> 4], dim=-1).reshape(rows, half * 2)
    vals = _E2M1.to(data.device)[vals.long()]
    scale = t._columnwise_scale_inv[:rows, : half * 2 // 16].float()
    global_dec = t._amax_columnwise.float() / (6.0 * 448.0)
    return vals * scale.repeat_interleave(16, dim=1) * global_dec


@pytest.mark.parametrize("shape", [(256, 1024), (4096, 1024), (1024, 1856), (2048, 4096)])
@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_plain_cast_matches_reference(shape, stochastic_rounding):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    ref, fused = _quantizers(stochastic_rounding)
    y_ref = ref(x)
    # First call is the exact two-pass path and seeds the carry; the second call is single pass
    # with the carried amax of the identical tensor, so both must agree with the reference.
    fused(x)
    y = fused(x)
    assert torch.equal(y._amax_rowwise, y_ref._amax_rowwise)
    assert torch.allclose(y._amax_columnwise, y_ref._amax_columnwise, rtol=1e-6, atol=0)
    assert torch.equal(y._rowwise_scale_inv, y_ref._rowwise_scale_inv)
    dq, dq_ref = _dq(y), _dq(y_ref)
    if stochastic_rounding:
        step = (dq_ref.abs().amax(dim=1, keepdim=True) / 6 * 2).clamp(min=1e-3)
        assert ((dq - dq_ref).abs() <= step).all()
    else:
        assert torch.equal(y._rowwise_data, y_ref._rowwise_data)
    # Columnwise: same transform, f32 sums in a different order -> rare one-step differences.
    cdq, cdq_ref = _col_dq(y), _col_dq(y_ref)
    scales_equal = (y._columnwise_scale_inv == y_ref._columnwise_scale_inv).float().mean()
    assert scales_equal > 0.999
    diff = (cdq - cdq_ref).abs()
    step = cdq_ref.abs().amax(dim=1, keepdim=True) / 6 * (2 if stochastic_rounding else 1)
    assert (diff <= step + 1e-6).all()
    if not stochastic_rounding:
        assert (diff > 0).float().mean() < 1e-3


@pytest.mark.parametrize("act", ["gelu", "gelu_erf"])
def test_fused_activation_matches_unfused(act):
    torch.manual_seed(0)
    x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    ref, fused = _quantizers(False)
    fwd = getattr(tex, act)
    y_ref = fwd(x, ref)
    fwd(x, fused)
    y = fwd(x, fused)
    assert torch.equal(y._amax_rowwise, y_ref._amax_rowwise)
    assert torch.equal(y._rowwise_data, y_ref._rowwise_data)
    assert torch.equal(y._rowwise_scale_inv, y_ref._rowwise_scale_inv)
    cdq, cdq_ref = _col_dq(y), _col_dq(y_ref)
    step = cdq_ref.abs().amax(dim=1, keepdim=True) / 6
    assert ((cdq - cdq_ref).abs() <= step + 1e-6).all()


@pytest.mark.parametrize("act", ["dgelu", "dgelu_erf"])
def test_fused_dbias_dact_matches_unfused(act):
    torch.manual_seed(0)
    x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    ref, fused = _quantizers(True)
    fn = getattr(tex, "dbias_" + act)
    db_ref, y_ref = fn(dy, x, ref)
    fn(dy, x, fused)
    db, y = fn(dy, x, fused)
    assert torch.equal(y._amax_rowwise, y_ref._amax_rowwise)
    assert torch.allclose(db.float(), db_ref.float(), rtol=2e-2, atol=1e-2)
    dq, dq_ref = _dq(y), _dq(y_ref)
    step = dq_ref.abs().amax(dim=1, keepdim=True) / 6 * 2
    assert ((dq - dq_ref).abs() <= step + 1e-6).all()


def test_plain_dbias_matches_unfused():
    torch.manual_seed(0)
    dy = torch.randn(2048, 1024, dtype=torch.bfloat16, device="cuda")
    ref, fused = _quantizers(True)
    db_ref, y_ref = tex.bgrad_quantize(dy, ref)
    tex.bgrad_quantize(dy, fused)
    db, y = tex.bgrad_quantize(dy, fused)
    assert torch.equal(y._amax_rowwise, y_ref._amax_rowwise)
    assert torch.allclose(db.float(), db_ref.float(), rtol=2e-2, atol=1e-2)


def test_carried_amax_tracks_growth():
    """The carried amax is the max over the history times the margin; values above it saturate."""
    torch.manual_seed(0)
    ref, fused = _quantizers(False, margin=2.0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    fused(x)
    y = fused(x)
    assert torch.allclose(y._amax_rowwise, 2.0 * x.float().abs().amax(), rtol=1e-6)
    big = x * 8
    y = fused(big)  # cast with the stale (2x) amax: values beyond it saturate
    dq = _dq(y)
    assert dq.abs().amax() <= 2.0 * x.float().abs().amax() * (1 + 1e-6)
    y = fused(big)  # next cast carries the fresh amax of `big`
    assert torch.allclose(y._amax_rowwise, 2.0 * big.float().abs().amax(), rtol=1e-6)
