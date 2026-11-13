# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_layernorm_mlp_saves_only_columnwise_inputs(activation):
    """The saved FC1/FC2 inputs are only read column-wise by the weight-gradient GEMMs,
    so their row-wise MXFP8 copies must not be kept alive by the autograd context."""
    hidden, ffn, tokens = 1024, 4096, 2048
    module = te.LayerNormMLP(
        hidden, ffn, params_dtype=torch.bfloat16, activation=activation, bias=False
    ).cuda()
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe.MXFP8BlockScaling()):
        out = module(x)
    fn = out.grad_fn
    while fn is not None and not hasattr(fn, "saved_tensors"):
        fn = fn.next_functions[0][0]
    fp8_data = [tuple(t.shape) for t in fn.saved_tensors if t is not None and t.dtype == torch.uint8]
    # One fp8 copy of ln_out (tokens x hidden) and one of act_out (tokens x ffn); the
    # row-wise and column-wise MXFP8 data share the logical shape, so a kept row-wise
    # copy shows up as a second tensor of the same shape.
    assert fp8_data.count((tokens, hidden)) == 1, fp8_data
    assert fp8_data.count((tokens, ffn)) == 1, fp8_data
    out.float().sum().backward()
    assert module.fc1_weight.grad is not None and module.fc2_weight.grad is not None
