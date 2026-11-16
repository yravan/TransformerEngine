# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Operations with no compiled fused kernel in this build.

Each function here is the unfused reference: the same operation built from the
compiled bindings and PyTorch operations, equal to a fused kernel up to rounding.
Callers look a function up through this module at call time
(``fused_kernels.layernorm_bwd_add(...)``), so a faster implementation with the
same signature can be assigned over the module attribute and takes effect
everywhere.
"""

from typing import List, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex

from .module._common import qkv_rope_backward
from .quantized_tensor import QuantizedTensorStorage, Quantizer

__all__ = [
    "gelu_erf",
    "dgelu_erf",
    "geglu_erf",
    "dgeglu_erf",
    "dbias_dgelu_erf",
    "layernorm_bwd_add",
    "bgrad_quantize_rope",
]


def _quantize(
    tensor: torch.Tensor, quantizer: Optional[Quantizer]
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    return tensor if quantizer is None else quantizer(tensor)


def gelu_erf(
    input_: torch.Tensor, quantizer: Optional[Quantizer]
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    """Exact (erf) GELU, quantized when a quantizer is given."""
    return _quantize(torch.nn.functional.gelu(input_), quantizer)


def dgelu_erf(
    grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Optional[Quantizer]
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    """Backward of the exact (erf) GELU, quantized when a quantizer is given."""
    return _quantize(torch.ops.aten.gelu_backward(grad, fwd_input, approximate="none"), quantizer)


def geglu_erf(
    input_: torch.Tensor, quantizer: Optional[Quantizer]
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    """Exact (erf) GeGLU: GELU of the first half of the last dimension times the second half."""
    act_in, gate = input_.float().chunk(2, dim=-1)
    out = torch.nn.functional.gelu(act_in) * gate
    return _quantize(out.to(input_.dtype), quantizer)


def dgeglu_erf(
    grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Optional[Quantizer]
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    """Backward of the exact (erf) GeGLU, quantized when a quantizer is given."""
    act_in, gate = fwd_input.float().chunk(2, dim=-1)
    grad = grad.float()
    dact = torch.ops.aten.gelu_backward(torch.ones_like(act_in), act_in, approximate="none")
    out = torch.cat((dact * grad * gate, grad * torch.nn.functional.gelu(act_in)), dim=-1)
    return _quantize(out.to(fwd_input.dtype), quantizer)


def dbias_dgelu_erf(
    grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Optional[Quantizer]
) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
    """Backward of the exact (erf) GELU, then the bias gradient and the quantize.

    Returns the bias gradient and the (quantized) activation input gradient.
    """
    return tex.bgrad_quantize(dgelu_erf(grad, fwd_input, None), quantizer)


def layernorm_bwd_add(
    dz: torch.Tensor,
    x: torch.Tensor,
    add: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of LayerNorm with ``add`` added to the input gradient.

    Returns ``(dx + add, dgamma, dbeta)``.
    """
    dx, dgamma, dbeta = tex.layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)
    return dx.add_(add), dgamma, dbeta


def bgrad_quantize_rope(
    grad_output: torch.Tensor,
    quantizer: Optional[Quantizer],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    head_dim: int,
    rope_cols: int,  # pylint: disable=unused-argument
) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
    """Inverse RoPE on the q and k parts of a packed q/k/v gradient, then the bias
    gradient and the quantize.

    ``rope_cos`` / ``rope_sin`` are FP32 ``[seq_len, rot_dim]`` tables and ``rope_cols``
    is the width of the q and k parts, ``2 * heads * head_dim``. Returns the bias
    gradient and the (quantized) rotated gradient.
    """
    grad_output = qkv_rope_backward(grad_output, rope_cos, rope_sin, head_dim)
    return tex.bgrad_quantize(grad_output, quantizer)
