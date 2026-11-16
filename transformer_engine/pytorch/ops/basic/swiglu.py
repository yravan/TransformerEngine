# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for SwiGLU and variants."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...constants import DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...dynamo import TensorOrQuantized, TensorSpec
from ...dynamo.fake import fake_activation
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_autocast_dtype, maybe_dequantize

__all__ = [
    "SwiGLU",
    "SiTUGLU",
    "ClampedSwiGLU",
    "ScaledSwiGLU",
    "ScaledSiTUGLU",
    "ScaledClampedQGeGLU",
]


@dataclass(slots=True)
class SwiGLUFwdArgs:
    """Tensor input and configuration for the SwiGLU forward custom op."""

    input_: TensorOrQuantized
    dtype: torch.dtype
    glu_interleave_size: Optional[int]
    output_quantizer: Optional[Quantizer]
    # Quantizes the input saved for backward (cache_quantized_input); None keeps it as is.
    saved_input_quantizer: Optional[Quantizer]
    grad_input_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class SwiGLUBwdArgs:
    """Saved tensor and configuration for the SwiGLU backward custom op."""

    grad_output: TensorOrQuantized
    input_: TensorOrQuantized
    dtype: torch.dtype
    glu_interleave_size: Optional[int]
    grad_input_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class SiTUGLUFwdArgs(SwiGLUFwdArgs):
    """SwiGLU forward args plus the SiTUGLU soft-cap parameters."""

    beta1: float
    beta2: float


@dataclass(slots=True)
class SiTUGLUBwdArgs(SwiGLUBwdArgs):
    """SwiGLU backward args plus the SiTUGLU soft-cap parameters."""

    beta1: float
    beta2: float


def _deinterleave_glu(x: torch.Tensor, interleave_size: int) -> torch.Tensor:
    """Block-interleaved gates and linear units -> concatenated halves."""
    shape = x.size()
    x = x.reshape(-1, shape[-1] // (2 * interleave_size), 2, interleave_size)
    return x.transpose(1, 2).contiguous().view(shape)


def _interleave_glu(x: torch.Tensor, interleave_size: int) -> torch.Tensor:
    """Concatenated halves -> block-interleaved gates and linear units."""
    shape = x.size()
    x = x.reshape(-1, 2, shape[-1] // (2 * interleave_size), interleave_size)
    return x.transpose(1, 2).contiguous().view(shape)


class SwiGLU(BasicOperation):
    r"""Swish gated linear unit

    The input tensor is split into chunks :math:``a`` and :math:``b``
    along the last dimension and the following is computed:

    .. math::

       \text{SwiGLU}(a,b) = \text{SiLU}(a) * b

    where

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:``a`` and
       :math:``b``. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    The Sigmoid Linear Unit (SiLU) gating function is also known as
    the swish function. See
    `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.
    glu_interleave_size : int, optional
        When set, the GLU activations will use a block interleaved
        format. Instead of interpreting the input tensor as a
        concatenation of gates and linear units (e.g.
        :math:``[a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4]``
        in the above notation), it will be interpreted
        as alternating blocks of gates and linear units (e.g.
        :math:``[a_1, a_2, b_1, b_2, a_3, a_4, b_3, b_4]``
        when the interleave size is 2). This data format is highly
        experiental and is primarily intended to support some advanced
        fused kernels.

    """

    fwd_args_type = SwiGLUFwdArgs
    bwd_args_type = SwiGLUBwdArgs

    def __init__(
        self,
        *,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size
        self._saved_input_quantizer: Optional[Quantizer] = None
        if cache_quantized_input:
            quantizer = Float8CurrentScalingQuantizer(
                DType.kFloat8E4M3, device=None, rowwise=True, columnwise=False
            )
            quantizer.internal = True
            self._saved_input_quantizer = quantizer

    @classmethod
    def _tex_swiglu_forward(
        cls,
        args: SwiGLUFwdArgs,  # pylint: disable=unused-argument
        input_: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call the Transformer Engine SwiGLU forward kernel."""
        return tex.swiglu(input_, quantizer)

    @classmethod
    def _tex_swiglu_backward(
        cls,
        args: SwiGLUBwdArgs,  # pylint: disable=unused-argument
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call the Transformer Engine SwiGLU backward kernel."""
        return tex.dswiglu(grad_output, input_, quantizer)

    def _kernel_args(self) -> dict[str, Any]:
        """Kernel parameters this instance adds to its op args (none for SwiGLU)."""
        return {}

    def pack_forward_args(
        self,
        basic_op_ctxs,
        input_,
        *,
        prev_op_grad_output_quantizer,
        next_op_input_quantizer,
        basic_op_kwargs,
        **unused,  # pylint: disable=unused-argument
    ) -> SwiGLUFwdArgs:
        if basic_op_kwargs[0]:
            raise ValueError(f"{self.__class__.__name__} forward does not expect keyword arguments")
        dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")
        requires_grad = basic_op_ctxs[0].requires_grad
        return self.fwd_args_type(
            input_=input_,
            dtype=dtype,
            glu_interleave_size=self.glu_interleave_size,
            output_quantizer=next_op_input_quantizer,
            saved_input_quantizer=self._saved_input_quantizer if requires_grad else None,
            grad_input_quantizer=prev_op_grad_output_quantizer,
            **self._kernel_args(),
        )

    @classmethod
    def forward_compute(
        cls, args: SwiGLUFwdArgs, *, in_custom_op: bool = False
    ):  # pylint: disable=unused-argument
        x = maybe_dequantize(args.input_, args.dtype)
        swiglu_in = x
        if args.glu_interleave_size is not None:
            swiglu_in = _deinterleave_glu(x, args.glu_interleave_size)
        out = cls._tex_swiglu_forward(args, swiglu_in, args.output_quantizer)
        saved_input = None
        if args.saved_input_quantizer is not None:
            saved_input = args.saved_input_quantizer(x)
        return out, [()], (saved_input,)

    @classmethod
    def forward_compute_fake(cls, args: SwiGLUFwdArgs):
        out = fake_activation(args.input_, args.output_quantizer, gated=True, dtype=args.dtype)
        saved_input = None
        if args.saved_input_quantizer is not None:
            saved_input = TensorSpec(
                shape=args.input_.shape,
                dtype=args.dtype,
                device=args.input_.device,
                quantizer=args.saved_input_quantizer,
            )
        return out, [()], (saved_input,)

    def forward_setup_context(self, basic_op_ctxs, args, aux) -> None:
        ctx = basic_op_ctxs[0]
        if not ctx.requires_grad:
            return
        (saved_input,) = aux
        if saved_input is None:
            saved_input = args.input_
        if is_cpu_offload_enabled():
            mark_activation_offload(saved_input)
        ctx.save_for_backward(saved_input)
        ctx.dtype = args.dtype
        ctx.grad_input_quantizer = args.grad_input_quantizer

    def pack_backward_args(
        self, basic_op_ctxs, grad_output, **unused  # pylint: disable=unused-argument
    ) -> SwiGLUBwdArgs:
        ctx = basic_op_ctxs[0]
        return self.bwd_args_type(
            grad_output=grad_output,
            input_=ctx.saved_tensors[0],
            dtype=ctx.dtype,
            glu_interleave_size=self.glu_interleave_size,
            grad_input_quantizer=ctx.grad_input_quantizer,
            **self._kernel_args(),
        )

    @classmethod
    def backward_compute(cls, args: SwiGLUBwdArgs, *, in_custom_op: bool = False):
        x = maybe_dequantize(args.input_, args.dtype)
        dy = maybe_dequantize(args.grad_output, args.dtype)
        interleave_size = args.glu_interleave_size
        swiglu_in = x
        quantizer = args.grad_input_quantizer
        if interleave_size is not None:
            swiglu_in = _deinterleave_glu(x, interleave_size)
            quantizer = None
        dx = cls._tex_swiglu_backward(args, dy, swiglu_in, quantizer)
        if interleave_size is not None:
            dx = _interleave_glu(dx, interleave_size)
        if not in_custom_op:
            clear_tensor_data(args.input_)
        return dx, [()], [()]

    @classmethod
    def backward_compute_fake(cls, args: SwiGLUBwdArgs):
        quantizer = None if args.glu_interleave_size is not None else args.grad_input_quantizer
        dx = fake_activation(args.input_, quantizer, dtype=args.dtype)
        return dx, [()], [()]


class SiTUGLU(SwiGLU):
    r"""Soft-capped SiLU gated linear unit used by Kimi K3.

    See the `Kimi K3 technical report
    <https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>`__.

    The input is split into gate and up-projection halves and computes

    .. math::

       \beta_1 \tanh(a / \beta_1) \sigma(a)
       \; \beta_2 \tanh(b / \beta_2).

    Parameters
    ----------
    beta1 : float, default = 4.0
        Positive gate soft-cap parameter.
    beta2 : float, default = 25.0
        Positive up-branch soft-cap parameter.
    cache_quantized_input : bool, default = False
        Quantize the saved input for backward, as in :class:`SwiGLU`.
    glu_interleave_size : int, optional
        Block-interleaved GLU layout, as in :class:`SwiGLU`.
    """

    def __init__(
        self,
        *,
        beta1: float = 4.0,
        beta2: float = 25.0,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            cache_quantized_input=cache_quantized_input,
            glu_interleave_size=glu_interleave_size,
        )
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        if not math.isfinite(self.beta1) or self.beta1 <= 0.0:
            raise ValueError(f"beta1 must be finite and positive, got {self.beta1}")
        if not math.isfinite(self.beta2) or self.beta2 <= 0.0:
            raise ValueError(f"beta2 must be finite and positive, got {self.beta2}")

    fwd_args_type = SiTUGLUFwdArgs
    bwd_args_type = SiTUGLUBwdArgs

    def _kernel_args(self) -> dict[str, Any]:
        return {"beta1": self.beta1, "beta2": self.beta2}

    @classmethod
    def _tex_swiglu_forward(
        cls,
        args: SiTUGLUFwdArgs,
        input_: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        return tex.situglu(input_, quantizer, args.beta1, args.beta2)

    @classmethod
    def _tex_swiglu_backward(
        cls,
        args: SiTUGLUBwdArgs,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        return tex.dsituglu(grad_output, input_, quantizer, args.beta1, args.beta2)


class ClampedSwiGLU(BasicOperation):
    r"""GPT-OSS
    Implementation based on `GPT-OSS <https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250>`__.

    This activation has two differences compared to the original SwiGLU
       1. Both gate and pre-activations are clipped based on parameter limit.
       2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation.

    .. warning::

       The input tensor is chunked along the last dimension to get
       gates/pre-activations which is different from GPT OSS
       implementation where the gates/pre-activations are assumed to
       be interleaved in the input tensor.

    Parameters
    ----------
    limit : float
        The clamp limit.
    alpha : float
        The scaling factor for the sigmoid function used in the activation.
    glu_linear_offset : float
        Offset added to the linear (gate) component after clamping.
        Set to ``0.0`` to disable the offset.
    cache_quantized_input : bool, default = ``False``
        Quantize input tensor when caching for use in the backward pass.
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.

    """

    def __init__(
        self,
        *,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.limit: float = limit
        self.alpha: float = alpha
        self.glu_linear_offset: float = glu_linear_offset
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def _tex_clamped_swiglu_forward(
        self,
        swiglu_in: torch.Tensor,
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_swiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_swiglu(
            swiglu_in,
            next_op_input_quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

    def _tex_clamped_dswiglu(
        self,
        dy: torch.Tensor,
        swiglu_in: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_dswiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_dswiglu(
            dy,
            swiglu_in,
            quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Compute dtype
        dtype: torch.dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")

        # Check input tensor
        x = maybe_dequantize(input_.contiguous(), dtype)

        # Remove interleaving if needed
        swiglu_in = x
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Launch kernel
        out = self._tex_clamped_swiglu_forward(swiglu_in, next_op_input_quantizer)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.save_for_backward(x)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (input_,) = ctx.saved_tensors

        # Make sure tensors have correct dtypes
        x = maybe_dequantize(input_.contiguous(), ctx.dtype)
        dy = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        # Remove interleaving if needed
        swiglu_in = x
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Quantizer for grad input
        quantizer = ctx.prev_op_grad_output_quantizer
        if self.glu_interleave_size is not None:
            quantizer = None

        # Launch kernel
        grad_swiglu_in = self._tex_clamped_dswiglu(dy, swiglu_in, quantizer)

        # Apply interleaving if needed
        dx = grad_swiglu_in
        if self.glu_interleave_size is not None:
            shape = dx.size()
            dx = dx.reshape(
                -1,
                2,
                shape[-1] // (2 * self.glu_interleave_size),
                self.glu_interleave_size,
            )
            dx = dx.transpose(1, 2).contiguous()
            dx = dx.view(shape)

        # Clear input tensor if possible
        clear_tensor_data(input_)

        return dx, ()


class _ScaledGLU(BasicOperation):
    """SwiGLU-family activation with per-row scales (fused grouped MLP middle op)."""

    num_extra_inputs: int = 1

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.glu_interleave_size: Optional[int] = glu_interleave_size
        if activation_recompute_in_mlp:
            raise ValueError(
                f"{self.__class__.__name__} does not support activation recomputation "
                "in the fused grouped MLP"
            )
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[torch.Tensor]]]:
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        extra_input = basic_op_extra_inputs[0][0]
        # Determine compute dtype
        dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
        extra_input_dtype = maybe_autocast_dtype(default_dtype=extra_input.dtype)

        # Prepare the activation input in the compute dtype.
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra_input, extra_input_dtype)
        out = self._scaled_glu_forward(input_, scales)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_dtype = extra_input_dtype
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                input_,
                scales if ctx.input_requires_grad or ctx.extra_input_requires_grad else None,
            )

        return out, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.extra_input_dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)

        grad_input, grad_extra_input = self._scaled_glu_backward(
            grad_output,
            input_,
            scales,
            compute_scale_grad=ctx.extra_input_requires_grad,
        )
        if not ctx.input_requires_grad:
            grad_input = None

        # Clear input tensor if possible
        clear_tensor_data(ctx.saved_tensors[0])  # input_

        return grad_input, [()], [(grad_extra_input,)]


class ScaledSwiGLU(_ScaledGLU):
    r"""SwiGLU with post-scaling (matches cuDNN grouped GEMM ``act_func="swiglu"``).

    If the GLU output has shape ``(d_1, ..., d_n)``, it is multiplied
    with an extra input tensor of shape ``(d_1, ..., d_{n-1})``.

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.
    activation_recompute_in_mlp : bool, default = ``False``
        Must be ``False``. Fused grouped-MLP activation recomputation is not
        implemented for SwiGLU.

    """

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        return tex.scaled_swiglu(
            input_,
            scales,
            None,
            int(self.glu_interleave_size or 0),
        )

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return tex.scaled_dswiglu(
            grad_output,
            input_,
            scales,
            None,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )


class ScaledSiTUGLU(_ScaledGLU):
    r"""SiTU-GLU with a row-wise post-activation scale.

    This is the native fallback and the pattern marker for cuDNN-frontend's
    block-scaled grouped-MLP SiTU-GLU fusion.

    Parameters
    ----------
    beta1 : float, default = 4.0
        Positive gate soft-cap parameter.
    beta2 : float, default = 25.0
        Positive up-branch soft-cap parameter.
    glu_interleave_size : int, optional
        Block-interleaved GLU layout, as in :class:`ScaledSwiGLU`.
    activation_recompute_in_mlp : bool, default = ``False``
        Must be ``False``. Fused grouped-MLP activation recomputation is not
        implemented for SiTU-GLU.
    """

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        beta1: float = 4.0,
        beta2: float = 25.0,
    ) -> None:
        super().__init__(
            glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
        )
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        if not math.isfinite(self.beta1) or self.beta1 <= 0.0:
            raise ValueError(f"beta1 must be finite and positive, got {self.beta1}")
        if not math.isfinite(self.beta2) or self.beta2 <= 0.0:
            raise ValueError(f"beta2 must be finite and positive, got {self.beta2}")

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        return tex.scaled_situglu(
            input_,
            scales,
            None,
            self.beta1,
            self.beta2,
            int(self.glu_interleave_size or 0),
        )

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return tex.scaled_dsituglu(
            grad_output,
            input_,
            scales,
            None,
            self.beta1,
            self.beta2,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )


class ScaledClampedQGeGLU(_ScaledGLU):
    r"""Clamped QGeGLU with post-scaling
    (matches cuDNN grouped GEMM ``act_func="geglu"``).

    Same layout and scaling contract as :class:`ScaledSwiGLU`, but the GLU
    uses :class:`ClampedSwiGLU` numerics (default ``limit`` / ``alpha`` match
    cuDNN).

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See :class:`ClampedSwiGLU`.
    activation_recompute_in_mlp : bool, default = ``False``
        Must be ``False``. Fused grouped-MLP activation recomputation is not
        implemented for clamped QGeGLU.
    limit : float, default ``7.0``
        Clamp limit (see :class:`ClampedSwiGLU`).
    alpha : float, default ``1.702``
        Sigmoid scale (see :class:`ClampedSwiGLU`).
    glu_linear_offset : float, default ``1.0``
        Offset added to the linear component after clamping
        (see :class:`ClampedSwiGLU`).

    """

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> None:
        super().__init__(
            glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
        )
        self._clamped: ClampedSwiGLU = ClampedSwiGLU(
            limit=limit,
            alpha=alpha,
            glu_linear_offset=glu_linear_offset,
        )

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        clamped = self._clamped
        return tex.scaled_clamped_swiglu(
            input_,
            scales,
            None,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
        )

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        clamped = self._clamped
        return tex.scaled_clamped_dswiglu(
            grad_output,
            input_,
            scales,
            None,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )
