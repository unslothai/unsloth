# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""LoRA on a GGUF checkpoint whose quantized Linears and MoE expert stacks stay in their GGUF blocks.

`from_pretrained(repo, gguf_file = ..., gguf_packed = True)` drives transformers' GGUF quantizer (>= 5.18)
with Unsloth's dequantize kernels instead of letting it unpack every weight. After the load:
- each quantized Linear is an `UnslothGGUFLinear`: frozen `weight_packed` uint8 blocks plus `ggml_type`
  and `shape`, dequantized per use; dX flows through the dequantized weight, LoRA A / B train;
- MoE expert stacks keep their blocks as `GgufExpertParam` (grouped matmul over dequantized chunks);
- embeddings and lm_head are dequantized to 16-bit and norms / routers stay full precision, as for NF4.
"""

import contextlib
import contextvars
import functools
import os
import threading
import types

import torch
from torch import nn

from ..kernels import gguf_reference as _ref

__all__ = [
    "GGUF_PACKED_KWARG",
    "UnslothGGUFLinear",
    "GgufPackedState",
    "gguf_ops",
    "gguf_packed_requested",
    "gguf_packed_supported",
    "gguf_packed_quantization_config",
    "check_gguf_packed_request",
    "gguf_packed_load",
    "convert_gguf_packed_model",
    "is_gguf_packed_model",
    "densify_gguf_packed_model",
]

GGUF_PACKED_KWARG = "gguf_packed"
# The release whose GGUF quantizer can keep blocks packed for every architecture it maps, MoE included.
GGUF_PACKED_MIN_TRANSFORMERS = "5.18.0"
_REQUIRED_ZOO_OPS = ("dequantize_gguf", "gguf_linear", "gguf_linear_backward_input")


# ----------------------------------------------------------------------------------------------------------
# Kernel shim: the pure torch reference, or unsloth_zoo.gguf_kernels when asked for and it ships the ops.
# ----------------------------------------------------------------------------------------------------------


def _reference_linear(
    x,
    packed,
    ggml_type,
    shape,
    bias = None,
):
    weight = _ref.dequantize_gguf(packed, ggml_type, shape, x.dtype)
    return torch.nn.functional.linear(x, weight, None if bias is None else bias.to(x.dtype))


def _reference_linear_backward_input(grad_output, packed, ggml_type, shape):
    return torch.matmul(
        grad_output, _ref.dequantize_gguf(packed, ggml_type, shape, grad_output.dtype)
    )


_OPS = []


def gguf_ops():
    """Namespace of (dequantize, linear, linear_backward_input, grouped_mm, supports, name), resolved once. A plain
    global read, so compiled code traces straight into the ops. `UNSLOTH_GGUF_KERNELS`: unset / "auto" picks
    unsloth_zoo.gguf_kernels when that zoo launches its dequantize kernel from compiled graphs
    (`COMPILE_TRACED_DEQUANTIZE`), else the pure torch reference; "zoo" forces the zoo kernels, "reference" the
    reference. On a 30-step Qwen3.5-4B Q4_K_M LoRA run (RTX PRO 6000) the zoo kernels train at 1.06x the
    dequantized model's step time, the reference at 1.17x; MoE expert stacks gain far more (grouped GEMM)."""
    if not _OPS:
        _OPS.append(_resolve_ops())
    return _OPS[0]


def _resolve_ops():
    reference = types.SimpleNamespace(
        name = "reference",
        dequantize = _ref.dequantize_gguf,
        linear = _reference_linear,
        linear_backward_input = _reference_linear_backward_input,
        # None: expert stacks take the zoo's chunked dequantize + grouped GEMM.
        grouped_mm = None,
        supports = _ref.is_supported,
    )
    choice = os.environ.get("UNSLOTH_GGUF_KERNELS", "auto").strip().lower()
    if choice not in ("auto", "zoo", ""):
        return reference
    try:
        import unsloth_zoo.gguf_kernels as zoo_kernels
    except Exception:
        return reference
    if not all(callable(getattr(zoo_kernels, name, None)) for name in _REQUIRED_ZOO_OPS):
        return reference
    if choice != "zoo" and not getattr(zoo_kernels, "COMPILE_TRACED_DEQUANTIZE", False):
        # An older zoo runs every decode as a Python custom op: slower than the reference in training.
        return reference
    zoo_types = getattr(zoo_kernels, "SUPPORTED_TYPES", None)
    zoo_types = None if zoo_types is None else frozenset(int(t) for t in zoo_types)

    forced = choice == "zoo"

    def _zoo_has(ggml_type, packed):
        # Types the zoo kernels do not decode stay on the reference, and by default so do blocks off CUDA: there the
        # zoo runs the same torch decode inside an opaque custom op, which torch.compile cannot fuse (far slower).
        return (forced or packed.is_cuda) and (zoo_types is None or int(ggml_type) in zoo_types)

    def dequantize(
        packed,
        ggml_type,
        shape,
        dtype = torch.float32,
    ):
        if _zoo_has(ggml_type, packed):
            return zoo_kernels.dequantize_gguf(packed, ggml_type, tuple(shape), dtype)
        return _ref.dequantize_gguf(packed, ggml_type, shape, dtype)

    def linear(
        x,
        packed,
        ggml_type,
        shape,
        bias = None,
    ):
        if _zoo_has(ggml_type, packed):
            # The bias in x's dtype, as the reference: the zoo's dequantize + F.linear path rejects a mixed bias.
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return zoo_kernels.gguf_linear(x, packed, ggml_type, tuple(shape), bias)
        return _reference_linear(x, packed, ggml_type, shape, bias)

    def linear_backward_input(grad_output, packed, ggml_type, shape):
        if _zoo_has(ggml_type, packed):
            return zoo_kernels.gguf_linear_backward_input(
                grad_output, packed, ggml_type, tuple(shape)
            )
        return _reference_linear_backward_input(grad_output, packed, ggml_type, shape)

    zoo_grouped = getattr(zoo_kernels, "gguf_grouped_mm", None)

    def grouped_mm(x, packed, ggml_type, shape, counts, transpose_b):
        # Rows sorted by expert, counts[e] each; transpose_b: x @ W[e]^T (forward), else x @ W[e] (dX). None declines.
        if zoo_grouped is None or not _zoo_has(ggml_type, packed):
            return None
        return zoo_grouped(
            x,
            packed,
            ggml_type,
            tuple(shape),
            counts.to(torch.int32),
            transpose_b = bool(transpose_b),
        )

    return types.SimpleNamespace(
        name = "unsloth_zoo.gguf_kernels",
        dequantize = dequantize,
        linear = linear,
        linear_backward_input = linear_backward_input,
        grouped_mm = grouped_mm,
        supports = lambda t: zoo_types is None or int(t) in zoo_types or _ref.is_supported(t),
    )


def _compute_dtype(x):
    device_type = x.device.type
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except Exception:
        pass
    return x.dtype


def _is_transposed(W):
    # W is weight_packed (out, row_bytes) or its .t() view; only the view has a unit leading stride.
    return W.dim() == 2 and W.stride(0) == 1 and W.stride(1) != 1


# ----------------------------------------------------------------------------------------------------------
# Linear
# ----------------------------------------------------------------------------------------------------------


def _effective_weight(packed, ggml_type, shape, dtype, inverse):
    weight = gguf_ops().dequantize(packed, ggml_type, shape, dtype)
    return weight if inverse is None else weight.index_select(1, inverse)


# True while a gradient-checkpoint backward re-runs a layer's forward (mirrored from unsloth_zoo's recompute
# marker, see `_mirror_checkpoint_recompute`): that forward's backward runs right after it, so a Linear can keep
# the weight it decoded for the forward until then (one layer's 16-bit weights at a time) instead of decoding it
# a third time for dX. A plain global, so a compiled layer guards on it.
_IN_CHECKPOINT_RECOMPUTE = [False]
# Recomputes in flight, every thread together: autograd runs one device's backward per thread, so two checkpoint
# recomputes can overlap without nesting, and a saved previous value restored on exit could leave the flag on.
_RECOMPUTE_DEPTH = [0]
_RECOMPUTE_LOCK = threading.Lock()


# UNSLOTH_GGUF_RECOMPUTE_REUSE=0 decodes for dX again (the memory of one layer's 16-bit weights back, one more
# decode per Linear per step).
_RECOMPUTE_REUSE = os.environ.get("UNSLOTH_GGUF_RECOMPUTE_REUSE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _reuse_recompute_weight():
    return _RECOMPUTE_REUSE and _IN_CHECKPOINT_RECOMPUTE[0]


def _mirror_checkpoint_recompute():
    """Wrap unsloth_zoo's gradient-checkpoint recompute marker so `_IN_CHECKPOINT_RECOMPUTE` follows it."""
    try:
        from unsloth_zoo import gradient_checkpointing as zoo_gc
    except Exception:
        return False
    original = getattr(zoo_gc, "_gradient_checkpoint_recompute_marker", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_gguf_mirror", False):
        return True

    @contextlib.contextmanager
    def marker(*args, **kwargs):
        with _RECOMPUTE_LOCK:
            _RECOMPUTE_DEPTH[0] += 1
            _IN_CHECKPOINT_RECOMPUTE[0] = True
        try:
            with original(*args, **kwargs):
                yield
        finally:
            with _RECOMPUTE_LOCK:
                _RECOMPUTE_DEPTH[0] -= 1
                _IN_CHECKPOINT_RECOMPUTE[0] = _RECOMPUTE_DEPTH[0] > 0

    marker._unsloth_gguf_mirror = True
    marker.__wrapped__ = original
    zoo_gc._gradient_checkpoint_recompute_marker = marker
    return True


class _GGUFLinearFunction(torch.autograd.Function):
    """Frozen packed base: forward and dX only. Nothing 16-bit is saved for backward and dX decodes again, except
    in a gradient-checkpoint recompute (`keep_weight`), whose backward follows at once: the decoded weight is
    saved for it. `inverse`: argsort of an input permutation, applied to the dequantized columns (bitwise the
    dense model's product)."""

    @staticmethod
    def forward(
        ctx,
        x,
        packed,
        bias,
        ggml_type,
        shape,
        inverse,
        keep_weight = False,
    ):
        ctx.ggml_type, ctx.shape = ggml_type, shape
        with torch.no_grad():
            if inverse is None and not keep_weight:
                ctx.save_for_backward(packed, inverse, None)
                return gguf_ops().linear(x, packed, ggml_type, shape, bias)
            weight = _effective_weight(packed, ggml_type, shape, x.dtype, inverse)
            ctx.save_for_backward(packed, inverse, weight if keep_weight else None)
            return torch.nn.functional.linear(x, weight, None if bias is None else bias.to(x.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        packed, inverse, weight = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                if weight is not None and weight.dtype == grad_output.dtype:
                    grad_x = torch.matmul(grad_output, weight)
                elif inverse is None:
                    grad_x = gguf_ops().linear_backward_input(
                        grad_output, packed, ctx.ggml_type, ctx.shape
                    )
                else:
                    weight = _effective_weight(
                        packed, ctx.ggml_type, ctx.shape, grad_output.dtype, inverse
                    )
                    grad_x = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, None, grad_bias, None, None, None, None


class UnslothGGUFLinear(nn.Linear):
    """A Linear whose frozen weight stays as its GGUF blocks: `weight_packed` uint8 `(out, row_bytes)`, `ggml_type`,
    `shape = (out, in)`. `input_permutation` (transformers' `PermuteInputFeatures`, Qwen3.5 `out_proj`): the stored
    columns are the model's columns reordered, so the effective weight is `blocks[:, argsort(perm)]`."""

    # The packed-base protocol of kernels.utils / fast_lora (named for its first user): `weight_packed` plus a quant
    # state from `mxfp4_quant_state()`, so the fused LoRA kernels never keep a 16-bit copy of the weight.
    _unsloth_mxfp4_packed_linear = True
    _unsloth_gguf_packed_linear = True
    compute_dtype = torch.bfloat16

    @classmethod
    def from_packed(
        cls,
        packed,
        ggml_type,
        in_features,
        out_features,
        bias = None,
        input_permutation = None,
        compute_dtype = None,
    ):
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = int(in_features)
        module.out_features = int(out_features)
        module.ggml_type = ggml_type
        expected = (module.out_features, _ref.row_bytes(module.ggml_type, module.in_features))
        if packed.dtype != torch.uint8 or tuple(packed.shape) != expected:
            raise ValueError(
                f"Unsloth: GGUF {_ref.GGML_TYPE_NAMES.get(ggml_type, ggml_type)} blocks for a "
                f"{module.out_features} x {module.in_features} Linear must be uint8 {expected}, got "
                f"{packed.dtype} {tuple(packed.shape)}."
            )
        module.weight_packed = nn.Parameter(packed, requires_grad = False)
        if bias is None:
            module.register_parameter("bias", None)
        else:
            module.bias = (
                bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias, requires_grad = False)
            )
        module._set_input_permutation(input_permutation)
        if compute_dtype is not None and compute_dtype.is_floating_point:
            module.compute_dtype = compute_dtype
        gguf_ops()  # resolved before any trace reaches forward
        _mirror_checkpoint_recompute()
        return module

    def _set_input_permutation(self, permutation):
        self.register_buffer("input_permutation", permutation, persistent = False)
        inverse = None if permutation is None else torch.argsort(permutation)
        self.register_buffer("input_inverse", inverse, persistent = False)

    # Layers of one decoder class mix ggml types and sizes (Q4_K_M: Q6_K on some down_proj). torch.compile guards an
    # int module attribute as a constant, then on the next layer turns it into a SymInt: the zoo kernels then break the
    # graph (and the reference loses its per-type code). A str attribute is always specialized, and the shape is
    # read off the (static) packed parameter, so every layer compiles to constants.
    @property
    def ggml_type(self):
        return int(self._ggml_type_key)

    @ggml_type.setter
    def ggml_type(self, value):
        self._ggml_type_key = str(int(value))

    @property
    def shape(self):
        packed = self._parameters["weight_packed"]
        block_elems, block_bytes = _ref.GGML_BLOCK[self.ggml_type]
        return (packed.shape[0], packed.shape[1] // block_bytes * block_elems)

    @property
    def ggml_type_name(self):
        return _ref.GGML_TYPE_NAMES.get(self.ggml_type, str(self.ggml_type))

    def _inverse_permutation(self):
        return self.input_inverse

    def dequantize_weight(self, dtype = None):
        return _effective_weight(
            self.weight_packed,
            self.ggml_type,
            self.shape,
            dtype or self.compute_dtype,
            self._inverse_permutation(),
        )

    @property
    def weight(self):
        # A fresh decode per read, never kept: writes are lost (merges densify first).
        return self.dequantize_weight()

    def forward(self, x):
        bias = self.bias
        dtype = _compute_dtype(x)
        if x.dtype != dtype:
            x = x.to(dtype)
        inverse = self._inverse_permutation()
        if torch.is_grad_enabled() and (
            x.requires_grad or (bias is not None and bias.requires_grad)
        ):
            return _GGUFLinearFunction.apply(
                x,
                self.weight_packed,
                bias,
                self.ggml_type,
                self.shape,
                inverse,
                _reuse_recompute_weight(),
            )
        if inverse is None:
            return gguf_ops().linear(x, self.weight_packed, self.ggml_type, self.shape, bias)
        # Columns cross blocks: the dense effective weight, so the product is bitwise the dense model's.
        weight = _effective_weight(self.weight_packed, self.ggml_type, self.shape, dtype, inverse)
        return torch.nn.functional.linear(x, weight, None if bias is None else bias.to(dtype))

    def mxfp4_quant_state(self):
        """The quant state kernels.utils pairs with `weight_packed` (see `_packed_base`)."""
        state = self.__dict__.get("_unsloth_packed_state")
        packed = self._parameters["weight_packed"]
        if (
            state is None
            or state.packed is not packed
            or state.dtype != self.compute_dtype
            or state.input_permutation is not self.input_permutation
        ):
            state = self.__dict__["_unsloth_packed_state"] = GgufPackedState(self)
        return state

    def reset_parameters(self):
        return

    def _apply(self, fn, *args, **kwargs):
        try:
            dtype = fn(torch.empty(0, dtype = self.compute_dtype)).dtype
        except Exception:
            dtype = self.compute_dtype
        result = super()._apply(fn, *args, **kwargs)
        if dtype.is_floating_point:
            self.compute_dtype = dtype
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ggml_type={self.ggml_type_name}, "
            f"permuted_input={self.input_permutation is not None}"
        )


class GgufPackedState:
    """Quant state for a packed GGUF base in Unsloth's LoRA kernels: `W` is `weight_packed` (or its `.t()`)."""

    _unsloth_packed_weight_state = True

    def __init__(self, module):
        self.module = module
        self.packed = module._parameters["weight_packed"]
        self.ggml_type = module.ggml_type
        self.shape = module.shape
        self.dtype = module.compute_dtype
        self.input_permutation = module.input_permutation

    def dequantize(
        self,
        W = None,
        dtype = None,
    ):
        weight = self.module.dequantize_weight(dtype or self.dtype)
        return weight.t() if W is not None and _is_transposed(W) else weight

    def matmul(
        self,
        X,
        W,
        out = None,
    ):
        """`X @ D(W).t()`, D(W) being W's 16-bit value, as matmul_lora computes it."""
        transposed = W is not None and _is_transposed(W)
        if self.input_permutation is not None:
            weight = self.module.dequantize_weight(X.dtype)
            y = torch.matmul(X, weight if transposed else weight.t())
        elif transposed:
            y = gguf_ops().linear_backward_input(X, self.packed, self.ggml_type, self.shape)
        else:
            y = gguf_ops().linear(X, self.packed, self.ggml_type, self.shape, None)
        if out is not None and out.shape == y.shape:
            return out.copy_(y)
        return y


# ----------------------------------------------------------------------------------------------------------
# MoE expert stacks
# ----------------------------------------------------------------------------------------------------------


def _packed_expert_base():
    try:
        from unsloth_zoo import packed_experts
    except Exception:
        return None, None
    base = getattr(packed_experts, "_PackedExpertParam", None)
    # The zoo routes any packed stack that brings its own grouped linear (zoo >= the GGUF hook) to it.
    if base is None or not getattr(packed_experts, "PACKED_EXPERT_OWN_GROUPED_LINEAR", False):
        return None, None
    return packed_experts, base


@functools.lru_cache(maxsize = None)
def gguf_expert_param_class():
    """`GgufExpertParam`, or None when this unsloth_zoo cannot run a packed GGUF expert stack."""
    packed_experts, base = _packed_expert_base()
    if base is None:
        return None

    class GgufExpertParam(base):
        """Frozen GGUF expert stack: uint8 `(E, N, row_bytes)` blocks of one ggml type, rows = output features."""

        packed_format = "gguf"

        def __new__(
            cls,
            data,
            requires_grad = False,
            ggml_type = None,
            packed_cols = None,
            **kwargs,
        ):
            if isinstance(data, GgufExpertParam):
                ggml_type = data.ggml_type if ggml_type is None else ggml_type
                packed_cols = data.packed_cols if packed_cols is None else packed_cols
                kwargs = {**data._init_kwargs(), **kwargs}
                data = data.data
            if ggml_type is None or packed_cols is None:
                raise ValueError(
                    "Unsloth: a GGUF expert stack needs its ggml_type and column count"
                )
            if data.dtype != torch.uint8 or data.dim() != 3:
                raise ValueError("Unsloth: a GGUF expert stack must be uint8 (E, N, row_bytes)")
            E, N, nbytes = data.shape
            if nbytes != _ref.row_bytes(int(ggml_type), int(packed_cols)):
                raise ValueError(
                    f"Unsloth: GGUF {_ref.GGML_TYPE_NAMES.get(int(ggml_type), ggml_type)} expert rows of "
                    f"{packed_cols} columns take {_ref.row_bytes(int(ggml_type), int(packed_cols))} bytes, got {nbytes}"
                )
            param = torch.Tensor._make_subclass(cls, data, False)
            param.ggml_type = int(ggml_type)
            param.packed_cols = int(packed_cols)
            # The base class's companions: GGUF blocks carry their own block scales; `packed_scales` holds the
            # file's optional per-expert weight scale (llama.cpp's `<name>.scale`, float32 (E,)), else is empty.
            scales = kwargs.get("packed_scales")
            if scales is None or scales.numel() == 0:
                scales = torch.empty((0,), dtype = torch.float32, device = data.device)
            elif scales.numel() != E:
                raise ValueError(
                    f"Unsloth: a GGUF expert stack of {E} experts takes {E} weight scales, got {scales.numel()}"
                )
            param.packed_scales = scales.reshape(-1).to(torch.float32)
            param.packed_global = None
            param.packed_transposed = False
            param.packed_dtype = kwargs.get("packed_dtype", torch.bfloat16)
            param.packed_global_mode = "divide"
            param._original_shape = torch.Size((E, N, int(packed_cols)))
            return param

        def packed_dims(self):
            E, N, _ = self.data.shape
            return E, N, self.packed_cols

        def packed_grouped_mm(self, x, counts, transpose_b):
            """The zoo's hook for a stack's own grouped GEMM (frozen base, forward and dX); None declines."""
            grouped = gguf_ops().grouped_mm
            if grouped is None:
                return None
            E, N, _ = self.data.shape
            blocks = self.as_subclass(torch.Tensor)
            out = grouped(x, blocks, self.ggml_type, (E, N, self.packed_cols), counts, transpose_b)
            scales, _ = self._companions()
            if out is None or scales.numel() == 0:
                return out
            # x @ (s_e W_e) = s_e (x @ W_e), in both directions; rows are sorted by expert.
            rows = torch.repeat_interleave(scales, counts.to(scales.device), output_size = x.shape[0])
            return (out.float() * rows[:, None]).to(out.dtype)

        def _init_kwargs(self):
            kwargs = super()._init_kwargs()
            kwargs.update(ggml_type = self.ggml_type, packed_cols = self.packed_cols)
            return kwargs

        def _decode(
            self,
            codes,
            scales,
            glob,
            dtype,
            transpose,
            token_counts,
            out = None,
        ):
            E, N, _ = codes.shape
            if scales is not None and scales.numel():
                # The per-expert weight scale, applied in float32 and rounded once.
                dense = gguf_ops().dequantize(
                    codes, self.ggml_type, (E, N, self.packed_cols), torch.float32
                )
                dense = (dense * scales.to(dense.device).view(-1, 1, 1)).to(dtype)
            else:
                dense = gguf_ops().dequantize(
                    codes, self.ggml_type, (E, N, self.packed_cols), dtype
                )
            if transpose:
                dense = dense.transpose(-2, -1).contiguous()
            if out is not None:
                return out.copy_(dense)
            return dense

        def nbytes_packed(self):
            return self.data.numel()

        def __repr__(self):
            return (
                f"GgufExpertParam({_ref.GGML_TYPE_NAMES.get(self.ggml_type, self.ggml_type)}, "
                f"logical_shape={tuple(self._original_shape)}, device={self.device})"
            )

    return GgufExpertParam


# ----------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------


def gguf_packed_requested(kwargs):
    """Pop `gguf_packed` from from_pretrained kwargs; True when a packed GGUF load is asked for."""
    value = kwargs.pop(GGUF_PACKED_KWARG, None)
    if value is None:
        value = os.environ.get("UNSLOTH_GGUF_PACKED", "0").strip().lower() in ("1", "true", "yes")
    return bool(value) and bool(kwargs.get("gguf_file"))


def gguf_packed_supported():
    """None if this transformers can keep GGUF blocks packed, else the reason it cannot."""
    try:
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__.split("+")[0]) < Version(GGUF_PACKED_MIN_TRANSFORMERS):
            return f"transformers {transformers.__version__} is older than {GGUF_PACKED_MIN_TRANSFORMERS}"
        import transformers.integrations.gguf.utils  # noqa: F401
        import transformers.quantizers.quantizer_gguf  # noqa: F401
    except Exception as error:
        return f"transformers has no GGUF quantizer ({error})"
    return None


def gguf_packed_quantization_config():
    from transformers import GgufConfig

    config = GgufConfig(dequantize = False)
    config._unsloth_gguf_packed = True
    return config


def check_gguf_packed_request(
    kwargs,
    full_finetuning = False,
    fast_inference = False,
    load_in_fp8 = False,
    qat_scheme = None,
):
    """Validate a `gguf_packed` request at the loader's entry point; returns whether it is on. The flag stays in
    kwargs until `gguf_dequantize_kwargs` turns it into the quantization config."""
    if not kwargs.get(GGUF_PACKED_KWARG):
        kwargs.pop(GGUF_PACKED_KWARG, None)
        return False
    if not kwargs.get("gguf_file"):
        raise ValueError(
            "Unsloth: `gguf_packed = True` keeps a GGUF file's blocks packed; pass the file as `gguf_file = ...`."
        )
    if kwargs.get("quantization_config") is not None:
        raise ValueError(
            "Unsloth: `gguf_packed = True` cannot be combined with a `quantization_config`."
        )
    refused = [
        name
        for name, on in (
            ("full_finetuning", full_finetuning),
            ("fast_inference", fast_inference),
            ("load_in_fp8", load_in_fp8),
            ("qat_scheme", qat_scheme is not None),
        )
        if on
    ]
    if refused:
        raise NotImplementedError(
            f"Unsloth: `gguf_packed = True` trains LoRA on a frozen packed base; it does not support "
            f"{', '.join(refused)}. Load without `gguf_packed` to dequantize the GGUF to 16-bit instead."
        )
    reason = gguf_packed_supported()
    if reason is not None:
        raise RuntimeError(
            f"Unsloth: `gguf_packed = True` needs transformers >= {GGUF_PACKED_MIN_TRANSFORMERS} ({reason}). "
            "Upgrade transformers, or load without `gguf_packed` to dequantize the GGUF to 16-bit."
        )
    return True


# (model name, gguf_file) of the packed load in progress: its configs come from the GGUF itself.
_GGUF_CONFIG_SOURCE = contextvars.ContextVar("_unsloth_gguf_packed_config_source", default = None)
_AUTOCONFIG_HOOKED = []


def _install_autoconfig_hook():
    """Within a packed load, `AutoConfig.from_pretrained(<that repo>)` reads the GGUF's metadata. A GGUF repo may
    ship the source model's config.json (Qwen3.8 ships its vision-language one), whose module names the GGUF's
    tensor map does not produce."""
    if _AUTOCONFIG_HOOKED:
        return
    from transformers import AutoConfig

    original = AutoConfig.from_pretrained

    @functools.wraps(original)
    def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        source = _GGUF_CONFIG_SOURCE.get()
        if (
            source is not None
            and kwargs.get("gguf_file") is None
            and isinstance(pretrained_model_name_or_path, (str, os.PathLike))
            and os.fspath(pretrained_model_name_or_path) == source[0]
        ):
            kwargs["gguf_file"] = source[1]
        return original(pretrained_model_name_or_path, *args, **kwargs)

    AutoConfig.from_pretrained = from_pretrained
    _AUTOCONFIG_HOOKED.append(original)


def gguf_packed_config_scope(fn):
    """Decorator for the loaders' `from_pretrained`: a `gguf_packed` load takes its config from the GGUF."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        model_name = kwargs.get("model_name", args[0] if args else None)
        if not (kwargs.get(GGUF_PACKED_KWARG) and kwargs.get("gguf_file")) or not isinstance(
            model_name, (str, os.PathLike)
        ):
            return fn(*args, **kwargs)
        _install_autoconfig_hook()
        token = _GGUF_CONFIG_SOURCE.set((os.fspath(model_name), kwargs["gguf_file"]))
        try:
            with _extra_gguf_config_archs():
                return fn(*args, **kwargs)
        finally:
            _GGUF_CONFIG_SOURCE.reset(token)

    return wrapper


@contextlib.contextmanager
def _extra_gguf_config_archs():
    """Within: transformers' GGUF config reader also knows the architectures in
    `gguf_packed_archs.EXTRA_GGUF_CONFIG_ARCHS` (Inkling 2), which it has no config mapping for."""
    try:
        import transformers.integrations.gguf.gguf_config_mapping as config_module
        from .gguf_packed_archs import EXTRA_GGUF_CONFIG_ARCHS
        table = config_module.GGUF_CONFIG_ARCHS
    except Exception:
        # transformers without the fast GGUF reader: the request is refused with its own message later.
        yield
        return
    added = [arch for arch in EXTRA_GGUF_CONFIG_ARCHS if arch not in table]
    for arch in added:
        table[arch] = EXTRA_GGUF_CONFIG_ARCHS[arch]
    try:
        yield
    finally:
        for arch in added:
            table.pop(arch, None)


# ----------------------------------------------------------------------------------------------------------
# Split GGUFs and per-tensor / per-expert weight scales
# ----------------------------------------------------------------------------------------------------------

_SPLIT_RE = None


def gguf_split_paths(path):
    """Every shard of a llama.cpp split GGUF (`name-00001-of-00009.gguf`), in order; `[path]` for a single file."""
    global _SPLIT_RE
    import re

    if _SPLIT_RE is None:
        _SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
    match = _SPLIT_RE.match(os.fspath(path))
    if match is None:
        return [os.fspath(path)]
    stem, count = match.group(1), int(match.group(3))
    paths = [f"{stem}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Unsloth: the split GGUF {os.path.basename(os.fspath(path))} needs all {count} shards next to it; "
            f"missing {', '.join(os.path.basename(p) for p in missing[:3])}."
        )
    return paths


def _merged_header(path):
    """One `GgufHeader` over every shard of a split file (tensor order kept), plus {tensor name: its shard's
    header}. Offsets stay relative to each shard; `_split_state_dict` reads them from the right file."""
    from transformers.integrations.gguf.reader import GgufHeader

    headers = [GgufHeader.from_file(p) for p in gguf_split_paths(path)]
    owner = {info.name: header for header in headers for info in header.tensors}
    first = headers[0]
    if len(headers) == 1:
        return first, owner
    tensors = tuple(info for header in headers for info in header.tensors)
    return first._replace(tensors = tensors), owner


def _split_state_dict(owners, header):
    """Every tensor of `header` (the merged, split-aware header), each read from its own shard's file."""
    from transformers.integrations.gguf.reader import load_gguf_state_dict

    by_shard = {}
    for info in header.tensors:
        shard = owners[info.name]
        by_shard.setdefault(id(shard), (shard, []))[1].append(info)
    state_dict = {}
    for shard, infos in by_shard.values():
        state_dict.update(load_gguf_state_dict(shard._replace(tensors = tuple(infos))))
    return state_dict


def _weight_scale_tensors(header):
    """{weight tensor name: its `.scale` TensorInfo} for llama.cpp's optional per-tensor ({1}, 2-D weight) or
    per-expert ({n_expert}, 3-D stack) F32 scale: llama.cpp multiplies the weight's matmul output by it
    (`build_lora_mm` / `build_lora_mm_id`), so the effective weight is `scale * W` (per expert for a stack)."""
    by_name = {info.name: info for info in header.tensors}
    scales = {}
    for name, info in by_name.items():
        if not name.endswith(".scale"):
            continue
        weight = by_name.get(name[: -len(".scale")] + ".weight")
        if weight is None:
            continue
        experts = weight.shape[0] if len(weight.shape) == 3 else 1
        numel = 1
        for dim in info.shape:
            numel *= dim
        if info.ggml_type != 0 or numel != experts or len(weight.shape) not in (2, 3):
            continue
        scales[weight.name] = info
    return scales


def _read_scale(header, info):
    import numpy as np
    data = np.memmap(
        header.path,
        dtype = np.float32,
        mode = "r",
        offset = header.data_start + info.offset,
        shape = (info.nbytes // 4,),
    )
    return torch.from_numpy(np.array(data, dtype = np.float32))


def _apply_weight_scales(quantizer, model):
    """Fold the file's `.scale` tensors into the weights they scale: a packed GGUF expert stack keeps the per-expert
    scale beside its blocks (applied in float32 on every decode); a dense parameter is multiplied in float32."""
    scales = getattr(quantizer, "_unsloth_weight_scales", None)
    if not scales:
        return 0
    import copy

    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    mapping = copy.deepcopy(quantizer.mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    owners = quantizer._unsloth_tensor_owner
    targets = {}
    for weight_name, info in scales.items():
        param_name, pattern = rename_source_key(weight_name, renamings, converters)
        targets.setdefault(param_name, []).append((weight_name, info, pattern))
    applied = 0
    for param_name, sources in targets.items():
        converter = next((c for c in converters if sources[0][2] in c.source_patterns), None)
        if len(sources) != 1 or (converter is not None and len(converter.source_patterns) != 1):
            raise NotImplementedError(
                f"Unsloth: `gguf_packed = True` cannot apply the GGUF weight scales of {param_name}, which is built "
                "from several tensors. Load without `gguf_packed` instead."
            )
        weight_name, info, _ = sources[0]
        scale = _read_scale(owners[info.name], info)
        module_name, _, leaf = param_name.rpartition(".")
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue
        if (
            leaf == "weight"
            and hasattr(module, "ggml_type")
            and module._parameters.get("weight") is not None
            and (module._parameters["weight"].dtype == torch.uint8)
        ):
            # A packed 2-D GGUF Linear / Embedding (transformers' GgufLinear / GgufEmbedding): made dense below with
            # the scale folded in, since no Inkling or Qwen file ships one and the packed linear kernels carry none.
            module._unsloth_gguf_weight_scale_pending = scale
            applied += 1
            continue
        param = module._parameters.get(leaf)
        if param is None:
            continue
        if param.dtype == torch.uint8 and param.dim() == 3:
            # A packed expert placeholder: GgufExpertParam takes the scale from here.
            module.__dict__.setdefault("_unsloth_gguf_expert_scales", {})[leaf] = scale
        else:
            view = (
                scale.view(-1, *([1] * (param.dim() - 1)))
                if scale.numel() > 1
                else scale.reshape(())
            )
            with torch.no_grad():
                param.data = (param.data.float() * view.to(param.device)).to(param.dtype)
        applied += 1
    return applied


def _keep_float32_casts(conversions, model, dtype):
    """transformers' GGUF loader casts every tensor to the load dtype last (`Cast`), which ignores the model's
    `_keep_in_fp32_modules_strict` (Inkling 2's router, residual queries, short convolutions, message gates are
    float32 modules and compute so). Keep those tensors float32, as transformers' own checkpoint load does."""
    import transformers.integrations.gguf.gguf_conversion_mapping as mapping_module

    cast_cls = getattr(mapping_module, "Cast", None)
    patterns = set()
    if dtype in (torch.float16, torch.bfloat16):
        patterns |= set(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    if dtype == torch.float16:
        patterns |= set(getattr(model, "_keep_in_fp32_modules", None) or ())
    if cast_cls is None or not patterns:
        return conversions
    import re

    keep = re.compile("|".join(f"(?:{p.replace('*', '.*')})" for p in sorted(patterns)))

    class _CastKeepFloat32(cast_cls):
        @torch.no_grad
        def convert(
            self,
            input_dict,
            source_patterns,
            target_patterns,
            full_layer_name = None,
            **kwargs,
        ):
            name = full_layer_name if full_layer_name is not None else target_patterns[0]
            tensor = next(iter(input_dict.values()))
            tensor = tensor[0] if isinstance(tensor, list) else tensor
            if tensor.dtype != torch.uint8 and tensor.is_floating_point() and keep.search(name):
                return {name: tensor.to(torch.float32)}
            return super().convert(
                input_dict,
                source_patterns,
                target_patterns,
                full_layer_name = full_layer_name,
                **kwargs,
            )

    for conversion in conversions:
        operations = getattr(conversion, "operations", None)
        if not operations:
            continue
        for i, op in enumerate(operations):
            if type(op) is cast_cls:
                keep_op = _CastKeepFloat32(op.dtype)
                operations[i] = keep_op
    return conversions


class _UnslothGgufKernel:
    """What transformers' GGUF quantizer asks of a kernel at load: which ggml types may stay packed."""

    def supports(self, ggml_type):
        return (
            int(ggml_type) in _ref.GGML_BLOCK
            and int(ggml_type) not in _UNPACKED_TYPES
            and (gguf_ops().supports(int(ggml_type)))
        )


# Value types: a GGUF stores these as plain numbers, never as blocks.
_UNPACKED_TYPES = {0, 1, 30}


def _gguf_architecture(path):
    from transformers.integrations.gguf.reader import read_gguf_metadata
    metadata, _ = read_gguf_metadata(path)
    return metadata.get("general.architecture")


# Tensors newer llama.cpp converters (and so `save_pretrained_gguf`) write as one: `[first | second]` stacked on rows,
# each half exactly what the unfused file stores under its own name. Leaf name -> (first, second, rows of first from
# the same block's tensors).
def _rows_of_attn_qkv(block, fused_rows):
    # attn_gate is in_proj_z: one row per value element, which ssm_out (out_proj) consumes as its columns.
    out_proj = block.get("ssm_out.weight")
    return None if out_proj is None else fused_rows - out_proj.shape[-1]


def _rows_of_ssm_beta(block, fused_rows):
    # beta and alpha each carry one row per value head.
    return fused_rows // 2 if fused_rows % 2 == 0 else None


_FUSED_ROW_SPLITS = {
    "attn_qkv_gate.weight": ("attn_qkv.weight", "attn_gate.weight", _rows_of_attn_qkv),
    "ssm_beta_alpha.weight": ("ssm_beta.weight", "ssm_alpha.weight", _rows_of_ssm_beta),
}


def _block_leaf(name):
    """`blk.3.attn_qkv_gate.weight` -> ("3", "attn_qkv_gate.weight"); ("", "") outside the blocks."""
    if not name.startswith("blk."):
        return "", ""
    index, _, leaf = name[len("blk.") :].partition(".")
    return index, leaf


def _fused_source_name(name):
    """`blk.3.attn_qkv.weight` -> `blk.3.attn_qkv_gate.weight` (the fused tensor a split half came from)."""
    index, leaf = _block_leaf(name)
    for fused, (first, second, _) in _FUSED_ROW_SPLITS.items():
        if leaf in (first, second):
            return f"blk.{index}.{fused}"
    raise KeyError(name)


def _split_fused_gguf_tensors(header):
    """The header with each fused tensor listed as its two halves: row slices of the same bytes (GGUF blocks never
    span rows), so the tensor map, the packed plan and the loaded bytes are exactly an unfused file's."""
    if header is None or not any(
        _block_leaf(info.name)[1] in _FUSED_ROW_SPLITS for info in header.tensors
    ):
        return header
    blocks = {}
    for info in header.tensors:
        index, leaf = _block_leaf(info.name)
        if leaf:
            blocks.setdefault(index, {})[leaf] = info
    tensors = []
    for info in header.tensors:
        index, leaf = _block_leaf(info.name)
        split = _FUSED_ROW_SPLITS.get(leaf)
        if split is None:
            tensors.append(info)
            continue
        first, second, first_rows_of = split
        block, rows = blocks[index], int(info.shape[0])
        first_rows = first_rows_of(block, rows) if len(info.shape) == 2 else None
        if (
            first_rows is None
            or not 0 < first_rows < rows
            or info.nbytes % rows
            or first in block
            or second in block
        ):
            raise NotImplementedError(
                f"Unsloth: `gguf_packed = True` cannot split the fused GGUF tensor {info.name} "
                f"{tuple(info.shape)}. Load without `gguf_packed` to use transformers' dense loader."
            )
        row_nbytes = info.nbytes // rows
        for name, start, count in ((first, 0, first_rows), (second, first_rows, rows - first_rows)):
            tensors.append(
                info._replace(
                    name = f"blk.{index}.{name}",
                    shape = (count, *info.shape[1:]),
                    offset = info.offset + start * row_nbytes,
                    nbytes = count * row_nbytes,
                )
            )
    return header._replace(tensors = tuple(tensors))


def _check_every_parameter_fed(quantizer, model):
    """Refuse a load that would leave parameters randomly initialised: a GGUF layout the tensor map does not know
    (for example a converter that fuses tensors under a name neither the map nor `_FUSED_ROW_SPLITS` knows) loads
    with MISSING keys otherwise."""
    header, mapping = getattr(quantizer, "header", None), getattr(quantizer, "mapping", None)
    if header is None or not mapping:
        return
    import copy

    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    mapping = copy.deepcopy(mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    fed = {rename_source_key(info.name, renamings, converters)[0] for info in header.tensors}
    params = dict(model.named_parameters())
    missing = [name for name in params if name not in fed]
    if missing:
        unknown = [
            info.name
            for info in header.tensors
            if rename_source_key(info.name, renamings, converters)[0] not in params
        ]
        raise NotImplementedError(
            f"Unsloth: `gguf_packed = True` cannot map this {header.architecture} GGUF: {len(missing)} model "
            f"parameters have no tensor in the file (first: {missing[:4]}), and the file holds tensors the map does "
            f"not place (first: {unknown[:4]}). Load without `gguf_packed` to use transformers' dense loader."
        )


def _check_architecture(arch, known):
    if arch not in known:
        raise NotImplementedError(
            f"Unsloth: `gguf_packed = True` does not know the GGUF architecture {arch!r} yet "
            f"(known: {', '.join(sorted(known))}). Load without `gguf_packed` to dequantize it to 16-bit instead."
        )


def _replace_with_packed_modules_factory(original):
    """transformers' `replace_with_gguf_modules`, except MoE expert stacks keep their module (and so Unsloth's MoE
    forward and expert LoRA) with uint8 placeholder parameters the loader fills with blocks."""

    def replace(
        model,
        plan,
        kernel,
        dtype = None,
    ):
        expert_suffixes = (".experts.gate_up_proj", ".experts.down_proj", ".experts.up_proj")
        experts_plan = {name: t for name, t in plan.items() if name.endswith(expert_suffixes)}
        rest = {name: t for name, t in plan.items() if name not in experts_plan}
        replaced = original(model, rest, kernel, dtype)
        if experts_plan and gguf_expert_param_class() is None:
            print(
                "Unsloth: this unsloth_zoo cannot keep GGUF MoE experts packed, so they are dequantized to 16-bit "
                "at load. Update unsloth_zoo to keep them packed."
            )
        elif experts_plan:
            for name, ggml_type in experts_plan.items():
                if not kernel.supports(ggml_type):
                    continue
                module_name, _, param_name = name.rpartition(".")
                module = model.get_submodule(module_name)
                param = module._parameters.get(param_name)
                if param is None or param.dim() != 3:
                    continue
                E, rows, cols = param.shape
                if cols % _ref.GGML_BLOCK[ggml_type][0]:
                    continue
                placeholder = torch.empty(
                    (E, rows, _ref.row_bytes(ggml_type, cols)),
                    dtype = torch.uint8,
                    device = param.device,
                )
                module._parameters[param_name] = nn.Parameter(placeholder, requires_grad = False)
                module.__dict__.setdefault("_unsloth_gguf_expert_types", {})[param_name] = (
                    int(ggml_type),
                    int(cols),
                )
                replaced[name] = module
        return replaced

    return replace


def _dequantize_blocks_compat(
    data,
    ggml_type,
    dtype = torch.float32,
):
    """transformers' `dequant.dequantize(data, ggml_type, dtype)` for every type Unsloth reads (flat result)."""
    block_elems, block_bytes = _ref.GGML_BLOCK[int(ggml_type)]
    cols = data.shape[-1] // block_bytes * block_elems
    rows = data.numel() // data.shape[-1] if data.dim() else 1
    return _ref.dequantize_gguf(
        data.reshape(rows, -1), int(ggml_type), (rows, cols), dtype
    ).reshape(-1)


@contextlib.contextmanager
def gguf_packed_load(gguf_path = None):
    """Within: transformers' GGUF quantizer keeps blocks packed with Unsloth's kernel, reads every ggml type the
    reference decodes, and maps the extra architectures in `gguf_packed_archs`."""
    reason = gguf_packed_supported()
    if reason is not None:
        raise RuntimeError(
            f"Unsloth: `gguf_packed = True` needs transformers >= {GGUF_PACKED_MIN_TRANSFORMERS} ({reason}). "
            "Upgrade transformers, or load without `gguf_packed` to dequantize the GGUF to 16-bit."
        )
    import transformers.integrations.gguf.dequant as dequant_module
    import transformers.integrations.gguf.gguf_conversion_mapping as mapping_module
    import transformers.integrations.gguf.kernels as kernels_module
    import transformers.quantizers.quantizer_gguf as quantizer_module

    from .gguf_packed_archs import EXTRA_GGUF_ARCHS

    saved_block = dict(dequant_module.GGML_BLOCK)
    saved_name = dict(dequant_module.GGML_NAME)
    saved_archs = dict(mapping_module.GGUF_ARCHS)
    quantizer_cls = quantizer_module.GgufHfQuantizer
    saved_read_header = quantizer_cls.read_header
    saved_after = quantizer_cls._process_model_after_weight_loading
    saved_update_dtype = quantizer_cls.__dict__.get("update_dtype")
    saved_get_kernel = quantizer_module.get_gguf_kernel
    saved_replace = quantizer_module.replace_with_gguf_modules
    saved_kernel_global = kernels_module._gguf_kernel
    saved_dequantize = kernels_module.dequantize
    saved_get_state_dict = quantizer_cls.get_state_dict
    saved_update_conversions = quantizer_cls.__dict__.get("update_weight_conversions")
    config_archs = _extra_gguf_config_archs()
    config_archs.__enter__()
    try:
        for ggml_type, block in _ref.GGML_BLOCK.items():
            if ggml_type not in _UNPACKED_TYPES:
                dequant_module.GGML_BLOCK.setdefault(ggml_type, block)
                dequant_module.GGML_NAME.setdefault(ggml_type, _ref.GGML_TYPE_NAMES[ggml_type])
        for arch, builder in EXTRA_GGUF_ARCHS.items():
            mapping_module.GGUF_ARCHS.setdefault(arch, builder)
        if gguf_path is not None:
            _check_architecture(_gguf_architecture(gguf_path), mapping_module.GGUF_ARCHS)

        def read_header(self, gguf_file):
            _check_architecture(_gguf_architecture(gguf_file), mapping_module.GGUF_ARCHS)
            config = self.quantization_config
            asked_dequantize = config.dequantize
            result = saved_read_header(self, gguf_file)
            if getattr(self, "header", None) is not None:
                # Split files: one header over every shard. Weight scales: kept out of the tensor map and folded
                # into their weights after the load.
                header, owners = _merged_header(gguf_file)
                # Fused attn_qkv_gate / ssm_beta_alpha (current llama.cpp converters) read as the unfused pair: row
                # slices of the same bytes in the same shard.
                header = _split_fused_gguf_tensors(header)
                for info in header.tensors:
                    if info.name not in owners:
                        owners[info.name] = owners[_fused_source_name(info.name)]
                scales = _weight_scale_tensors(header)
                if scales:
                    drop = {info.name for info in scales.values()}
                    header = header._replace(
                        tensors = tuple(i for i in header.tensors if i.name not in drop)
                    )
                self.header = header
                self._unsloth_tensor_owner = owners
                self._unsloth_weight_scales = scales
                if (
                    len(gguf_split_paths(gguf_file)) > 1
                    and config.dequantize
                    and not asked_dequantize
                ):
                    # The first shard of a split file holds metadata only, which read as "nothing quantized".
                    config.dequantize = False
                    self.validate_environment()
            return result

        def get_state_dict(self, checkpoint_file, model):
            owners = getattr(self, "_unsloth_tensor_owner", None)
            # Any tensor in a shard other than the merged header's own file (the first, often metadata only) needs
            # the owner-aware reader, even when a single shard holds every tensor.
            if (
                getattr(self, "supported", False)
                and owners
                and any(h.path != self.header.path for h in owners.values())
            ):
                return _split_state_dict(owners, self.header)
            state_dict = saved_get_state_dict(self, checkpoint_file, model)
            scales = getattr(self, "_unsloth_weight_scales", None)
            if scales and isinstance(state_dict, dict):
                drop = {info.name for info in scales.values()}
                state_dict = {k: v for k, v in state_dict.items() if k not in drop}
            return state_dict

        def after_weight_loading(self, model, **kwargs):
            _check_every_parameter_fed(self, model)
            _apply_weight_scales(self, model)
            model = saved_after(self, model, **kwargs)
            convert_gguf_packed_model(model, self.dtype)
            return model

        def update_dtype(self, dtype):
            # transformers advises float32 for a packed load because its own kernels compute in float32; Unsloth's
            # dequantize into the compute dtype, so that advice does not apply here.
            config = self.quantization_config
            dequantize = config.dequantize
            config.dequantize = True
            try:
                return saved_update_dtype(self, dtype)
            finally:
                config.dequantize = dequantize

        def update_weight_conversions(
            self,
            weight_conversions,
            model = None,
        ):
            # transformers before 5.24 passes no model (and has no model with float32-strict modules to keep).
            if model is None:
                return saved_update_conversions(self, weight_conversions)
            conversions = saved_update_conversions(self, weight_conversions, model)
            if getattr(self, "supported", False) and isinstance(conversions, list):
                conversions = _keep_float32_casts(conversions, model, self.dtype)
            return conversions

        quantizer_cls.read_header = read_header
        quantizer_cls.get_state_dict = get_state_dict
        if saved_update_conversions is not None:
            quantizer_cls.update_weight_conversions = update_weight_conversions
        quantizer_cls._process_model_after_weight_loading = after_weight_loading
        if saved_update_dtype is not None:
            quantizer_cls.update_dtype = update_dtype
        quantizer_module.get_gguf_kernel = _UnslothGgufKernel
        quantizer_module.replace_with_gguf_modules = _replace_with_packed_modules_factory(
            saved_replace
        )
        # Unpacking (embeddings, tensors a conversion must rewrite) runs the reference on any device, and never
        # reaches for the hub kernel.
        kernels_module._gguf_kernel = False
        kernels_module.dequantize = _dequantize_blocks_compat
        yield
    finally:
        dequant_module.GGML_BLOCK.clear()
        dequant_module.GGML_BLOCK.update(saved_block)
        dequant_module.GGML_NAME.clear()
        dequant_module.GGML_NAME.update(saved_name)
        mapping_module.GGUF_ARCHS.clear()
        mapping_module.GGUF_ARCHS.update(saved_archs)
        quantizer_cls.read_header = saved_read_header
        quantizer_cls.get_state_dict = saved_get_state_dict
        if saved_update_conversions is not None:
            quantizer_cls.update_weight_conversions = saved_update_conversions
        config_archs.__exit__(None, None, None)
        quantizer_cls._process_model_after_weight_loading = saved_after
        if saved_update_dtype is not None:
            quantizer_cls.update_dtype = saved_update_dtype
        quantizer_module.get_gguf_kernel = saved_get_kernel
        quantizer_module.replace_with_gguf_modules = saved_replace
        kernels_module._gguf_kernel = saved_kernel_global
        kernels_module.dequantize = saved_dequantize


# ----------------------------------------------------------------------------------------------------------
# After the load
# ----------------------------------------------------------------------------------------------------------


def _dense_parameter(weight, dtype):
    return nn.Parameter(weight.to(dtype), requires_grad = False)


def _dequantize_gguf_rows(
    packed,
    ggml_type,
    rows,
    cols,
    dtype,
    chunk_elems = 1 << 26,
):
    out = torch.empty((rows, cols), dtype = dtype, device = packed.device)
    step = max(1, chunk_elems // max(cols, 1))
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        out[start:stop] = gguf_ops().dequantize(
            packed[start:stop], ggml_type, (stop - start, cols), dtype
        )
    return out


def convert_gguf_packed_model(model, dtype = None):
    """Swap transformers' GGUF modules for Unsloth's trainable ones, dequantize embeddings and lm_head, wrap expert
    stacks, and drop the (not trainable) quantizer. Returns a census dict."""
    from transformers.integrations.gguf.utils import GgufEmbedding, GgufExperts, GgufLinear

    if dtype is None:
        dtype = getattr(model.config, "torch_dtype", None) or getattr(model.config, "dtype", None)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
    census = {"packed_linear": 0, "packed_experts": 0, "dequantized": [], "types": {}}
    dense_of = {}
    output_embeddings = model.get_output_embeddings()
    for name, module in list(model.named_modules()):
        if isinstance(module, GgufEmbedding):
            key = module.weight
            weight_scale = module.__dict__.pop("_unsloth_gguf_weight_scale_pending", None)
            if key not in dense_of:
                table = _dequantize_gguf_rows(
                    module.weight,
                    module.ggml_type,
                    module.num_embeddings,
                    module.embedding_dim,
                    dtype if weight_scale is None else torch.float32,
                )
                if weight_scale is not None:
                    table = table * weight_scale.to(table.device).reshape(())
                dense_of[key] = _dense_parameter(table, dtype)
            dense = nn.Embedding(
                module.num_embeddings, module.embedding_dim, device = "meta", dtype = dtype
            )
            dense.weight = dense_of[key]
            model.set_submodule(name, dense)
            census["dequantized"].append(name)
        elif isinstance(module, GgufLinear):
            in_features, out_features = module.in_features, module.out_features
            weight_scale = module.__dict__.pop("_unsloth_gguf_weight_scale_pending", None)
            if module is output_embeddings or name.endswith("lm_head") or weight_scale is not None:
                # lm_head stays 16-bit, as with NF4.
                # A 2-D weight with a GGUF weight scale: dense too, the scale folded in float32.
                key = module.weight
                if key not in dense_of:
                    weight = _dequantize_gguf_rows(
                        module.weight,
                        module.ggml_type,
                        out_features,
                        in_features,
                        dtype if weight_scale is None else torch.float32,
                    )
                    if module.input_permutation is not None:
                        weight = weight.index_select(1, torch.argsort(module.input_permutation))
                    if weight_scale is not None:
                        weight = weight * weight_scale.to(weight.device).reshape(())
                    dense_of[key] = _dense_parameter(weight, dtype)
                dense = nn.Linear(
                    in_features, out_features, bias = module.bias is not None, device = "meta"
                )
                dense.weight = dense_of[key]
                if module.bias is not None:
                    dense.bias = nn.Parameter(module.bias.detach().to(dtype), requires_grad = False)
                model.set_submodule(name, dense)
                census["dequantized"].append(name)
                continue
            bias = module.bias
            if bias is not None:
                bias = nn.Parameter(bias.detach().to(dtype), requires_grad = False)
            new = UnslothGGUFLinear.from_packed(
                module.weight.data,
                module.ggml_type,
                in_features,
                out_features,
                bias = bias,
                input_permutation = module.input_permutation,
                compute_dtype = dtype,
            )
            model.set_submodule(name, new)
            census["packed_linear"] += 1
            census["types"][new.ggml_type_name] = census["types"].get(new.ggml_type_name, 0) + 1
        elif isinstance(module, GgufExperts):
            # Only reached when this unsloth_zoo cannot run packed expert stacks: the experts were unpacked by
            # transformers' own module, which has no LoRA path; refuse instead of training around it.
            raise RuntimeError(
                "Unsloth: this unsloth_zoo cannot keep GGUF MoE experts packed. Update unsloth_zoo, or load "
                "without `gguf_packed`."
            )
        else:
            types_of = module.__dict__.pop("_unsloth_gguf_expert_types", None)
            expert_scales = module.__dict__.pop("_unsloth_gguf_expert_scales", {})
            if not types_of:
                continue
            cls = gguf_expert_param_class()
            for param_name, (ggml_type, cols) in types_of.items():
                param = module._parameters[param_name]
                scale = expert_scales.get(param_name)
                module._parameters[param_name] = cls(
                    param.data,
                    ggml_type = ggml_type,
                    packed_cols = cols,
                    packed_dtype = dtype,
                    packed_scales = None if scale is None else scale.to(param.device),
                )
                if scale is not None:
                    census["scaled_experts"] = census.get("scaled_experts", 0) + 1
                census["packed_experts"] += 1
                type_name = _ref.GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))
                census["types"][type_name] = census["types"].get(type_name, 0) + 1
    # Tied embeddings: lm_head and embed_tokens share the one dequantized table.
    if getattr(model.config.get_text_config(), "tie_word_embeddings", False):
        try:
            model.tie_weights()
        except Exception:
            pass
    _drop_gguf_quantizer(model)
    model._unsloth_gguf_packed = True
    model.config.unsloth_gguf_packed = True
    model._unsloth_gguf_packed_census = census
    return census


def _drop_gguf_quantizer(model):
    """transformers marks a GGUF quantizer `is_trainable = False`, so Trainer would refuse the LoRA; the packed
    modules are Unsloth's own from here on."""
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        if getattr(owner, "hf_quantizer", None) is not None:
            owner.hf_quantizer = None
        if getattr(owner, "is_quantized", False):
            owner.is_quantized = False
    config = model.config
    for cfg in {
        id(config): config,
        id(config.get_text_config()): config.get_text_config(),
    }.values():
        if hasattr(cfg, "quantization_config"):
            try:
                delattr(cfg, "quantization_config")
            except Exception:
                cfg.quantization_config = None


def is_gguf_packed_model(model):
    """Whether any Linear or expert stack of `model` (PEFT-wrapped or not) still holds GGUF blocks."""
    if not isinstance(model, nn.Module):
        return False
    return bool(_merge_densified_bases(model)) or any(
        isinstance(module, UnslothGGUFLinear)
        or any(getattr(p, "packed_format", None) == "gguf" for p in module._parameters.values())
        for module in model.modules()
    )


def densify_gguf_packed_model(model, dtype = None):
    """Replace every packed GGUF Linear / expert stack with its 16-bit value in place (merges, full saves, parity
    references). Returns the count."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, UnslothGGUFLinear):
            weight = module.dequantize_weight(dtype or module.compute_dtype)
            dense = nn.Linear(
                module.in_features, module.out_features, bias = module.bias is not None, device = "meta"
            )
            dense.weight = nn.Parameter(weight, requires_grad = False)
            if module.bias is not None:
                dense.bias = module.bias
            model.set_submodule(name, dense)
            count += 1
            continue
        for param_name, param in list(module._parameters.items()):
            if getattr(param, "packed_format", None) == "gguf":
                module._parameters[param_name] = nn.Parameter(
                    param.dequantize(dtype = dtype or param.packed_dtype), requires_grad = False
                )
                count += 1
    return count


# ----------------------------------------------------------------------------------------------------------
# PEFT: merges densify the packed base first; inits that rewrite the base weight are refused.
# ----------------------------------------------------------------------------------------------------------

_GGUF_PACKED_STATE = "_unsloth_gguf_packed_state"


def _swap_class(module, cls):
    # accelerate keeps the wrapped forward as `_old_forward`; it must follow the class.
    old = module.__dict__.get("_old_forward")
    hooked = old is not None and getattr(old, "__func__", None) is type(module).forward
    module.__class__ = cls
    if hooked:
        module._old_forward = types.MethodType(cls.forward, module)


def _densify_in_place(module):
    weight = module.dequantize_weight()
    packed = module._parameters.pop("weight_packed")
    permutation = module._buffers.pop("input_permutation", None)
    module._buffers.pop("input_inverse", None)
    module.__dict__.pop("_unsloth_packed_state", None)
    module.__dict__[_GGUF_PACKED_STATE] = (packed, permutation, module.compute_dtype)
    _swap_class(module, nn.Linear)
    module.weight = nn.Parameter(weight, requires_grad = False)


def _merge_densified_bases(model):
    """{id: base} of the packed bases a PEFT merge densified in place while their LoRA wrapper is still there
    (merge_adapter): the dense weight is base + delta, the kept GGUF blocks the base. After merge_and_unload the
    wrapper is gone and the dense weight is the model's own."""
    found = {}
    for module in model.modules():
        base = module._modules.get("base_layer")
        if base is not None and _GGUF_PACKED_STATE in base.__dict__:
            found[id(base)] = base
    return found


def _unmerged_packed_weight(module, dtype = None):
    """The base weight of a Linear a PEFT merge densified in place, decoded from the GGUF blocks it kept: its dense
    `weight` holds the merged value (base + adapter delta)."""
    packed, permutation, compute_dtype = module.__dict__[_GGUF_PACKED_STATE]
    shape = (module.out_features, module.in_features)
    inverse = None if permutation is None else torch.argsort(permutation)
    ggml_type = int(module.__dict__["_ggml_type_key"])
    return _effective_weight(packed, ggml_type, shape, dtype or compute_dtype, inverse)


def _restore_packed_in_place(module):
    packed, permutation, _ = module.__dict__.pop(_GGUF_PACKED_STATE)
    device, dtype = module.weight.device, module.weight.dtype
    del module._parameters["weight"]
    _swap_class(module, UnslothGGUFLinear)
    module.compute_dtype = dtype
    if packed.device != device:
        packed = nn.Parameter(packed.data.to(device), requires_grad = False)
    module.register_parameter("weight_packed", packed)
    module._set_input_permutation(None if permutation is None else permutation.to(device))


def patch_peft_for_gguf_packed_linears():
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear, LoraLayer
    except Exception:
        return False
    if not getattr(LoraLinear.merge, "_unsloth_gguf_patched", False):
        original_merge, original_unmerge = LoraLinear.merge, LoraLinear.unmerge

        def merge(self, *args, **kwargs):
            base = self.get_base_layer()
            if not isinstance(base, UnslothGGUFLinear):
                return original_merge(self, *args, **kwargs)
            _densify_in_place(base)
            try:
                return original_merge(self, *args, **kwargs)
            finally:
                if not self.merged_adapters:
                    _restore_packed_in_place(base)

        def unmerge(self, *args, **kwargs):
            result = original_unmerge(self, *args, **kwargs)
            base = self.get_base_layer()
            if _GGUF_PACKED_STATE in base.__dict__ and not self.merged_adapters:
                _restore_packed_in_place(base)
            return result

        merge._unsloth_gguf_patched = unmerge._unsloth_gguf_patched = True
        LoraLinear.merge, LoraLinear.unmerge = merge, unmerge

    def refuse(original):
        @functools.wraps(original)
        def init(self, *args, **kwargs):
            if isinstance(self.get_base_layer(), UnslothGGUFLinear):
                raise NotImplementedError(
                    f"Unsloth: `{original.__name__}` rewrites the base weight, which stays packed in its GGUF "
                    "blocks here and cannot be written. Use the default LoRA initialisation."
                )
            return original(self, *args, **kwargs)

        init._unsloth_gguf_patched = True
        return init

    for name in ("pissa_init", "olora_init", "corda_init", "loftq_init", "lora_ga_init"):
        original = getattr(LoraLayer, name, None)
        if original is not None and not getattr(original, "_unsloth_gguf_patched", False):
            setattr(LoraLayer, name, refuse(original))
    return True


patch_peft_for_gguf_packed_linears()


# ----------------------------------------------------------------------------------------------------------
# Merged exports: the zoo merge reads the base checkpoint, which for a GGUF is its dequantized value.
# ----------------------------------------------------------------------------------------------------------


def _base_state_items(model, dtype):
    """(checkpoint name, 16-bit tensor) of every base weight: LoRA wrappers stripped, GGUF blocks dequantized."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    modules = dict(base.named_modules())
    densified = _merge_densified_bases(base)
    seen = set()
    for key, tensor in base.state_dict(keep_vars = True).items():
        if "lora_" in key or "lora_magnitude_vector" in key:
            continue
        # modules_to_save / trainable-token wrappers: the base is `original_module`; the trained copies are the
        # adapter's, which the merge reads from the model.
        if ".modules_to_save." in key or ".token_adapter." in key:
            continue
        name = key.replace(".base_layer", "").replace(".original_module", "")
        owner_name, _, leaf = key.rpartition(".")
        owner = modules.get(owner_name)
        if leaf == "weight_packed" and isinstance(owner, UnslothGGUFLinear):
            name, value = name[: -len("_packed")], owner.dequantize_weight(dtype)
        elif leaf == "weight" and id(owner) in densified:
            # merge_adapter() densified this base in place: the merge adds the adapter itself, so write the base.
            value = _unmerged_packed_weight(owner, dtype)
        elif getattr(tensor, "packed_format", None) == "gguf":
            value = tensor.dequantize(dtype = dtype)
        else:
            # Unpacked tensors keep the dtype the model holds them in (A_log / dt_bias and other kept-float32
            # parameters stay float32); only a float32 export widens the rest.
            value = tensor.detach()
            if value.is_floating_point() and dtype == torch.float32 and value.dtype != dtype:
                value = value.to(dtype)
        if name in seen:
            continue
        seen.add(name)
        yield name, value


_CHECKPOINT_FILES = (
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
)


def write_gguf_dense_base(
    model,
    directory,
    dtype = None,
    max_shard_bytes = 4 << 30,
):
    """Write the dequantized 16-bit base of a packed GGUF model (safetensors shards + index + config.json), one
    tensor on the GPU at a time. Returns `directory`."""
    import json

    from safetensors.torch import save_file

    import glob

    os.makedirs(directory, exist_ok = True)
    # A checkpoint already there would shadow or mix with this one (transformers prefers model.safetensors over a
    # sharded index): replace it, as save_pretrained does.
    for pattern in _CHECKPOINT_FILES:
        for stale in glob.glob(os.path.join(glob.escape(str(directory)), pattern)):
            os.remove(stale)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if dtype is None:
        dtype = next(
            (m.compute_dtype for m in base.modules() if isinstance(m, UnslothGGUFLinear)),
            None,
        ) or next(
            (m.__dict__[_GGUF_PACKED_STATE][2] for m in _merge_densified_bases(base).values()),
            torch.bfloat16,
        )
    tied = bool(getattr(base.config.get_text_config(), "tie_word_embeddings", False))
    # Only the shard being filled is held in host memory; written shards are kept as file names.
    shards, shard, size, weight_map, total = [], {}, 0, {}, 0

    def flush():
        nonlocal shard, size
        if shard:
            file = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(shard, os.path.join(directory, file), metadata = {"format": "pt"})
            shards.append(file)
            for name in shard:
                weight_map[name] = file
        shard, size = {}, 0

    for name, value in _base_state_items(model, dtype):
        if tied and name == "lm_head.weight":
            continue
        value = value.contiguous().cpu()
        nbytes = value.numel() * value.element_size()
        if size and size + nbytes > max_shard_bytes:
            flush()
        shard[name] = value
        size += nbytes
        total += nbytes
    flush()
    count = len(shards)
    renamed = {}
    for index, _ in enumerate(shards, start = 1):
        old = os.path.join(directory, f"model-{index:05d}.safetensors")
        new = f"model-{index:05d}-of-{count:05d}.safetensors"
        os.replace(old, os.path.join(directory, new))
        renamed[f"model-{index:05d}.safetensors"] = new
    weight_map = {name: renamed[file] for name, file in weight_map.items()}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent = 2)
    config = base.config.to_dict()
    config.pop("quantization_config", None)
    config.pop("unsloth_gguf_packed", None)
    config["torch_dtype"] = config["dtype"] = str(dtype).replace("torch.", "")
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2, default = str)
    return directory


@contextlib.contextmanager
def gguf_dense_base_for_merge(
    model,
    save_directory,
    dtype = None,
):
    """Yields a merge-base resolver pointing at a temporary dequantized copy of the base, or None when the model
    is not a packed GGUF model. The copy sits beside `save_directory` and is removed afterwards. `dtype`: the
    copy's dtype (default the compute dtype); a float32 export decodes the blocks in float32."""
    if not is_gguf_packed_model(model):
        yield None
        return
    import shutil
    import tempfile

    parent = os.path.dirname(os.path.abspath(os.fspath(save_directory))) or "."
    os.makedirs(parent, exist_ok = True)
    directory = tempfile.mkdtemp(prefix = ".unsloth_gguf_dense_base_", dir = parent)
    try:
        print("Unsloth: dequantizing the packed GGUF base to 16-bit for the merge...")
        write_gguf_dense_base(model, directory, dtype = dtype)

        def resolver(
            model_name = None,
            *args,
            **kwargs,
        ):
            return directory

        yield resolver
    finally:
        shutil.rmtree(directory, ignore_errors = True)
