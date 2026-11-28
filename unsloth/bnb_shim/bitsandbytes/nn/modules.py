# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Params4bit / Linear4bit for Unsloth's bitsandbytes stand-in, with the bitsandbytes attribute
layout and state_dict format (transformers, PEFT and accelerate rebuild these from `__dict__`).
LLM.int8 (Int8Params, Linear8bitLt) needs the real library and refuses with a clear error."""

import copy
from typing import Optional

import torch
from torch import nn

from .. import functional as F
from ..autograd._functions import matmul_4bit
from ..functional import QuantState

__all__ = [
    "Int8Params",
    "Linear4bit",
    "Linear8bitLt",
    "LinearFP4",
    "LinearNF4",
    "Params4bit",
    "fix_4bit_weight_quant_state_from_module",
]

_INT8_MESSAGE = (
    "Unsloth: 8-bit LLM.int8 (load_in_8bit=True) needs the bitsandbytes package, which is not "
    "installed or could not load its native library. Use load_in_4bit=True (Unsloth runs NF4 "
    "without bitsandbytes) or 16-bit, or install bitsandbytes."
)


class Params4bit(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad = False,
        quant_state: Optional[QuantState] = None,
        blocksize: Optional[int] = None,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module = None,
        bnb_quantized: bool = False,
        **kwargs,
    ) -> "Params4bit":
        if data is None:
            data = torch.empty(0)
        if blocksize is None:
            blocksize = 64
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state["requires_grad"]
        self.blocksize = state["blocksize"]
        self.compress_statistics = state["compress_statistics"]
        self.quant_type = state["quant_type"]
        self.quant_state = state["quant_state"]
        self.data = state["data"]
        self.quant_storage = state["quant_storage"]
        self.bnb_quantized = state["bnb_quantized"]
        self.module = state["module"]

    def _qs(
        self,
        name,
        nested = False,
    ):
        qs = self.__dict__.get("quant_state")
        if qs is not None and (not nested or qs.state2 is not None):
            return qs
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    absmax = property(lambda self: self._qs("absmax").absmax)
    code = property(lambda self: self._qs("code").code)
    quant_map = property(lambda self: self._qs("quant_map").code)
    offset = property(lambda self: self._qs("offset").offset)
    state2 = property(lambda self: self._qs("state2").state2)
    nested_absmax = property(lambda self: self._qs("nested_absmax", True).state2.absmax)
    nested_blocksize = property(lambda self: self._qs("nested_blocksize", True).state2.blocksize)
    nested_quant_map = property(lambda self: self._qs("nested_quant_map", True).state2.code)
    nested_dtype = property(lambda self: self._qs("nested_dtype", True).state2.dtype)
    nested_offset = property(lambda self: self._qs("nested_offset").offset)

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state["quant_state"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        new_instance.__setstate__(self.__getstate__())
        return new_instance

    @classmethod
    def from_prequantized(
        cls,
        data: torch.Tensor,
        quantized_stats: dict,
        requires_grad: bool = False,
        device = "cuda",
        module = None,
        **kwargs,
    ) -> "Params4bit":
        self = torch.Tensor._make_subclass(cls, data.to(device))
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict = quantized_stats, device = device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True
        self.quant_storage = data.dtype
        self.module = module
        if self.module is not None:
            self.module.quant_state = self.quant_state
        return self

    def _quantize(self, device):
        w = self.data.contiguous().to(device)
        w_4bit, quant_state = F.quantize_4bit(
            w,
            blocksize = self.blocksize,
            compress_statistics = self.compress_statistics,
            quant_type = self.quant_type,
            quant_storage = self.quant_storage,
        )
        self.data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self

    def cpu(self):
        return self.to(device = "cpu")

    def cuda(
        self,
        device = None,
        non_blocking: bool = False,
    ):
        return self.to(device = "cuda" if device is None else device, non_blocking = non_blocking)

    def xpu(
        self,
        device = None,
        non_blocking: bool = False,
    ):
        return self.to(device = "xpu" if device is None else device, non_blocking = non_blocking)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type != "meta" and not self.bnb_quantized:
            return self._quantize(device)
        if self.quant_state is not None:
            self.quant_state.to(device)
        return Params4bit(
            super().to(device = device, dtype = dtype, non_blocking = non_blocking),
            requires_grad = self.requires_grad,
            quant_state = self.quant_state,
            blocksize = self.blocksize,
            compress_statistics = self.compress_statistics,
            quant_type = self.quant_type,
            quant_storage = self.quant_storage,
            bnb_quantized = self.bnb_quantized,
        )

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if kwargs is None:
            kwargs = {}
        if func in (torch.chunk, torch.split):
            tensor = args[0]
            result = super().__torch_function__(func, types, args, kwargs)

            def wrap(chunk):
                return cls(
                    data = chunk,
                    requires_grad = tensor.requires_grad,
                    quant_state = tensor.quant_state,
                    blocksize = tensor.blocksize,
                    compress_statistics = tensor.compress_statistics,
                    quant_type = tensor.quant_type,
                    quant_storage = tensor.quant_storage,
                    module = tensor.module,
                    bnb_quantized = tensor.bnb_quantized,
                )

            if isinstance(result, tuple):
                return tuple(wrap(chunk) for chunk in result)
            return wrap(result)
        return super().__torch_function__(func, types, args, kwargs)


def fix_4bit_weight_quant_state_from_module(module):
    if getattr(module.weight, "quant_state", None) is not None:
        return
    if getattr(module, "quant_state", None) is None:
        import warnings
        warnings.warn(
            "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the "
            "LinearFP4 layer first."
        )
    assert module.weight.shape[1] == 1
    if not isinstance(module.weight, Params4bit):
        module.weight = Params4bit(
            module.weight, quant_storage = module.quant_storage, bnb_quantized = True
        )
    module.weight.quant_state = module.quant_state


class Linear4bit(nn.Linear):
    """4-bit QLoRA linear layer (bitsandbytes Linear4bit layout); quantizes on `.to(device)`."""

    def __init__(
        self,
        input_features,
        output_features,
        bias = True,
        compute_dtype = None,
        compress_statistics = True,
        quant_type = "fp4",
        quant_storage = torch.uint8,
        device = None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad = False,
            compress_statistics = compress_statistics,
            quant_type = quant_type,
            quant_storage = quant_storage,
            module = self,
        )
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = compute_dtype is not None
        self.quant_state = None
        self.quant_storage = quant_storage
        self.support_avx512bf16_for_cpu = False

    def set_compute_type(self, x):
        if x.dtype in (torch.float32, torch.bfloat16):
            self.compute_dtype = x.dtype

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed = True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor):
        fix_4bit_weight_quant_state_from_module(self)
        quant_state = self.weight.quant_state
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        bias = self.bias
        if bias is not None:
            if bias.dtype != x.dtype:
                bias.data = bias.data.to(x.dtype)
            bias = bias.to(self.compute_dtype)
        return matmul_4bit(x, self.weight, bias = bias, quant_state = quant_state).to(inp_dtype)


class LinearFP4(Linear4bit):
    def __init__(
        self,
        input_features,
        output_features,
        bias = True,
        compute_dtype = None,
        compress_statistics = True,
        quant_storage = torch.uint8,
        device = None,
    ):
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "fp4",
            quant_storage,
            device,
        )


class LinearNF4(Linear4bit):
    def __init__(
        self,
        input_features,
        output_features,
        bias = True,
        compute_dtype = None,
        compress_statistics = True,
        quant_storage = torch.uint8,
        device = None,
    ):
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "nf4",
            quant_storage,
            device,
        )


class Int8Params(torch.nn.Parameter):
    """isinstance target only: LLM.int8 weights need the real bitsandbytes."""

    def __new__(
        cls,
        data = None,
        requires_grad = True,
        has_fp16_weights = False,
        CB = None,
        SCB = None,
        **kwargs,
    ):
        raise RuntimeError(_INT8_MESSAGE)


class Linear8bitLt(nn.Linear):
    """isinstance target only: LLM.int8 layers need the real bitsandbytes."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(_INT8_MESSAGE)
