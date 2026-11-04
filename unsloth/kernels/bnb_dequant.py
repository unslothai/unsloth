# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
import functools
import inspect
import os

import torch

_ORIGINAL = None
_FUNCTIONAL = None
_FAST = None
_UTILS = None
_STATE = None
_NAMES = ("A", "quant_state", "absmax", "out", "blocksize", "quant_type")


def _enabled():
    return (
        os.environ.get("UNSLOTH_DECODE_FUSED_NF4", "0") in ("1", "force")
        and os.environ.get("UNSLOTH_BNB_TRITON", "1") != "0"
        and os.environ.get("UNSLOTH_BNB_NF4_LINEAR", "auto") != "0"
    )


def _replacement(
    A,
    quant_state = None,
    absmax = None,
    out = None,
    blocksize = None,
    quant_type = "fp4",
):
    if not _enabled() or not _UTILS._USE_NF4_KERNELS:
        return _ORIGINAL(A, quant_state, absmax, out, blocksize, quant_type)
    s = quant_state
    valid = (
        isinstance(s, _STATE)
        and s.quant_type == "nf4"
        and (s.nested or out is not None)
        and torch.version.hip is None
        and A.device.type == "cuda"
        and A.dtype == torch.uint8
        and A.ndim == 2
        and A.is_contiguous()
        and len(s.shape) == 2
        and s.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and s.blocksize in (32, 64, 128, 256, 512, 1024, 2048, 4096)
        and s.shape[0] > 0
        and s.shape[1] > 0
        and s.shape[1] % s.blocksize == 0
        and A.numel() * 2 == s.shape[0] * s.shape[1]
        and s.code is not None
        and s.code.dtype == torch.float32
        and s.code.numel() == 16
    )
    if valid:
        t = s.state2
        blocks = s.shape[0] * s.shape[1] // s.blocksize
        tensors = (s.code, s.absmax)
        if s.nested:
            valid = (
                s.absmax.dtype == torch.uint8
                and s.absmax.numel() == blocks
                and t is not None
                and t.blocksize == 256
                and t.dtype == torch.float32
                and t.absmax.dtype == torch.float32
                and t.absmax.numel() == (blocks + 255) // 256
                and t.code.dtype == torch.float32
                and t.code.numel() == 256
                and s.offset is not None
                and s.offset.dtype == torch.float32
                and s.offset.numel() == 1
            )
            if valid:
                tensors += (t.code, t.absmax, s.offset)
        else:
            valid = s.absmax.dtype == torch.float32 and s.absmax.numel() == blocks
        # Dynamo cannot inline generators inside an autograd.Function backward.
        for v in tensors:
            valid = valid and v.device == A.device and v.is_contiguous()
        if out is not None:
            valid = valid and (
                isinstance(out, torch.Tensor)
                and out.device == A.device
                and out.dtype == s.dtype
                and tuple(out.shape) == tuple(s.shape)
                and out.is_contiguous()
                and not out.requires_grad
            )
    if not valid:
        return _ORIGINAL(A, quant_state, absmax, out, blocksize, quant_type)
    result = _FAST(
        A.data if type(A) is not torch.Tensor else A,
        s.absmax,
        t.code if s.nested else None,
        t.absmax if s.nested else None,
        s.offset if s.nested else None,
        s.code,
        s.blocksize,
        256 if s.nested else 0,
        s.shape,
        s.dtype,
        out = out,
    )
    return result.t() if A.shape[0] == 1 else result


def install_bnb_nf4_dequant():
    if not _enabled() or torch.version.hip is not None:
        return False
    try:
        from . import utils

        if not utils._USE_NF4_KERNELS:
            return False
        import bitsandbytes.functional as functional
        from .bnb_nf4_ops import dequantize_nf4

        state_type = getattr(functional, "QuantState", None)
        if not isinstance(state_type, type):
            return False
        current = functional.dequantize_4bit
        if inspect.unwrap(current, stop = lambda fn: fn is _replacement) is _replacement:
            return True
        signature = inspect.signature(current)
        if tuple(signature.parameters) != _NAMES:
            return False
        defaults = tuple(signature.parameters[name].default for name in _NAMES[1:])
        if any(value is inspect.Parameter.empty for value in defaults):
            return False
    except (ImportError, OSError, TypeError, ValueError):
        return False
    global _ORIGINAL, _FUNCTIONAL, _FAST, _UTILS, _STATE
    _ORIGINAL, _FUNCTIONAL = current, functional
    _FAST, _UTILS, _STATE = dequantize_nf4, utils, state_type
    _replacement.__defaults__ = defaults
    functools.update_wrapper(_replacement, current)
    functional.dequantize_4bit = _replacement
    return True


def uninstall_bnb_nf4_dequant():
    if _FUNCTIONAL is not None and _FUNCTIONAL.dequantize_4bit is _replacement:
        _FUNCTIONAL.dequantize_4bit = _ORIGINAL
