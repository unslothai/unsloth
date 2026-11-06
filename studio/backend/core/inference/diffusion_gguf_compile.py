# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compiles GGUF dequant with dynamic=True for the default tier and for max where the block cannot
compile; never where max compiles the block (would break it); torch and diffusers imported
lazily; UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT=0 disables.

Q8_0 GGUFLinear weights also get one fused dequant+cast kernel (UNSLOTH_DIFFUSION_GGUF_FUSED_Q8_0=0 disables): the
stock forward runs the compiled dequant (fp16 out, several kernels) and then an eager ``.to(compute_dtype)`` copy per
linear per forward. The fused kernel keeps the fp16 rounding of ``d * q`` before the cast (see ``_q8_0_blocks_dequant_cast``),
so the weight it produces is bit-identical to the stock one."""

from __future__ import annotations

import functools
import os
from typing import Any

from .diffusion_patch_backend import apply_patch, revert_patch


_ENV_COMPILE_DEQUANT = "UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT"
_DISABLED = {"0", "off", "false", "no"}


def _enabled(env_name: str) -> bool:
    """Enabled unless explicitly disabled (default ON)."""
    return str(os.environ.get(env_name, "1")).strip().lower() not in _DISABLED


def _gguf_utils():
    """The diffusers GGUF utils module, or None if this diffusers build lacks it."""
    try:
        from diffusers.quantizers.gguf import utils as gguf_utils  # noqa: PLC0415
        return gguf_utils
    except Exception:  # noqa: BLE001 - old/!GGUF diffusers -> accelerator is a no-op
        return None


_compiled_dequant_installed = False
_compile_error: str | None = None
_DEQUANT_ATTR = "dequantize_gguf_tensor"

_ENV_FUSED_Q8_0 = "UNSLOTH_DIFFUSION_GGUF_FUSED_Q8_0"
_FORWARD_ATTR = "forward_native"
_Q8_0 = 8  # gguf.GGMLQuantizationType.Q8_0
_Q8_0_TYPE_SIZE = 34  # one fp16 scale + 32 int8 codes per block
_fused_q8_0_installed = False


def _q8_0_blocks_dequant_cast(raw: Any, dtype: Any) -> Any:
    """Q8_0 bytes (any shape, ``n_blocks * 34`` bytes) -> ``(n_blocks, 32)`` weight in ``dtype``. Same ops and order
    as diffusers' ``dequantize_blocks_Q8_0`` (fp16 scale * int8 code -> fp16) followed by ``.to(dtype)``; the dtype
    views are taken on the contiguous byte tensor before slicing, so no slice is materialised.

    Inductor computes a fused fp16 product in fp32 and would round it once to ``dtype`` (bf16: a different result
    than fp16 round then bf16 round, and no fp16 overflow to inf). The int16 bitcast forces the fp16 store rounding
    inside the kernel; the ``& -1`` keeps inductor from folding the bitcast pair away. Eager it is a no-op. The
    result stays in the block layout: a reshape to the symbolic row width inside the graph costs a second copy."""
    import torch  # noqa: PLC0415

    blocks = raw.reshape(-1, _Q8_0_TYPE_SIZE)
    d = blocks.view(torch.int16)[:, :1].view(torch.float16)
    x = blocks.view(torch.int8)[:, 2:]
    half_bits = (d * x).view(torch.int16) & -1
    return half_bits.view(torch.float16).to(dtype)


def _q8_0_dequant_cast(
    raw: Any,
    dtype: Any,
    blocks_fn: Any = None,
) -> Any:
    """Q8_0 bytes ``(rows, n_blocks * 34)`` -> ``(rows, n_blocks * 32)`` weight in ``dtype``; ``blocks_fn`` is the
    (compiled) block dequant, eager by default. The final reshape is a free view of the contiguous block output."""
    return (blocks_fn or _q8_0_blocks_dequant_cast)(raw, dtype).reshape(raw.shape[0], -1)


def is_fused_q8_0_installed() -> bool:
    return _fused_q8_0_installed


def is_compiled_dequant_installed() -> bool:
    return _compiled_dequant_installed


def compile_error() -> str | None:
    """Why the installed compiled dequant now runs eager, or None while it still compiles."""
    return _compile_error if _compiled_dequant_installed else None


def install_compiled_dequant(logger: Any = None) -> bool:
    """Replace ``dequantize_gguf_tensor`` with ``torch.compile(orig, dynamic=True)`` via the patch
    backend. Idempotent. Returns True if the compiled dequant is in place, False otherwise."""
    global _compiled_dequant_installed, _compile_error
    if not _enabled(_ENV_COMPILE_DEQUANT):
        return False
    if _compiled_dequant_installed:
        return True
    gguf_utils = _gguf_utils()
    if gguf_utils is None or not hasattr(gguf_utils, _DEQUANT_ATTR):
        return False
    try:
        import torch  # noqa: PLC0415

        eager = gguf_utils.dequantize_gguf_tensor
        compiled = _guard(torch.compile(eager, dynamic = True), eager, logger)
        # force=True: the compiled callable's fingerprint differs from the original, which can_safely_patch would
        # correctly reject.
        if apply_patch(gguf_utils, _DEQUANT_ATTR, compiled, force = True):
            _compiled_dequant_installed = True
            _compile_error = None
            _install_fused_q8_0(gguf_utils, logger)
            return True
        return False
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "install_compiled_dequant", exc)
        _compiled_dequant_installed = False
        return False


def _guard(compiled: Any, eager: Any, logger: Any) -> Any:
    """``compiled`` behind an eager fallback, like the block and VAE decode compiles: a lazy compile that fails on
    the first generation (#9897, Triton could not build its driver) ran nothing, so the stock dequant answers."""
    failed: list = []

    from . import diffusion_compile_config  # noqa: PLC0415

    @functools.wraps(eager)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        from .diffusion_bg_compile import eager_forced  # noqa: PLC0415
        if not failed and not eager_forced():
            # The lazy compile runs in the caller's context; on torch 2.12+ the load's knob writes may not be visible.
            diffusion_compile_config.apply()
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - reraised unless a compile-time failure
                from .diffusion_speed import is_compile_failure  # noqa: PLC0415 - diffusion_speed imports this

                if not is_compile_failure(exc):
                    raise
                global _compile_error
                failed.append(type(exc).__name__)
                _compile_error = (
                    f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:300]
                )
                # The handled exception pins inductor's frames through the eager retry.
                exc.__traceback__ = None
                _warn(logger, "compiled dequant (running eager)", exc)
        return eager(*args, **kwargs)

    return guarded


def _install_fused_q8_0(gguf_utils: Any, logger: Any) -> bool:
    """Patch ``GGUFLinear.forward_native`` so a Q8_0 weight is dequantised and cast in one compiled kernel; every
    other weight (other quant types, no compute dtype) takes the stock forward unchanged."""
    global _fused_q8_0_installed
    if _fused_q8_0_installed or not _enabled(_ENV_FUSED_Q8_0):
        return _fused_q8_0_installed
    linear_cls = getattr(gguf_utils, "GGUFLinear", None)
    if linear_cls is None or not callable(getattr(linear_cls, _FORWARD_ATTR, None)):
        return False
    try:
        import torch  # noqa: PLC0415

        compiled = torch.compile(_q8_0_blocks_dequant_cast, dynamic = True)
        blocks_fn = _guard(compiled, _q8_0_blocks_dequant_cast, logger)
        stock_forward = getattr(linear_cls, _FORWARD_ATTR)
        F = torch.nn.functional

        @functools.wraps(stock_forward)
        def forward_native(self: Any, inputs: Any) -> Any:
            weight = self.weight
            dtype = self.compute_dtype
            if (
                getattr(weight, "quant_type", None) != _Q8_0
                or dtype is None
                or weight.dim() != 2
                or weight.shape[-1] % _Q8_0_TYPE_SIZE
            ):
                return stock_forward(self, inputs)
            dense = _q8_0_dequant_cast(weight.as_tensor(), dtype, blocks_fn)
            bias = self.bias.to(dtype) if self.bias is not None else None
            return F.linear(inputs, dense, bias)

        if apply_patch(linear_cls, _FORWARD_ATTR, forward_native, force = True):
            _fused_q8_0_installed = True
        return _fused_q8_0_installed
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "install fused Q8_0 dequant", exc)
        return False


def _uninstall_fused_q8_0(gguf_utils: Any) -> None:
    global _fused_q8_0_installed
    if not _fused_q8_0_installed:
        return
    linear_cls = getattr(gguf_utils, "GGUFLinear", None) if gguf_utils is not None else None
    if linear_cls is not None:
        revert_patch(linear_cls, _FORWARD_ATTR)
    _fused_q8_0_installed = False


def uninstall_compiled_dequant() -> None:
    """Restore the original ``dequantize_gguf_tensor`` and ``GGUFLinear.forward_native``. Idempotent."""
    global _compiled_dequant_installed
    if not _compiled_dequant_installed:
        return
    gguf_utils = _gguf_utils()
    _uninstall_fused_q8_0(gguf_utils)
    if gguf_utils is not None:
        revert_patch(gguf_utils, _DEQUANT_ATTR)
    _compiled_dequant_installed = False


def uninstall_all() -> None:
    """Uninstall the GGUF accelerator. Idempotent; safe to call on every unload."""
    uninstall_compiled_dequant()


def is_installed() -> bool:
    return is_compiled_dequant_installed()


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.gguf_compile: %s failed: %s", what, exc)
