# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 RoPE in real arithmetic under ``torch.compile``, bit-identical to the complex form.

The stock ``apply_rotary_emb_qwen(..., use_real=False)`` multiplies ``view_as_complex`` tensors. Inductor
cannot generate complex arithmetic, so inside the compiled block the multiply falls back to ATen's
complex kernel with a copy kernel on each side, and the query/key RMSNorm in front of it cannot fuse
with it. That chain is about 6% of the bf16 kernel time on Qwen-Image-2.1.

ATen's CUDA complex multiply computes, per pair ``(a + bi) * (c + di)``::

    real = fma(a, c, -(b * d))
    imag = fma(b, c, a * d)

which torch 2.11+ inductor emits exactly for ``addcmul(-(b * d), a, c)`` and ``addcmul(a * d, b, c)``
(it lowers ``addcmul`` to a single ``fma``). Written over the full head dim as
``addcmul(swap_pairs(x) * [-d, d], x, [c, c])``, the rotation fuses with the norm into one kernel and
the result matches the stock compiled output bit for bit. Where inductor does not lower ``addcmul`` to
``fma`` (torch <= 2.10, ROCm) the products would round differently, so the stock form is kept there.
Eager calls always take the stock form: outside a compile the complex kernel is already one launch.

Installed on the diffusers module global only when the installed ``apply_rotary_emb_qwen`` and its
caller are the functions this was written against (AST fingerprint). Kill switch:
``UNSLOTH_DIFFUSION_Q21_REAL_ROPE=0``.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import os
import textwrap
import threading
from typing import Any, Optional

REAL_ROPE_ENV = "UNSLOTH_DIFFUSION_Q21_REAL_ROPE"

_MODULE = "diffusers.models.transformers.transformer_qwenimage21"

# AST digests (docstrings stripped) of the stock functions, from the diffusers commit that added
# Qwen-Image 2.1 through current main.
_FINGERPRINTS: dict[str, frozenset] = {
    "apply_rotary_emb_qwen": frozenset({"d0ce7d309f0321b0"}),
    "_qwenimage21_prepare_qkv": frozenset({"e482e20a0db7ce74"}),
}

_LOCK = threading.Lock()
_INSTALLED: dict = {}
# One wrapper per stock function: dynamo guards on the global's identity, so a reinstall that reused
# a fresh wrapper would recompile every block.
_WRAPPERS: dict = {}


def real_rope_disabled() -> bool:
    return (os.environ.get(REAL_ROPE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _digest(fn: Any) -> Optional[str]:
    """Hash of ``fn``'s AST with docstrings removed, or None when its source is unreadable."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(fn))))
    except (OSError, TypeError, SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, annotate_fields = False).encode()).hexdigest()[:16]


def why_unsupported(module: Any) -> Optional[str]:
    for name, want in _FINGERPRINTS.items():
        got = _digest(getattr(module, name, None))
        if got not in want:
            return f"{name} differs from the version this was written against ({got})"
    return None


@functools.lru_cache(maxsize = 1)
def inductor_addcmul_is_fma() -> bool:
    """Whether this torch's inductor lowers ``addcmul`` (value 1) to one ``fma`` on CUDA, which is what
    makes the real form round exactly like ATen's complex multiply. Read from the lowering itself."""
    try:
        import torch
        from torch._inductor import lowering

        if getattr(torch.version, "hip", None):
            return False
        return "ops.fma(t1_val, t2_val, self_val)" in inspect.getsource(lowering.addcmul)
    except Exception:  # noqa: BLE001 - no inductor, or its source is unreadable
        return False


def _real_rope(x: Any, freqs_cis: Any) -> Any:
    """``apply_rotary_emb_qwen(x, freqs_cis, use_real=False)`` without complex tensors.

    ``x``: (B, S, H, D); ``freqs_cis``: complex64 (S, D/2). Pairs are adjacent (x[2k], x[2k+1])."""
    import torch

    xf = x.float()
    f = torch.view_as_real(freqs_cis).unsqueeze(1)
    cos = f[..., :1].expand(*f.shape[:-1], 2).flatten(-2)
    sin = torch.stack([-f[..., 1], f[..., 1]], dim = -1).flatten(-2)
    swapped = xf.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    # Even lanes: fma(a, c, b * -d) == fma(a, c, -(b * d)); odd lanes: fma(b, c, a * d).
    return torch.addcmul(swapped * sin, xf, cos).type_as(x)


def _make_rope(stock: Any) -> Any:
    import torch

    def apply_rotary_emb_qwen(x, freqs_cis, use_real = True, use_real_unbind_dim = -1):
        if (
            not use_real
            and torch.compiler.is_compiling()
            and x.device.type == "cuda"
            and getattr(freqs_cis, "dtype", None) == torch.complex64
            and x.shape[-1] % 2 == 0
        ):
            return _real_rope(x, freqs_cis)
        return stock(x, freqs_cis, use_real, use_real_unbind_dim)

    functools.update_wrapper(apply_rotary_emb_qwen, stock, assigned = ("__name__", "__doc__"), updated = ())
    apply_rotary_emb_qwen.__unsloth_real_rope__ = True
    return apply_rotary_emb_qwen


def install(logger: Any = None) -> bool:
    """Swap the module's ``apply_rotary_emb_qwen`` for the compile-time real form. Idempotent; False
    (stock kept) under the kill switch, without Qwen-Image 2.1, on a drifted diffusers or where
    inductor would not round like the complex kernel. Must run before the first compiled forward."""
    if real_rope_disabled() or not inductor_addcmul_is_fma():
        return False
    try:
        import importlib

        mod = importlib.import_module(_MODULE)
    except Exception:  # noqa: BLE001 - diffusers without Qwen-Image 2.1
        return False
    with _LOCK:
        current = getattr(mod, "apply_rotary_emb_qwen", None)
        if getattr(current, "__unsloth_real_rope__", False):
            return True
        why = why_unsupported(mod)
        if why is not None:
            if logger is not None:
                logger.info("diffusion.qwenimage21: complex RoPE kept: %s", why)
            return False
        wrapper = _WRAPPERS.get(current)
        if wrapper is None:
            wrapper = _WRAPPERS[current] = _make_rope(current)
        _INSTALLED[mod] = current
        mod.apply_rotary_emb_qwen = wrapper
    if logger is not None:
        logger.info("diffusion.qwenimage21: RoPE runs in real arithmetic inside the compiled blocks")
    return True


def uninstall() -> None:
    with _LOCK:
        for mod, stock in list(_INSTALLED.items()):
            if getattr(getattr(mod, "apply_rotary_emb_qwen", None), "__unsloth_real_rope__", False):
                mod.apply_rotary_emb_qwen = stock
            _INSTALLED.pop(mod, None)


def is_installed() -> bool:
    return bool(_INSTALLED)
