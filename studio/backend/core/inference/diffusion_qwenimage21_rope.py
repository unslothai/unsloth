# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 RoPE in real arithmetic under ``torch.compile``, bit-identical to the complex form.

Inductor cannot fuse complex multiply; ATen's fma placement differs per card (B200 vs A100), so
``install`` probes the card and mirrors it via ``addcmul`` (one ``fma``). Kill switch:
``UNSLOTH_DIFFUSION_Q21_REAL_ROPE=0``.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import io
import os
import textwrap
import threading
import tokenize
from typing import Any, Optional

REAL_ROPE_ENV = "UNSLOTH_DIFFUSION_Q21_REAL_ROPE"

_MODULE = "diffusers.models.transformers.transformer_qwenimage21"

_FINGERPRINTS: dict[str, frozenset] = {
    "apply_rotary_emb_qwen": frozenset({"ba81dd3fc907c23a"}),
    "_qwenimage21_prepare_qkv": frozenset({"574bacc8f355456d"}),
}

_LOCK = threading.Lock()
_INSTALLED: dict = {}
# Reuse one wrapper per stock function: dynamo guards on the global's identity.
_WRAPPERS: dict = {}


def real_rope_disabled() -> bool:
    return (os.environ.get(REAL_ROPE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _digest(fn: Any) -> Optional[str]:
    """Source text, not ``ast.dump`` (varies across Python versions), minus docstrings/comments."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
        tree = ast.parse(src)
        comments = [
            tok.start
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.COMMENT
        ]
    except (OSError, TypeError, SyntaxError, ValueError, tokenize.TokenError):
        return None
    lines = src.splitlines()
    for row, col in comments:
        lines[row - 1] = lines[row - 1][:col]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            for row in range(body[0].lineno, body[0].end_lineno + 1):
                lines[row - 1] = ""
    text = "\n".join(line.rstrip() for line in lines if line.strip())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def why_unsupported(module: Any) -> Optional[str]:
    for name, want in _FINGERPRINTS.items():
        got = _digest(getattr(module, name, None))
        if got not in want:
            return f"{name} differs from the version this was written against ({got})"
    return None


@functools.lru_cache(maxsize = 1)
def _addcmul_lowering() -> tuple:
    """(addcmul -> fma, only under emulate_precision_casts): torch 2.11 gates it, 2.12+ does not."""
    try:
        import torch
        from torch._inductor import lowering

        if getattr(torch.version, "hip", None):
            return False, False
        src = inspect.getsource(lowering.addcmul)
        return (
            "ops.fma(t1_val, t2_val, self_val)" in src,
            "not config.emulate_precision_casts" in src,
        )
    except Exception:  # noqa: BLE001 - no inductor, or its source is unreadable
        return False, False


def inductor_addcmul_is_fma() -> bool:
    return _addcmul_lowering()[0]


# Set by ``install``: read inside the traced wrapper, where the lru_cache above would break the graph.
_NEEDS_EMULATE = [True]


# device index -> (even lanes, odd lanes), each "x" (fma on x * cos) or "s" (fma on swapped * sin).
_FUSION: dict = {}


def classify_fusion(a: Any, b: Any, c: Any, d: Any, real: Any, imag: Any) -> Optional[tuple]:
    """Inputs in [1, 2) keep candidates exact in float64; "x" = real fma(a, c, -bd), imag fma(b, c, ad)."""
    import numpy as np

    f32, f64 = np.float32, np.float64
    a, b, c, d = (np.asarray(t, dtype = f32).astype(f64) for t in (a, b, c, d))
    bd, ac = (b * d).astype(f32).astype(f64), (a * c).astype(f32).astype(f64)
    ad, bc = (a * d).astype(f32).astype(f64), (b * c).astype(f32).astype(f64)
    real, imag = np.asarray(real, dtype = f32), np.asarray(imag, dtype = f32)
    real_x, real_s = (a * c - bd).astype(f32), (ac - b * d).astype(f32)
    imag_x, imag_s = (b * c + ad).astype(f32), (a * d + bc).astype(f32)
    even = "x" if np.array_equal(real, real_x) else "s" if np.array_equal(real, real_s) else None
    odd = "x" if np.array_equal(imag, imag_x) else "s" if np.array_equal(imag, imag_s) else None
    if even is None or odd is None:
        return None
    return even, odd


def probe_fusion(device: Any) -> Optional[tuple]:
    import torch

    g = torch.Generator(device = "cpu").manual_seed(0)
    sign = lambda *shape: torch.randint(0, 2, shape, generator = g).float() * 2 - 1  # noqa: E731
    x = (torch.rand(1, 64, 4, 16, 2, generator = g) + 1) * sign(1, 64, 4, 16, 2)
    f = (torch.rand(64, 16, 2, generator = g) + 1) * sign(64, 16, 2)
    out = torch.view_as_real(
        torch.view_as_complex(x.to(device)) * torch.view_as_complex(f.to(device)).unsqueeze(1)
    ).cpu()
    fb = f.unsqueeze(1).expand(64, 4, 16, 2).unsqueeze(0)
    return classify_fusion(
        x[..., 0].numpy(),
        x[..., 1].numpy(),
        fb[..., 0].numpy(),
        fb[..., 1].numpy(),
        out[..., 0].numpy(),
        out[..., 1].numpy(),
    )


def _real_rope(x: Any, freqs_cis: Any, fusion: tuple) -> Any:
    import torch

    xf = x.float()
    f = torch.view_as_real(freqs_cis).unsqueeze(1)
    cos = f[..., :1].expand(*f.shape[:-1], 2).flatten(-2)
    sin = torch.stack([-f[..., 1], f[..., 1]], dim = -1).flatten(-2)
    swapped = xf.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    on_x = torch.addcmul(swapped * sin, xf, cos) if "x" in fusion else None
    on_s = torch.addcmul(xf * cos, swapped, sin) if "s" in fusion else None
    if on_s is None:
        out = on_x
    elif on_x is None:
        out = on_s
    else:
        even = torch.arange(x.shape[-1], device = x.device) % 2 == 0
        out = torch.where(even, on_x, on_s) if fusion[0] == "x" else torch.where(even, on_s, on_x)
    return out.type_as(x)


def _inductor_emulates() -> bool:
    # Read at trace time (dynamo guards on it): without the flag torch 2.11 splits ``addcmul``.
    from torch._inductor import config
    return bool(getattr(config, "emulate_precision_casts", False))


def _make_rope(stock: Any) -> Any:
    import torch

    def apply_rotary_emb_qwen(
        x,
        freqs_cis,
        use_real = True,
        use_real_unbind_dim = -1,
    ):
        if (
            not use_real
            and torch.compiler.is_compiling()
            and x.device.type == "cuda"
            and getattr(freqs_cis, "dtype", None) == torch.complex64
            and x.shape[-1] % 2 == 0
        ):
            fusion = _FUSION.get(x.device.index)
            if fusion is not None and (not _NEEDS_EMULATE[0] or _inductor_emulates()):
                return _real_rope(x, freqs_cis, fusion)
        return stock(x, freqs_cis, use_real, use_real_unbind_dim)

    functools.update_wrapper(
        apply_rotary_emb_qwen, stock, assigned = ("__name__", "__doc__"), updated = ()
    )
    apply_rotary_emb_qwen.__unsloth_real_rope__ = True
    return apply_rotary_emb_qwen


def install(logger: Any = None, device: Any = None) -> bool:
    """Idempotent; False keeps the stock RoPE. Run before the first compiled forward."""
    if real_rope_disabled() or not inductor_addcmul_is_fma():
        uninstall()
        return False
    _NEEDS_EMULATE[0] = _addcmul_lowering()[1]
    try:
        import torch

        dev = (
            torch.device(device)
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        if dev.type != "cuda":
            return False
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index not in _FUSION:
            _FUSION[index] = probe_fusion(torch.device("cuda", index))
        if _FUSION[index] is None:
            if logger is not None:
                logger.info(
                    "diffusion.qwenimage21: complex RoPE kept: this card's complex multiply is not a known fma form"
                )
            return False
    except Exception as exc:  # noqa: BLE001 - no CUDA, or the probe failed: keep the complex form
        if logger is not None:
            logger.info("diffusion.qwenimage21: complex RoPE kept: probe failed: %s", exc)
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
        logger.info(
            "diffusion.qwenimage21: RoPE runs in real arithmetic inside the compiled blocks"
        )
    return True


def uninstall() -> None:
    with _LOCK:
        for mod, stock in list(_INSTALLED.items()):
            if getattr(getattr(mod, "apply_rotary_emb_qwen", None), "__unsloth_real_rope__", False):
                mod.apply_rotary_emb_qwen = stock
            _INSTALLED.pop(mod, None)


def is_installed() -> bool:
    return bool(_INSTALLED)
