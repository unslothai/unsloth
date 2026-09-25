# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 RoPE in real arithmetic under ``torch.compile``, bit-identical to the complex form.

The stock ``apply_rotary_emb_qwen(..., use_real=False)`` multiplies ``view_as_complex`` tensors. Inductor
cannot generate complex arithmetic, so inside the compiled block the multiply falls back to ATen's
complex kernel with a copy kernel on each side, and the query/key RMSNorm in front of it cannot fuse
with it. That chain is about 6% of the bf16 kernel time on Qwen-Image-2.1.

ATen's CUDA complex multiply computes each half of ``(a + bi) * (c + di)`` with one fused multiply-add,
and which product it fuses depends on how nvcc compiled it for the card: on B200 it is
``real = fma(a, c, -(b * d)), imag = fma(b, c, a * d)``, on A100 ``imag = fma(a, d, b * c)``. So
``install`` asks the card: it multiplies a probe on the device and classifies each half against the
two fused forms (exact in float64 on inputs in [1, 2)), and the compiled form then uses the same
fusion per lane via ``addcmul``, which inductor lowers to a single ``fma`` (torch 2.12+, and 2.11
under ``emulate_precision_casts``, which Studio's compiled tiers set). Written over the full head dim
(``swap_pairs(x)`` against interleaved cos / sin tables) the rotation fuses with the norm into one
kernel and matches the complex multiply bit for bit. Inductor may schedule the fused norm's reduction
differently from the unfused one: identical on B200 and RTX PRO 6000, a few 1-ulp elements on A100
and L4, as many as the stock compile already has against eager. A card whose multiply matches
neither form, torch <= 2.10 (no ``fma`` lowering), ROCm and eager calls keep the complex form.

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
def _addcmul_lowering() -> tuple:
    """(lowers ``addcmul`` with value 1 to one ``fma`` on CUDA, only while ``emulate_precision_casts``
    is set). Read from the lowering itself: torch 2.11 gates the fma on that flag, 2.12+ does not."""
    try:
        import torch
        from torch._inductor import lowering

        if getattr(torch.version, "hip", None):
            return False, False
        src = inspect.getsource(lowering.addcmul)
        return "ops.fma(t1_val, t2_val, self_val)" in src, "not config.emulate_precision_casts" in src
    except Exception:  # noqa: BLE001 - no inductor, or its source is unreadable
        return False, False


def inductor_addcmul_is_fma() -> bool:
    """Whether this torch's inductor can lower ``addcmul`` to one ``fma``, which is what makes the real
    form round exactly like ATen's complex multiply."""
    return _addcmul_lowering()[0]


# Set by ``install``: read inside the traced wrapper, where the lru_cache above would break the graph.
_NEEDS_EMULATE = [True]


# device index -> (even lanes, odd lanes), each "x" (fma on x * cos) or "s" (fma on swapped * sin).
_FUSION: dict = {}


def classify_fusion(a: Any, b: Any, c: Any, d: Any, real: Any, imag: Any) -> Optional[tuple]:
    """Which product each half of a complex multiply fused, from float32 numpy inputs in [1, 2) (so
    every candidate is exact in float64 before its one rounding) and the multiply's float32 output.
    ``("x", "x")`` is ``real = fma(a, c, -(b * d)), imag = fma(b, c, a * d)``; ``"s"`` fuses the other
    product. None when a half matches neither."""
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
    """``classify_fusion`` for ATen's complex multiply on ``device``, called the way the RoPE calls it
    (a (1, S, H, D/2) tensor against a broadcast (S, 1, D/2) table)."""
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
        x[..., 0].numpy(), x[..., 1].numpy(), fb[..., 0].numpy(), fb[..., 1].numpy(),
        out[..., 0].numpy(), out[..., 1].numpy(),
    )


def _real_rope(x: Any, freqs_cis: Any, fusion: tuple) -> Any:
    """``apply_rotary_emb_qwen(x, freqs_cis, use_real=False)`` without complex tensors.

    ``x``: (B, S, H, D); ``freqs_cis``: complex64 (S, D/2); pairs are adjacent (x[2k], x[2k+1]).
    Per lane ``fma(x, cos, swapped * sin)`` ("x") or ``fma(swapped, sin, x * cos)`` ("s"), with
    sin = (-d, d) and swapped = (b, a): even lanes give the real half, odd lanes the imaginary one."""
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
    """Read at trace time (dynamo guards on it): without the flag torch 2.11 splits ``addcmul``."""
    from torch._inductor import config

    return bool(getattr(config, "emulate_precision_casts", False))


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
            fusion = _FUSION.get(x.device.index)
            if fusion is not None and (not _NEEDS_EMULATE[0] or _inductor_emulates()):
                return _real_rope(x, freqs_cis, fusion)
        return stock(x, freqs_cis, use_real, use_real_unbind_dim)

    functools.update_wrapper(apply_rotary_emb_qwen, stock, assigned = ("__name__", "__doc__"), updated = ())
    apply_rotary_emb_qwen.__unsloth_real_rope__ = True
    return apply_rotary_emb_qwen


def install(logger: Any = None, device: Any = None) -> bool:
    """Swap the module's ``apply_rotary_emb_qwen`` for the compile-time real form. Idempotent; False
    (stock kept) under the kill switch, without Qwen-Image 2.1, on a drifted diffusers, where
    inductor would not round like the complex kernel, or when ``device`` (default: the current CUDA
    device) multiplies complex numbers in neither fused form. Run before the first compiled forward."""
    if real_rope_disabled() or not inductor_addcmul_is_fma():
        return False
    _NEEDS_EMULATE[0] = _addcmul_lowering()[1]
    try:
        import torch

        dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        if dev.type != "cuda":
            return False
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index not in _FUSION:
            _FUSION[index] = probe_fusion(torch.device("cuda", index))
        if _FUSION[index] is None:
            if logger is not None:
                logger.info("diffusion.qwenimage21: complex RoPE kept: this card's complex multiply is not a known fma form")
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
