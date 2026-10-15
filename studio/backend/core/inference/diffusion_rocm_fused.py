# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused RoPE and AdaLN modulation Triton kernels for the FLUX DiT on ROCm (bf16 / fp16).

RoPE reuses ``diffusion_flux2_rope``'s kernel (bit-identical to stock). AdaLN keeps stock's rounding chain; only the
row-reduction order differs. Inference only, never while compiling; any ineligible call runs the forward live at
install. ``UNSLOTH_DIFFUSION_FUSED_ROPE=auto|0|1`` (auto = ROCm only), ``UNSLOTH_DIFFUSION_FUSED_ADALN=auto|0|1``
(auto = off: no gfx1151 speedup and not bit-identical).
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from typing import Any, Callable, Optional

FUSED_ROPE_ENV = "UNSLOTH_DIFFUSION_FUSED_ROPE"
FUSED_ADALN_ENV = "UNSLOTH_DIFFUSION_FUSED_ADALN"
_MODULES = (
    "diffusers.models.transformers.transformer_flux",
    "diffusers.models.transformers.transformer_flux2",
)
_ROPE_ATTR = "apply_rotary_emb"
_ADALN_CLASSES = ("AdaLayerNormZero", "AdaLayerNormZeroSingle", "AdaLayerNormContinuous")
_MAX_D = 16384

_LOCK = threading.Lock()
_ROPE_STOCK: dict = {}
_ADALN_PREV: dict = {}
# Survives uninstall, so a fused forward restored later still reaches a real forward.
_ADALN_ORIGINAL: dict = {}
COUNTS = {"rope_fused": 0, "rope_stock": 0, "adaln_fused": 0, "adaln_stock": 0}


def _mode(env: str) -> str:
    v = (os.environ.get(env) or "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "force"):
        return "on"
    return "auto"


def _is_rocm() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.version, "hip", None))


# AdaLN fusion adds no speed and moves pixels; RoPE fusion is pixel-identical.
_AUTO_ON = {FUSED_ROPE_ENV: True, FUSED_ADALN_ENV: False}


def _switch_on(env: str) -> bool:
    mode = _mode(env)
    return mode == "on" or (mode == "auto" and _AUTO_ON.get(env, False) and _is_rocm())


def wanted(
    env: str,
    dtype: Any,
    device: Any = "cuda",
) -> bool:
    if not _switch_on(env):
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if dtype not in (torch.bfloat16, torch.float16) or not str(device).startswith("cuda"):
        return False
    if env == FUSED_ROPE_ENV:
        return _rope_kernel() is not None
    return _adaln_kernel() is not None


@functools.lru_cache(maxsize = 1)
def _rope_kernel() -> Optional[Callable]:
    from .diffusion_flux2_rope import _kernel
    return _kernel()


def _rope_eligible(
    x: Any, freqs_cis: Any, use_real: bool, use_real_unbind_dim: int, sequence_dim: int
) -> bool:
    import torch

    if not use_real or use_real_unbind_dim != -1 or sequence_dim != 1:
        return False
    if not isinstance(freqs_cis, (tuple, list)) or len(freqs_cis) != 2:
        return False
    if (
        not torch.is_tensor(x)
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_cuda
        or x.dim() != 4
    ):
        return False
    if x.requires_grad and torch.is_grad_enabled():
        return False
    B, S, H, D = x.shape
    if D % 2 or x.stride(3) != 1 or x.numel() == 0:
        return False
    if x.storage_offset() + sum((n - 1) * st for n, st in zip(x.shape, x.stride())) >= 2**31 - 1:
        return False
    for f in freqs_cis:
        if not torch.is_tensor(f) or f.dim() != 2 or tuple(f.shape) != (S, D):
            return False
        if f.device != x.device or f.stride(1) != 1 or f.dtype is not torch.float32:
            return False
    return True


def _make_rope(module_name: str) -> Callable:
    def _fused_apply_rotary_emb(
        x,
        freqs_cis,
        use_real: bool = True,
        use_real_unbind_dim: int = -1,
        sequence_dim: int = 2,
    ):
        import torch

        stock = _ROPE_STOCK.get(module_name) or _ROPE_ORIGINAL[module_name]
        # Checked first: Dynamo graph-breaks on Triton's import path.
        if not torch.compiler.is_compiling():
            launch = _rope_kernel()
            if launch is not None and _rope_eligible(
                x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim
            ):
                try:
                    out = launch(x, freqs_cis[0], freqs_cis[1])
                    COUNTS["rope_fused"] += 1
                    return out
                except Exception:  # noqa: BLE001 -- never fail a render over the fast path
                    pass
            COUNTS["rope_stock"] += 1
        return stock(
            x,
            freqs_cis,
            use_real = use_real,
            use_real_unbind_dim = use_real_unbind_dim,
            sequence_dim = sequence_dim,
        )

    _fused_apply_rotary_emb.__unsloth_fused_rope__ = module_name
    return _fused_apply_rotary_emb


# Survives uninstall so a forward looked up just before teardown still reaches stock.
_ROPE_ORIGINAL: dict = {}
_ROPE_FNS: dict = {m: _make_rope(m) for m in _MODULES}


def install_rope(
    module_name: str,
    dtype: Any,
    device: Any = "cuda",
) -> bool:
    if module_name not in _ROPE_FNS or not wanted(FUSED_ROPE_ENV, dtype, device):
        uninstall_rope()
        return False
    import importlib

    with _LOCK:
        if module_name in _ROPE_STOCK:
            return True
        try:
            mod = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            return False
        stock = getattr(mod, _ROPE_ATTR, None)
        if stock is None or getattr(stock, "__name__", "") != "apply_rotary_emb":
            return False
        _ROPE_STOCK[module_name] = stock
        _ROPE_ORIGINAL[module_name] = stock
        setattr(mod, _ROPE_ATTR, _ROPE_FNS[module_name])
    return True


def uninstall_rope() -> None:
    with _LOCK:
        for name in list(_ROPE_STOCK):
            stock = _ROPE_STOCK.pop(name)
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, _ROPE_ATTR, None) is _ROPE_FNS[name]:
                setattr(mod, _ROPE_ATTR, stock)


@functools.lru_cache(maxsize = 1)
def _adaln_kernel() -> Optional[Callable]:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        return None

    @triton.jit
    def _adaln_fwd(
        X, SCALE, SHIFT, OUT, S, D, sxb, sxs, scb, shb, sob, sos, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        b = row // S
        s = row - b * S
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(X + b * sxb + s * sxs + cols, mask = mask, other = 0.0).to(tl.float32)
        mean = tl.sum(x, axis = 0) / D
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis = 0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        dt = OUT.dtype.element_ty
        n = (xc * rstd).to(dt)
        sc = tl.load(SCALE + b * scb + cols, mask = mask, other = 0.0).to(tl.float32)
        m = (1.0 + sc).to(dt)
        p = (n.to(tl.float32) * m.to(tl.float32)).to(dt)
        sh = tl.load(SHIFT + b * shb + cols, mask = mask, other = 0.0).to(tl.float32)
        o = (p.to(tl.float32) + sh).to(dt)
        tl.store(OUT + b * sob + s * sos + cols, o, mask = mask)

    def launch(x: Any, scale: Any, shift: Any, eps: float) -> Any:
        import torch

        B, S, D = x.shape
        out = torch.empty((B, S, D), dtype = x.dtype, device = x.device)
        block = triton.next_power_of_2(D)
        with torch.cuda.device(x.device):
            _adaln_fwd[(B * S,)](
                x,
                scale,
                shift,
                out,
                S,
                D,
                x.stride(0),
                x.stride(1),
                scale.stride(0),
                shift.stride(0),
                out.stride(0),
                out.stride(1),
                float(eps),
                BLOCK = block,
                num_warps = 4 if block <= 1024 else (8 if block <= 4096 else 16),
                enable_fp_fusion = False,
            )
        return out

    return launch


def _plain_layer_norm(norm: Any) -> Optional[float]:
    """eps when ``norm`` is a plain ``nn.LayerNorm`` with no affine over the last dim, else None."""
    import torch

    if type(norm) is not torch.nn.LayerNorm or norm.weight is not None or norm.bias is not None:
        return None
    if len(tuple(norm.normalized_shape)) != 1:
        return None
    return float(norm.eps)


def _fused_modulate(norm: Any, x: Any, scale: Any, shift: Any) -> Optional[Any]:
    """``norm(x) * (1 + scale[:, None]) + shift[:, None]`` in one kernel, or None when not eligible."""
    import torch

    if torch.compiler.is_compiling():
        return None
    eps = _plain_layer_norm(norm)
    if eps is None or not torch.is_tensor(x) or not x.is_cuda or x.dim() != 3:
        return None
    if torch.is_grad_enabled() and (x.requires_grad or scale.requires_grad or shift.requires_grad):
        return None
    B, S, D = x.shape
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or D > _MAX_D
        or D != norm.normalized_shape[0]
        or x.numel() == 0
    ):
        return None
    if x.stride(2) != 1 or x.numel() >= 2**31 - 1:
        return None
    for t in (scale, shift):
        if (
            not torch.is_tensor(t)
            or t.dtype is not x.dtype
            or t.device != x.device
            or tuple(t.shape) != (B, D)
        ):
            return None
        if t.stride(1) != 1:
            return None
    launch = _adaln_kernel()
    if launch is None:
        return None
    try:
        return launch(x, scale, shift, eps)
    except Exception:  # noqa: BLE001 -- never fail a render over the fast path
        return None


def _prev(cls: type) -> Callable:
    prev = _ADALN_PREV.get(cls)
    return prev if prev is not None else _ADALN_ORIGINAL[cls]


def _adaln_zero_forward(
    self,
    x,
    timestep = None,
    class_labels = None,
    hidden_dtype = None,
    emb = None,
):
    cls = _CLASSES["AdaLayerNormZero"]
    if self.emb is not None or timestep is not None or class_labels is not None:
        COUNTS["adaln_stock"] += 1
        return _prev(cls)(self, x, timestep, class_labels, hidden_dtype, emb)
    e = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e.chunk(6, dim = 1)
    out = _fused_modulate(self.norm, x, scale_msa, shift_msa)
    if out is None:
        COUNTS["adaln_stock"] += 1
        out = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    else:
        COUNTS["adaln_fused"] += 1
    return out, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _adaln_zero_single_forward(
    self,
    x,
    emb = None,
):
    e = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa = e.chunk(3, dim = 1)
    out = _fused_modulate(self.norm, x, scale_msa, shift_msa)
    if out is None:
        COUNTS["adaln_stock"] += 1
        out = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    else:
        COUNTS["adaln_fused"] += 1
    return out, gate_msa


def _adaln_continuous_forward(self, x, conditioning_embedding):
    e = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = e.chunk(2, dim = 1)
    out = _fused_modulate(self.norm, x, scale, shift)
    if out is None:
        COUNTS["adaln_stock"] += 1
        out = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    else:
        COUNTS["adaln_fused"] += 1
    return out


_FORWARDS = {
    "AdaLayerNormZero": _adaln_zero_forward,
    "AdaLayerNormZeroSingle": _adaln_zero_single_forward,
    "AdaLayerNormContinuous": _adaln_continuous_forward,
}
# Source lines the fused forwards reimplement: a diffusers change keeps the stock class.
_STOCK_LINES = {
    "AdaLayerNormZero": (
        "shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)",
        "x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]",
    ),
    "AdaLayerNormZeroSingle": (
        "shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)",
        "x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]",
    ),
    "AdaLayerNormContinuous": (
        "emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))",
        "scale, shift = torch.chunk(emb, 2, dim=1)",
        "x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]",
    ),
}
_CLASSES: dict = {}


def _stock_classes() -> dict:
    if not _CLASSES:
        try:
            from diffusers.models import normalization as nm
        except Exception:  # noqa: BLE001
            return {}
        for name in _ADALN_CLASSES:
            cls = getattr(nm, name, None)
            if cls is not None:
                _CLASSES[name] = cls
    return _CLASSES


def _stock_body_ok(cls: type, name: str) -> bool:
    import inspect

    # Checked on the ORIGINAL forward (the Studio eager patch may be live on top).
    fn = cls.__dict__.get("forward")
    for cand in (getattr(fn, "__wrapped__", None), fn):
        if cand is None:
            continue
        try:
            src = inspect.getsource(cand)
        except (OSError, TypeError):
            continue
        if all(line in src for line in _STOCK_LINES[name]):
            return True
    return False


def install_adaln(dtype: Any, device: Any = "cuda") -> int:
    """Wrap the live AdaLN forwards (stock or Studio's eager addcmul patch). Returns the number wrapped."""
    uninstall_adaln()
    if not wanted(FUSED_ADALN_ENV, dtype, device):
        return 0
    classes = _stock_classes()
    with _LOCK:
        for name, cls in classes.items():
            live = cls.forward
            ours = (getattr(live, "__module__", "") or "").endswith("diffusion_eager_patches")
            if not ours and not _stock_body_ok(cls, name):
                continue
            _ADALN_PREV[cls] = live
            _ADALN_ORIGINAL.setdefault(cls, live)
            cls.forward = _FORWARDS[name]
    return len(_ADALN_PREV)


def uninstall_adaln() -> None:
    with _LOCK:
        for cls, prev in list(_ADALN_PREV.items()):
            # Not ours on top: keep the entry for the uninstall after that layer's.
            if cls.__dict__.get("forward") in _FORWARDS.values():
                cls.forward = prev
                del _ADALN_PREV[cls]


def install_for_pipe(
    pipe: Any,
    dtype: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> dict:
    """Install for a FLUX.1 / FLUX.2 denoiser, otherwise restore stock."""
    transformer = getattr(pipe, "transformer", None)
    module_name = type(transformer).__module__ if transformer is not None else ""
    if module_name not in _MODULES:
        uninstall()
        return {"rope": False, "adaln": 0}
    uninstall()
    rope = install_rope(module_name, dtype, device)
    adaln = install_adaln(dtype, device)
    if logger is not None and (rope or adaln):
        try:
            logger.info(
                "diffusion.rocm_fused: fused RoPE %s, fused AdaLN on %d classes",
                "on" if rope else "off",
                adaln,
            )
        except Exception:  # noqa: BLE001
            pass
    return {"rope": rope, "adaln": adaln}


def uninstall() -> None:
    uninstall_rope()
    uninstall_adaln()


def is_installed() -> bool:
    return bool(_ROPE_STOCK) or any(
        cls.__dict__.get("forward") in _FORWARDS.values() for cls in _ADALN_PREV
    )
