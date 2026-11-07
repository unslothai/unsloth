# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Expandable allocator segments for the span of one image VAE decode (Linux CUDA, native allocator only).

An untiled VAE decode allocates activations and cuDNN workspaces that grow stage by stage (SDXL at 1024^2: 134 MiB
up to 1 GiB). Each larger request misses the smaller blocks cached before it, so the caching allocator maps a new
fixed segment per stage and keeps them all: SDXL holds 11.3 GiB after its first render for a 9.0 GiB live peak (T4
fp16 and L4 bf16 alike). Inside one expandable segment the decode reuses and grows the same mapping instead (SDXL:
10.1 GiB). The setting is switched on for the decode and back off after it; kernels, tiling and pixels do not change.
Off when the process configured the allocator itself (any other allocator option, or expandable segments already on),
on ROCm, Windows, WSL and Tegra, and during compile or graph capture. UNSLOTH_DIFFUSION_DECODE_EXPANDABLE=0 turns it
off."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any, Iterator, Optional

DECODE_EXPANDABLE_ENV = "UNSLOTH_DIFFUSION_DECODE_EXPANDABLE"
ENABLED_ATTR = "_unsloth_decode_expandable"
_ALLOC_CONF_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")

_lock = threading.Lock()
_depth = 0
# None = not decided yet for this process; False = never engages; True = may engage.
_usable: Optional[bool] = None
STATS = {"engaged": 0, "skipped": 0}


def _env_off() -> bool:
    return os.environ.get(DECODE_EXPANDABLE_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _conf_keys(value: str) -> dict:
    """Top-level ``key:value`` pairs of an allocator config (bracketed lists keep their commas)."""
    out, part, depth = {}, "", 0
    for ch in value + ",":
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            key, _, val = part.partition(":")
            if key.strip():
                out[key.strip().lower()] = val.strip().lower()
            part = ""
        else:
            part += ch
    return out


def _allocator_configured() -> bool:
    """The process set allocator options this module would reset or override. A bare ``expandable_segments:True``
    is not one: unsloth_zoo exports it after CUDA initialised here, and the live segments decide that case."""
    for name in _ALLOC_CONF_VARS:
        keys = _conf_keys(os.environ.get(name) or "")
        if any(k != "expandable_segments" for k in keys):
            return True
        if keys.get("expandable_segments", "true") != "true":
            return True
    return False


def _platform_ok() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return False
    try:
        release = os.uname().release.lower()
    except Exception:  # noqa: BLE001
        return False
    if "microsoft" in release or "tegra" in release or os.path.exists("/etc/nv_tegra_release"):
        return False
    return True


def _setter(torch: Any) -> Any:
    c = getattr(torch, "_C", None)
    return getattr(c, "_accelerator_setAllocatorSettings", None) or getattr(
        c, "_cuda_cudaCachingAllocator_set_allocator_settings", None
    )


def _decide(torch: Any) -> bool:
    if not _platform_ok() or _allocator_configured():
        return False
    if getattr(getattr(torch, "version", None), "hip", None):
        return False
    if _setter(torch) is None:
        return False
    try:
        if torch.cuda.memory.get_allocator_backend() != "native":
            return False
        # Expandable segments already on process-wide: nothing to add, and switching them off after would be wrong.
        if any(seg.get("is_expandable") for seg in torch.cuda.memory_snapshot()):
            return False
    except Exception:  # noqa: BLE001 - unreadable allocator state: leave it alone
        return False
    return True


def _torch() -> Any:
    # Never imports torch: a decode on a CUDA tensor has it loaded already.
    return sys.modules.get("torch")


def _cuda_tensor(z: Any) -> bool:
    device = getattr(z, "device", None)
    return getattr(device, "type", None) == "cuda"


def _busy(torch: Any) -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and compiler.is_compiling():
            return True
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return True


@contextlib.contextmanager
def expandable_decode(vae: Any, z: Any) -> Iterator[None]:
    """Run the enclosed decode with expandable segments on, when ``vae`` opted in and this process allows it."""
    global _depth, _usable
    if not vae.__dict__.get(ENABLED_ATTR, False) or _env_off() or not _cuda_tensor(z):
        yield
        return
    torch = _torch()
    if torch is None or _busy(torch):
        yield
        return
    engaged = False
    with _lock:
        if _usable is None:
            _usable = _decide(torch)
        if _usable:
            try:
                if _depth == 0:
                    _setter(torch)("expandable_segments:True")
                _depth += 1
                engaged = True
            except Exception:  # noqa: BLE001 - this torch refuses the setting: never try again
                _usable = False
        if engaged:
            STATS["engaged"] += 1
        else:
            STATS["skipped"] += 1
    try:
        yield
    finally:
        if engaged:
            with _lock:
                _depth -= 1
                if _depth == 0:
                    try:
                        _setter(torch)("expandable_segments:False")
                    except Exception:  # noqa: BLE001
                        _usable = False


def enable(vae: Any) -> bool:
    """Opt an image VAE in; its outermost decode scope (diffusion_vae_oom) applies it."""
    if vae is None or _env_off():
        return False
    try:
        vae.__dict__[ENABLED_ATTR] = True
    except Exception:  # noqa: BLE001
        return False
    return True


__all__ = ["DECODE_EXPANDABLE_ENV", "ENABLED_ATTR", "STATS", "enable", "expandable_decode"]
