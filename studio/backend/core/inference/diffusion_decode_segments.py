# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run each image VAE decode in one expandable allocator segment (Linux CUDA, native allocator): its growing
activations and cuDNN workspaces otherwise each map a new fixed segment (SDXL 1024^2: 11.3 -> 10.1 GiB reserved).
UNSLOTH_DIFFUSION_DECODE_EXPANDABLE=0 turns it off."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any, Iterator, Optional

from .rocm_bf16 import is_rocm_torch

DECODE_EXPANDABLE_ENV = "UNSLOTH_DIFFUSION_DECODE_EXPANDABLE"
ENABLED_ATTR = "_unsloth_decode_expandable"
_ALLOC_CONF_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")

_lock = threading.Lock()
_depth = 0
_usable: Optional[bool] = None
_restore: Optional[str] = None
STATS = {"engaged": 0, "skipped": 0}


def _env_off() -> bool:
    return os.environ.get(DECODE_EXPANDABLE_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _conf_keys(value: str) -> dict:
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
    """Options a runtime setting would reset; a bare ``expandable_segments:True`` (unsloth_zoo, after CUDA init) is not."""
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
    if not _platform_ok() or is_rocm_torch(torch):
        return False
    if _setter(torch) is None:
        return False
    try:
        return torch.cuda.memory.get_allocator_backend() == "native"
    except Exception:  # noqa: BLE001
        return False


def _live_settings(torch: Any) -> Optional[str]:
    """Live settings string (torch >= 2.6, runtime sets included) if expandable segments are off and nothing else is set."""
    if _allocator_configured():
        return None
    try:
        snap = torch.cuda.memory._snapshot()
        live = snap["allocator_settings"]
        last = live["PYTORCH_CUDA_ALLOC_CONF"]
        if live["expandable_segments"] is not False or not isinstance(last, str):
            return None
        # Setting the flag resets max_split_size / roundup / gc threshold: a string setting any option is left alone.
        if any(k != "expandable_segments" for k in _conf_keys(last)):
            return None
    except Exception:  # noqa: BLE001
        return None
    return last


def _put_back(torch: Any, live: str) -> None:
    """A set that does not name the flag leaves it as is, so switch it off first, then restore the exact string."""
    setter = _setter(torch)
    if "expandable_segments" not in _conf_keys(live):
        setter("expandable_segments:False")
    setter(live)


def _torch() -> Any:
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
    global _depth, _usable, _restore
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
        live = _live_settings(torch) if _usable and _depth == 0 else None
        if _usable and (_depth > 0 or live is not None):
            try:
                if _depth == 0:
                    _setter(torch)("expandable_segments:True")
                    _restore = live
                _depth += 1
                engaged = True
            except Exception:  # noqa: BLE001 - refused: never retry
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
                        _put_back(torch, _restore or "")
                    except Exception:  # noqa: BLE001
                        _usable = False
                    _restore = None


def enable(vae: Any) -> bool:
    if vae is None or _env_off():
        return False
    try:
        vae.__dict__[ENABLED_ATTR] = True
    except Exception:  # noqa: BLE001
        return False
    return True


__all__ = ["DECODE_EXPANDABLE_ENV", "ENABLED_ATTR", "STATS", "enable", "expandable_decode"]
