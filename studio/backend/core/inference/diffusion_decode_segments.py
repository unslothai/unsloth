# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run each image VAE decode in one expandable allocator segment (Linux CUDA, native allocator): its growing
activations and cuDNN workspaces otherwise each map a new fixed segment (SDXL 1024^2: 11.3 -> 10.1 GiB reserved).
UNSLOTH_DIFFUSION_DECODE_EXPANDABLE=0 turns it off.

The toggle starts from the live allocator settings, not the environment: every ``set_allocator_settings`` call resets
``max_split_size_mb`` / ``garbage_collection_threshold`` / ``roundup_power2_divisions`` to their defaults unless the
string names them again (``AllocatorConfig::parseArgs``, torch 2.6 through 2.12), so both flips re-send the last
applied string with only ``expandable_segments`` changed, and a restore that does not read back equal ends it."""

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
_saved: Optional[tuple] = None
# (string a restore replaced, string it wrote): the next decode reads the latter as the former.
_written: tuple = (None, None)
STATS = {"engaged": 0, "skipped": 0}


def _env_off() -> bool:
    return os.environ.get(DECODE_EXPANDABLE_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _conf_parts(value: str) -> list:
    """Top-level ``key:value`` items of an allocator string, as written (bracketed lists kept whole)."""
    out, part, depth = [], "", 0
    for ch in value + ",":
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            if part.strip():
                out.append(part.strip())
            part = ""
        else:
            part += ch
    return out


def _conf_keys(value: str) -> dict:
    out = {}
    for part in _conf_parts(value):
        key, _, val = part.partition(":")
        if key.strip():
            out[key.strip().lower()] = val.strip().lower()
    return out


def _with_expandable(conf: str, on: bool) -> str:
    """``conf`` with only ``expandable_segments`` set to ``on``."""
    parts = [
        p for p in _conf_parts(conf) if p.partition(":")[0].strip().lower() != "expandable_segments"
    ]
    return ",".join(parts + [f"expandable_segments:{on}"])


_LAST_SETTINGS = "PYTORCH_CUDA_ALLOC_CONF"


def _live_settings(torch: Any) -> Optional[dict]:
    """The allocator's live settings (``memory._snapshot()['allocator_settings']``, torch 2.6+): the last applied
    string under ``PYTORCH_CUDA_ALLOC_CONF`` plus the values it produced. None when this torch cannot say."""
    try:
        settings = torch.cuda.memory._snapshot().get("allocator_settings")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(settings, dict) or "expandable_segments" not in settings:
        return None
    if not isinstance(settings.get(_LAST_SETTINGS), str):
        return None
    return settings


def _base_conf(settings: dict) -> Optional[str]:
    """The string the decode flips from, or None when it must not: already on (turning it off after would be
    wrong) or switched off on purpose. The string a previous restore wrote stands for the one it replaced."""
    if bool(settings["expandable_segments"]):
        return None
    last = settings[_LAST_SETTINGS]
    if _written[1] is not None and last == _written[1]:
        return _written[0]
    return None if "expandable_segments" in _conf_keys(last) else last


def _same_settings(a: Optional[dict], b: Optional[dict]) -> bool:
    if a is None or b is None:
        return False
    keys = (set(a) | set(b)) - {_LAST_SETTINGS}
    return all(a.get(k) == b.get(k) for k in keys)


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
    if not _platform_ok():
        return False
    if is_rocm_torch(torch):
        return False
    if _setter(torch) is None:
        return False
    try:
        if torch.cuda.memory.get_allocator_backend() != "native":
            return False
    except Exception:  # noqa: BLE001
        return False
    # Settings that cannot be read back cannot be restored.
    return _live_settings(torch) is not None


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
    global _depth, _usable, _saved
    if not vae.__dict__.get(ENABLED_ATTR, False) or _env_off() or not _cuda_tensor(z):
        yield
        return
    torch = _torch()
    if torch is None or _busy(torch):
        yield
        return
    engaged = False
    live = None
    with _lock:
        if _usable is None:
            _usable = _decide(torch)
        if _usable:
            try:
                if _depth == 0:
                    live = _live_settings(torch)
                    base = None if live is None else _base_conf(live)
                    if live is None:
                        _usable = False
                    elif base is not None:
                        _setter(torch)(_with_expandable(base, True))
                        _saved = (base, live)
                        _depth = 1
                        engaged = True
                else:
                    _depth += 1
                    engaged = True
            except Exception:  # noqa: BLE001 - refused: put the live string back, never retry
                _usable = False
                if live is not None:
                    _restore(torch, (live[_LAST_SETTINGS], live))
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
                    saved, _saved = _saved, None
                    if not _restore(torch, saved):
                        _usable = False


def _restore(torch: Any, saved: Optional[tuple]) -> bool:
    """Re-apply the saved string with expandable segments off; True when the live settings read back equal."""
    global _written
    if saved is None:
        return False
    base, live = saved
    restored = _with_expandable(base, False)
    try:
        _setter(torch)(restored)
    except Exception:  # noqa: BLE001
        return False
    _written = (base, restored)
    return _same_settings(_live_settings(torch), live)


def enable(vae: Any) -> bool:
    if vae is None or _env_off():
        return False
    try:
        vae.__dict__[ENABLED_ATTR] = True
    except Exception:  # noqa: BLE001
        return False
    return True


__all__ = ["DECODE_EXPANDABLE_ENV", "ENABLED_ATTR", "STATS", "enable", "expandable_decode"]
