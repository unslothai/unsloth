# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist the transformer-quant smoke-probe verdicts across processes (4-5 s of every cold quantised load).

Only a clean child table is stored, keyed by everything a verdict depends on (torch / CUDA / torchao / triton, driver,
card UUID + capability, the probe sources, the quant env). Any difference, or a torn file, is a miss that runs the
child as before. UNSLOTH_DIFFUSION_PROBE_CACHE=0 disables read and write.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

_ENV = "UNSLOTH_DIFFUSION_PROBE_CACHE"
_FILE_NAME = "diffusion_quant_probe.json"
_FORMAT = 1
_MAX_ENTRIES = 32
# Env prefixes that can change a verdict.
_ENV_PREFIXES = ("UNSLOTH_DIFFUSION_", "UNSLOTH_NVFP4", "UNSLOTH_INT8", "TORCHAO", "TORCHINDUCTOR_")
_ENV_IGNORED = frozenset({_ENV, "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"})
# Any edit to these invalidates every entry.
_PROBE_SOURCES = (
    "diffusion_transformer_quant.py",
    "diffusion_torchao_patches.py",
    "diffusion_native_quant.py",
)

_LOCK = threading.Lock()
_SOURCE_DIGEST: Optional[str] = None


def enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _cache_file() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import cache_root
    except ImportError:
        return None
    try:
        return Path(cache_root()) / _FILE_NAME
    except Exception:  # noqa: BLE001 - no resolvable root is a miss
        return None


def has_file() -> bool:
    """A verdict file exists (stdlib only, no key check): a later start that will most likely read its table."""
    path = _cache_file() if enabled() else None
    try:
        return path is not None and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _source_digest() -> str:
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        h = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in _PROBE_SOURCES:
            try:
                h.update(name.encode())
                h.update((here / name).read_bytes())
            except OSError:
                h.update(b"<missing>")
        _SOURCE_DIGEST = h.hexdigest()
    return _SOURCE_DIGEST


def _version(module: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:  # noqa: BLE001
        return None


def _ordinal(card: str) -> Optional[int]:
    if card == "cuda":
        return None
    if card.startswith("cuda:"):
        try:
            return int(card.split(":", 1)[1])
        except ValueError:
            return None
    return None


def fingerprint(card: str) -> Optional[dict[str, Any]]:
    """Everything a verdict for ``card`` depends on, or None. Creates no CUDA context."""
    if not str(card).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        ordinal = _ordinal(str(card))
        index = torch.cuda.current_device() if ordinal is None else ordinal
        props = torch.cuda.get_device_properties(index)
        uuid = str(getattr(props, "uuid", "") or "")
        if not uuid:
            return None
        driver = None
        get_driver = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(get_driver):
            try:
                driver = int(get_driver())
            except Exception:  # noqa: BLE001
                driver = None
        env = {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES) and k not in _ENV_IGNORED
        }
        return {
            "format": _FORMAT,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "hip": str(getattr(torch.version, "hip", None)),
            "torchao": _version("torchao"),
            "triton": _version("triton"),
            "driver": driver,
            "gpu_name": str(props.name),
            "gpu_uuid": uuid,
            "capability": f"{props.major}.{props.minor}",
            "probe_source": _source_digest(),
            "env": env,
        }
    except Exception:  # noqa: BLE001 - unidentifiable card: never persisted
        return None


def _key(fp: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys = True).encode()).hexdigest()[:32]


def _read_all(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - absent / torn / unreadable reads as empty
        return {}
    return data if isinstance(data, dict) else {}


def load(card: str) -> Optional[dict[str, bool]]:
    """The persisted verdict table for ``card`` in this exact environment, or None."""
    if not enabled():
        return None
    path = _cache_file()
    fp = fingerprint(card)
    if path is None or fp is None:
        return None
    entry = _read_all(path).get(_key(fp))
    if not isinstance(entry, dict) or entry.get("fingerprint") != fp:
        return None
    verdicts = entry.get("verdicts")
    if not isinstance(verdicts, dict):
        return None
    table = {str(k): v for k, v in verdicts.items() if isinstance(v, bool)}
    return table or None


def store(card: str, table: dict[str, Any]) -> bool:
    """Persist the definite (bool) verdicts of a clean child table. Best-effort; returns whether it was written."""
    if not enabled() or not isinstance(table, dict):
        return False
    verdicts = {str(k): v for k, v in table.items() if isinstance(v, bool)}
    # Busy/unavailable device fails every scheme alike: nothing durable learned.
    if not any(verdicts.values()):
        return False
    path = _cache_file()
    fp = fingerprint(card)
    if path is None or fp is None:
        return False
    with _LOCK:
        try:
            data = _read_all(path)
            data[_key(fp)] = {"fingerprint": fp, "verdicts": verdicts, "t": time.time()}
            if len(data) > _MAX_ENTRIES:
                oldest = sorted(
                    data,
                    key = lambda k: (data[k] or {}).get("t", 0) if isinstance(data[k], dict) else 0,
                )
                for k in oldest[: len(data) - _MAX_ENTRIES]:
                    data.pop(k, None)
            path.parent.mkdir(parents = True, exist_ok = True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, sort_keys = True), encoding = "utf-8")
            os.replace(tmp, path)
            return True
        except Exception:  # noqa: BLE001 - a cache write never fails a load
            try:
                tmp.unlink()  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass
            return False
