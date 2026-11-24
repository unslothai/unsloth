# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-process cache of quant probe verdicts, keyed by torch/CUDA/driver/card; mismatch is a miss."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
_DRIVER: list = []  # [identity or None], read once per process


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


# Distributions that install the ``triton`` module: Windows, ROCm and XPU wheels (current names and the older
# pytorch-triton-* ones) are not named ``triton``, and kernels are chosen by the Triton version (the INT8 attention
# rotation on 3.3), so verdicts must key on it there too.
_TRITON_DISTS = (
    "triton",
    "triton-windows",
    "triton-rocm",
    "triton-xpu",
    "pytorch-triton-rocm",
    "pytorch-triton",
    "pytorch-triton-xpu",
)


def _triton_version() -> Optional[str]:
    for dist in _TRITON_DISTS:
        found = _version(dist)
        if found is not None:
            return found if dist == "triton" else f"{dist}=={found}"
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


def _nvidia_driver() -> Optional[str]:
    """The NVIDIA driver build, without a CUDA context: the kernel module's version (Linux), else the loaded user-mode
    driver library's name, size and mtime (WSL, containers without the proc file, Windows), plus the CUDA version
    the driver supports. None when no build identity can be read."""
    import ctypes

    parts = []
    try:
        lib = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        version = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
            parts.append(f"cuda{version.value}")
    except Exception:  # noqa: BLE001
        pass
    build = None
    try:
        with open("/proc/driver/nvidia/version", encoding = "utf-8", errors = "replace") as f:
            line = f.readline()
        match = re.search(r"\s(\d+\.\d+(?:\.\d+)*)\s", line)
        build = f"nvrm{match.group(1)}" if match else None
    except OSError:
        pass
    if build is None:
        paths = []
        if os.name == "nt":
            paths.append(
                os.path.join(
                    os.environ.get("SystemRoot") or r"C:\Windows", "System32", "nvcuda.dll"
                )
            )
        else:
            try:
                with open("/proc/self/maps", encoding = "utf-8", errors = "replace") as f:
                    paths = sorted({ln.split()[-1] for ln in f if "/libcuda.so" in ln})
            except OSError:
                pass
        for path in paths:
            try:
                info = os.stat(path)
                build = (
                    f"{os.path.basename(os.path.realpath(path))}:{info.st_size}:{info.st_mtime_ns}"
                )
                break
            except OSError:
                continue
    if build is None:
        return None
    return ",".join(parts + [build])


def _rocm_driver() -> Optional[str]:
    from .diffusion_attention import _rocm_driver_version
    found = _rocm_driver_version()
    return None if not found or found.endswith(":?") else found


def _driver_identity(hip: bool) -> Optional[str]:
    """The GPU driver this process runs on, read once per process; None when it cannot be read, so a verdict that a
    driver update could change is never adopted on an unknown driver."""
    if not _DRIVER:
        try:
            found = _rocm_driver() if hip else _nvidia_driver()
        except Exception:  # noqa: BLE001
            found = None
        _DRIVER.append(found)
    return _DRIVER[0]


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
        hip = getattr(torch.version, "hip", None)
        driver = _driver_identity(bool(hip))
        if driver is None:  # a driver change could not be seen: never persisted
            return None
        env = {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES) and k not in _ENV_IGNORED
        }
        return {
            "format": _FORMAT,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "hip": str(hip),
            "torchao": _version("torchao"),
            "triton": _triton_version(),
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
    # No scheme passed: a busy / unavailable device fails every scheme alike, so nothing durable was learned.
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


class Namespace:
    """Per-card decisions in ``diffusion_<name>.json``, keyed by the probe fingerprint plus a digest of ``sources``.
    Unreadable files read as empty; writes never raise."""

    def __init__(
        self,
        name: str,
        sources: tuple,
        max_cards: int = _MAX_ENTRIES,
        extra: Optional[Any] = None,
    ) -> None:
        self.name = str(name)
        self.sources = tuple(sources)
        self.max_cards = int(max_cards)
        self.extra = extra  # () -> dict of extra key components
        self._digest: Optional[str] = None
        self._tables: dict = {}  # (file, card) -> (slot key, fingerprint, table)

    def _source_digest(self) -> str:
        if self._digest is None:
            h = hashlib.sha256()
            here = Path(__file__).resolve().parent
            for name in self.sources:
                try:
                    h.update(name.encode())
                    h.update((here / name).read_bytes())
                except OSError:
                    h.update(b"<missing>")
            self._digest = h.hexdigest()
        return self._digest

    def _path(self) -> Optional[Path]:
        path = _cache_file()
        return None if path is None else path.with_name(f"diffusion_{self.name}.json")

    def _slot(self, card: str) -> Optional[tuple]:
        if not enabled():
            return None
        path = self._path()
        if path is None:
            return None
        found = self._tables.get((str(path), card))
        if found is not None:
            return found
        fp = fingerprint(card)
        if fp is None:
            return None
        fp = dict(fp, namespace = self.name, source = self._source_digest())
        if self.extra is not None:
            try:
                more = self.extra()
            except Exception:  # noqa: BLE001
                return None
            if not isinstance(more, dict):
                return None
            fp["extra"] = json.loads(json.dumps(more, sort_keys = True, default = str))
        key = _key(fp)
        entry = _read_all(path).get(key)
        table = (
            dict(entry.get("decisions") or {})
            if isinstance(entry, dict) and entry.get("fingerprint") == fp
            else {}
        )
        found = self._tables[(str(path), card)] = (key, fp, table)
        return found

    def get(self, card: str, key: str) -> Any:
        slot = self._slot(card)
        return None if slot is None else slot[2].get(str(key))

    def put(self, card: str, key: str, value: Any) -> bool:
        slot = self._slot(card)
        if slot is None:
            return False
        slot_key, fp, table = slot
        table[str(key)] = value
        path = self._path()
        if path is None:
            return False
        tmp: Optional[Path] = None
        with _LOCK:
            try:
                data = _read_all(path)
                entry = data.get(slot_key)
                merged = (
                    dict(entry.get("decisions") or {})
                    if isinstance(entry, dict) and entry.get("fingerprint") == fp
                    else {}
                )
                merged.update(table)
                data[slot_key] = {"fingerprint": fp, "decisions": merged, "t": time.time()}
                if len(data) > self.max_cards:
                    oldest = sorted(
                        data,
                        key = lambda k: (data[k] or {}).get("t", 0)
                        if isinstance(data[k], dict)
                        else 0,
                    )
                    for k in oldest[: len(data) - self.max_cards]:
                        data.pop(k, None)
                path.parent.mkdir(parents = True, exist_ok = True)
                tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(data, sort_keys = True), encoding = "utf-8")
                os.replace(tmp, path)
                return True
            except Exception:  # noqa: BLE001
                if tmp is not None:
                    try:
                        tmp.unlink()
                    except Exception:  # noqa: BLE001
                        pass
                return False

    def forget(self) -> None:
        self._tables.clear()
