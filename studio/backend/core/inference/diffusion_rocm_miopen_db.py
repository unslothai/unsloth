# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm MIOpen find-db / perf-db for diffusion VAE convs on gfx1151. Studio runs MIOpen in immediate mode, which
takes a conv's solver from the user find-db when it has a record and otherwise from MIOpen's heuristic; the heuristic
pick is slower on the Wan VAE. Running Find on the user's machine costs minutes per shape, so Studio ships the db AMD
CI measured, one directory per exact stack in ``rocm_miopen_db/`` (arch, torch, HIP and MIOpen versions must all
match; MIOpen itself also keys the files on its build). ``UNSLOTH_DIFFUSION_ROCM_MIOPEN_DB=0`` turns it off.

MIOpen reads ``MIOPEN_USER_DB_PATH`` once, on the process's first conv: a db installed after another pipeline already
ran a conv in this process does not apply until the next process, and release restores the variable (for child
processes) but cannot unload a db MIOpen already opened."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

MIOPEN_DB_ENV = "UNSLOTH_DIFFUSION_ROCM_MIOPEN_DB"
USER_DB_ENV = "MIOPEN_USER_DB_PATH"
# A user who set any of these runs MIOpen their own way (their own db, or Find): Studio stays out.
_USER_ENVS = (USER_DB_ENV, "MIOPEN_FIND_MODE", "MIOPEN_FIND_ENFORCE")
# Measured archs only; widen with evidence.
TUNED_ARCHES = frozenset({"gfx1151"})
MANIFEST = "manifest.json"
STACK_KEYS = ("arch", "torch", "hip", "miopen", "platform")
_OWNER_ATTR = "_unsloth_miopen_db_owner"
_SHIPPED_MARK = ".unsloth_shipped"

_DB_DIR = Path(__file__).resolve().parent / "rocm_miopen_db"
_LOCK = threading.RLock()
_UNSET = object()
_STATE: dict = {"active": False, "path": None, "snapshot": _UNSET, "owners": set(), "db": None}


def _env() -> str:
    return (os.environ.get(MIOPEN_DB_ENV) or "").strip().lower()


def disabled() -> bool:
    return _env() in ("0", "off", "false", "no")


def _user_owns_miopen() -> bool:
    for k in _USER_ENVS:
        v = os.environ.get(k)
        if v is None:
            continue
        if k == USER_DB_ENV and _STATE["active"] and v == _STATE["path"]:
            continue  # ours
        return True
    return False


def _arch(device: Any = None) -> Optional[str]:
    try:
        import torch

        props = torch.cuda.get_device_properties(
            device if device is not None else torch.cuda.current_device()
        )
        name = str(getattr(props, "gcnArchName", "") or "")
        return name.split(":", 1)[0] or None
    except Exception:  # noqa: BLE001
        return None


def _device_count(torch: Any) -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def current_stack(arch: str) -> dict:
    """The keys a shipped db must match exactly; the MIOpen version comes from torch (MIOpen's own number)."""
    import torch

    try:
        miopen = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        miopen = None
    return {
        "arch": str(arch),
        "torch": str(torch.__version__),
        "hip": str(getattr(torch.version, "hip", None)),
        "miopen": str(miopen),
        "platform": sys.platform,
    }


def _manifest(path: Path) -> dict:
    try:
        data = json.loads((path / MANIFEST).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _db_files(path: Path, manifest: dict) -> list:
    names = manifest.get("files") or []
    files = [path / str(n) for n in names if isinstance(n, str) and "/" not in n and "\\" not in n]
    return files if files and all(f.is_file() for f in files) else []


def shipped_db(stack: dict) -> Optional[Path]:
    """The shipped db directory measured on exactly this stack, else None. A manifest missing a key never matches."""
    want = {k: str(stack.get(k)) for k in STACK_KEYS}
    for path in sorted(_DB_DIR.glob(f"{stack.get('arch')}_*")):
        if not path.is_dir():
            continue
        m = _manifest(path)
        have = m.get("stack") if isinstance(m.get("stack"), dict) else {}
        if {k: str(have.get(k)) for k in STACK_KEYS} == want and all(k in have for k in STACK_KEYS):
            if _db_files(path, m):
                return path
    return None


def eligible(device: Any = None) -> tuple:
    if disabled():
        return False, f"{MIOPEN_DB_ENV}=0"
    if _user_owns_miopen():
        return False, "MIOPEN_USER_DB_PATH / MIOPEN_FIND_* set by the user"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False, "no torch"
    if not is_rocm_torch(torch) or not torch.cuda.is_available():
        return False, "not ROCm"
    arch = _arch(device)
    if arch not in TUNED_ARCHES:
        return False, f"arch {arch} not measured"
    # MIOpen's user db is process-wide: every visible card must share the arch the db was measured on.
    others = {_arch(i) for i in range(_device_count(torch))} - {arch}
    if others:
        return False, f"mixed archs {sorted(map(str, others))} beside {arch}"
    return True, arch


def _digest(files: list) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _materialise(src: Path, manifest: dict) -> Path:
    """Copy the shipped files into a writable per-stack dir (MIOpen may append records there). A dir already seeded
    from the same shipped bytes is kept as is, so records MIOpen added since survive; a new shipped db replaces it."""
    from .diffusion_compile_cache import cache_root

    files = _db_files(src, manifest)
    dest = cache_root() / "miopen_db" / src.name
    digest = _digest(files)
    mark = dest / _SHIPPED_MARK
    try:
        if mark.read_text(encoding = "utf-8").strip() == digest and all(
            (dest / f.name).is_file() for f in files
        ):
            return dest
    except OSError:
        pass
    dest.mkdir(parents = True, exist_ok = True)
    for f in files:
        tmp = dest / f".{f.name}.{os.getpid()}.tmp"
        # MIOpen parses records line by line: a CRLF checkout must not leave "\r" in the last solver field.
        tmp.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        os.replace(tmp, dest / f.name)
    mark.write_text(digest + "\n", encoding = "utf-8")
    return dest


def _activate(arch: str, logger: Any = None) -> bool:
    with _LOCK:
        if _STATE["active"]:
            return True
        src = shipped_db(current_stack(arch))
        if src is None:
            return False
        manifest = _manifest(src)
        try:
            dest = _materialise(src, manifest)
        except OSError as exc:
            if logger is not None:
                logger.info(
                    "diffusion.rocm_miopen_db: could not seed %s (%s); MIOpen keeps its own db",
                    src.name,
                    exc,
                )
            return False
        _STATE.update(
            active = True,
            path = str(dest),
            snapshot = os.environ.get(USER_DB_ENV, _UNSET),
            db = src.name,
        )
        os.environ[USER_DB_ENV] = str(dest)
    if logger is not None:
        logger.info(
            "diffusion.rocm_miopen_db: %s convs read the measured MIOpen db %s (%s)",
            arch,
            src.name,
            manifest.get("measured") or "",
        )
    return True


def _restore() -> None:
    # Caller holds _LOCK.
    if _STATE["active"]:
        snap = _STATE["snapshot"]
        if os.environ.get(USER_DB_ENV) == _STATE["path"]:
            if snap is _UNSET:
                os.environ.pop(USER_DB_ENV, None)
            else:
                os.environ[USER_DB_ENV] = snap
    _STATE["owners"].clear()
    _STATE.update(active = False, path = None, snapshot = _UNSET, db = None)


def install_for_pipe(
    pipe: Any,
    device: Any = None,
    logger: Any = None,
) -> bool:
    ok, why = eligible(device)
    if not ok:
        if _STATE[
            "active"
        ]:  # a declined load (another arch, opted out) must not inherit an earlier load's db
            uninstall(None)
        return False
    with _LOCK:
        if not _activate(why, logger):
            return False
        _claim(pipe)
    return True


def _claim(pipe: Any) -> None:
    token = getattr(pipe, _OWNER_ATTR, None)
    if token is None:
        token = object()
        try:
            setattr(pipe, _OWNER_ATTR, token)
        except Exception:  # noqa: BLE001 - untaggable: never released, stays on as before
            return
        root = pipe
        while "_pipe" in getattr(
            root, "__dict__", {}
        ):  # a view: the claim lives on the pipe it wraps
            root = root.__dict__["_pipe"]
        try:
            weakref.finalize(root, _drop, token)
        except TypeError:
            pass
    with _LOCK:
        _STATE["owners"].add(token)


def _drop(token: object) -> None:
    with _LOCK:
        if token not in _STATE["owners"]:
            return
        _STATE["owners"].discard(token)
        if _STATE["active"] and not _STATE["owners"]:
            _restore()


def release(pipe: Any) -> None:
    """Pipeline teardown: drop its claim; the last owner out restores MIOPEN_USER_DB_PATH."""
    if pipe is None:
        return
    token = getattr(pipe, _OWNER_ATTR, None)
    if token is None:
        return
    try:
        delattr(pipe, _OWNER_ATTR)
    except Exception:  # noqa: BLE001
        pass
    _drop(token)


def uninstall(pipe: Any = None) -> None:
    with _LOCK:
        _restore()
    if pipe is not None:
        try:
            delattr(pipe, _OWNER_ATTR)
        except Exception:  # noqa: BLE001
            pass


def is_active() -> bool:
    return bool(_STATE["active"])
