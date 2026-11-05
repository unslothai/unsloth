# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fetch the HNSW pair on demand in the background (install_hnsw_prebuilt.py). Never raises."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

TIMEOUT_S = 900.0
RETRY_S = 600.0
_OFF = {"0", "false", "no", "off"}
# Retried after RETRY_S; any other status will not change in this process.
_TRANSIENT = {"failed", "busy"}
_READY = {"ready", "already"}
# What the installer reads besides TLS / proxy settings; credentials never reach it.
_PASS_KEYS = {
    "RAG_VEC_ANN",
    "RAG_VEC_ANN_AUTO_INSTALL",
    "UNSLOTH_HNSW_MIRROR",
    "UNSLOTH_HNSW_PYPI",
    "UNSLOTH_STUDIO_HOME",
    "UNSLOTH_STUDIO_NATIVE_TLS",
    "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND",
    "STUDIO_HOME",
    "UV_INSTALL_DIR",
    "XDG_BIN_HOME",
    "LOCALAPPDATA",
    "UV_NO_INDEX",
    "UV_OFFLINE",
    "PIP_NO_INDEX",
    "UNSLOTH_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "UV_OFFLINE",
    "PIP_NO_INDEX",
}

_lock = threading.Lock()
_inflight = False
_retry_after = 0.0
_final: str | None = None
_callbacks: list[Callable[[], None]] = []
last: dict = {}


def _auto_install_off() -> bool:
    return os.environ.get("RAG_VEC_ANN_AUTO_INSTALL", "1").strip().lower() in _OFF


def ready() -> bool:
    """An attempt in this process installed or found the pair."""
    return _final in _READY


def request(on_ready: Callable[[], None]) -> bool:
    """Start one background attempt unless one is running, backing off or settled. True if this call started it."""
    global _inflight, _final
    try:
        with _lock:
            if _auto_install_off():
                _final = "disabled"
            if _final is not None or time.monotonic() < _retry_after:
                return False
            if on_ready not in _callbacks:
                _callbacks.append(on_ready)
            if _inflight:
                return False
            _inflight = True
        threading.Thread(target = _attempt, name = "rag-hnsw-install", daemon = True).start()
        return True
    except Exception:  # noqa: BLE001
        logger.debug("hnsw_runtime.request failed", exc_info = True)
        with _lock:
            _inflight = False
        return False


def _installer():
    # Same search as utils.prebuilt.update_flow.find_installer_script, without its structlog import.
    env = os.environ.get("UNSLOTH_HNSW_INSTALLER")
    if env and os.path.isfile(env):
        return Path(env)
    for up in Path(__file__).resolve().parents:
        for cand in (up / "install_hnsw_prebuilt.py", up / "studio" / "install_hnsw_prebuilt.py"):
            if cand.is_file():
                return cand
    return None


def _env() -> dict:
    try:
        from utils.s3_runtime import _INSTALL_ENV_KEYS as keys
    except Exception:  # noqa: BLE001
        keys = {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR"}
    return {k: v for k, v in os.environ.items() if k in keys or k in _PASS_KEYS}


try:
    from utils.child_stdio import utf8_child_env
except Exception:  # noqa: BLE001

    def utf8_child_env(env: dict) -> dict:
        return {**env, "PYTHONIOENCODING": "utf-8"}


def _hidden():
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs
        return windows_hidden_subprocess_kwargs()
    except Exception:  # noqa: BLE001
        return {}


def _run_installer() -> dict:
    script = _installer()
    if script is None:
        return {"status": "unsupported", "detail": "install_hnsw_prebuilt.py not found"}
    try:
        out = subprocess.run(
            [sys.executable, str(script), "--json"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = TIMEOUT_S,
            env = utf8_child_env(_env()),
            **_hidden(),
        ).stdout
    except subprocess.TimeoutExpired:
        return {"status": "failed", "detail": "installer timed out"}
    for line in reversed((out or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except ValueError:
                break
            if isinstance(result, dict) and isinstance(result.get("status"), str):
                return result
            break
    return {"status": "failed", "detail": "installer gave no result"}


def _attempt() -> None:
    global _inflight, _retry_after, _final, last
    callbacks: list = []
    try:
        result = _run_installer()
        status = result["status"]
        if status in _READY:
            importlib.invalidate_caches()
            from . import vec_ann
            if vec_ann.usearch() is None:
                status, result = (
                    "failed",
                    {**result, "status": "failed", "detail": "installed but not importable"},
                )
        logger.info("HNSW runtime: %s %s", status, result.get("detail", ""))
        with _lock:
            last = result
            if status in _READY:
                _final = status
                callbacks = list(_callbacks)
            elif status in _TRANSIENT:
                _retry_after = time.monotonic() + RETRY_S
            else:
                _final = status
    except Exception:  # noqa: BLE001
        logger.debug("hnsw_runtime attempt failed", exc_info = True)
        with _lock:
            _retry_after = time.monotonic() + RETRY_S
    finally:
        with _lock:
            _inflight = False
            if callbacks:
                _callbacks.clear()
    for cb in callbacks:
        try:
            cb()
        except Exception:  # noqa: BLE001
            logger.debug("hnsw_runtime callback failed", exc_info = True)


def reset() -> None:
    global _inflight, _retry_after, _final, last
    with _lock:
        _inflight = False
        _retry_after = 0.0
        _final = None
        _callbacks.clear()
        last = {}
