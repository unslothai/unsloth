# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile a chat React component for the preview frame.

Node runs ``compile-react.mjs`` (oxc-transform only) over the source and prints
the compiled module; neither side evaluates it. The result is one of
``{"status": "ok", "code", "deps"}``, ``{"status": "error", "diagnostics"}`` or
``{"status": "unavailable", "reason"}``, never an exception, so a missing Node
reads as a message in the preview instead of a 500.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loggers import get_logger
from utils.native_path_leases import child_env_without_native_path_secret
from utils.node_runtime import resolve_node_executable
from utils.paths import ensure_dir, oxc_validator_tmp_root
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

MAX_SOURCE_BYTES = 256 * 1024
COMPILE_TIMEOUT_S = 10
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_CONCURRENT = 4

_CACHE_ENTRIES = 32
_MAX_CACHED_CODE_CHARS = 1024 * 1024
_MAX_DEPS = 100
_MAX_DEP_CHARS = 200
_MAX_DIAGNOSTICS = 20
_MAX_MESSAGE_CHARS = 2000

# Shares the OXC validator's node_modules, which the installers already `npm ci`.
_TOOL_DIR = Path(__file__).resolve().parents[1] / "data_recipe" / "oxc-validator"
_SCRIPT = _TOOL_DIR / "compile-react.mjs"
_TRANSFORM_PACKAGE = _TOOL_DIR / "node_modules" / "oxc-transform" / "package.json"

# Made on first use: on Python 3.9 a semaphore created at import binds to the main thread's
# loop, but Studio serves from a loop in another thread, so a queued compile would fail.
_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)
_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_cache_lock = threading.Lock()


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(loop)
    if semaphore is None:
        semaphore = _semaphores[loop] = asyncio.Semaphore(MAX_CONCURRENT)
    return semaphore


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _normalize_lang(lang: str) -> str:
    return "jsx" if lang == "jsx" else "tsx"


def _cache_key(source: str, lang: str) -> str:
    return hashlib.sha256(f"{lang}\0{source}".encode("utf-8", "surrogatepass")).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    with _cache_lock:
        result = _cache.get(key)
        if result is not None:
            _cache.move_to_end(key)
        return result


def _cache_put(key: str, result: dict[str, Any]) -> None:
    # Only settled answers: an unavailable Node or a timeout may be fixed by the next try.
    if result["status"] not in ("ok", "error"):
        return
    if len(result.get("code", "")) > _MAX_CACHED_CODE_CHARS:
        return
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last = False)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _normalize_result(raw: Any) -> dict[str, Any] | None:
    """The compiler's answer in the route's shape, clipped; None when it is not one."""
    if not isinstance(raw, dict):
        return None
    status = raw.get("status")
    if status == "ok":
        code, deps = raw.get("code"), raw.get("deps")
        if not isinstance(code, str) or not isinstance(deps, list):
            return None
        if not all(isinstance(dep, str) for dep in deps):
            return None
        # Clipping cannot let anything through: the frame refuses unknown specifiers itself.
        clipped = list(dict.fromkeys(dep[:_MAX_DEP_CHARS] for dep in deps))
        return {"status": "ok", "code": code, "deps": clipped[:_MAX_DEPS]}
    if status == "error":
        diagnostics = raw.get("diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            return None
        out = []
        for item in diagnostics[:_MAX_DIAGNOSTICS]:
            if not isinstance(item, dict) or not isinstance(item.get("message"), str):
                return None
            out.append(
                {
                    "message": item["message"][:_MAX_MESSAGE_CHARS],
                    "line": _non_negative_int(item.get("line")),
                    "column": _non_negative_int(item.get("column")),
                }
            )
        return {"status": "error", "diagnostics": out}
    if status == "unavailable" and raw.get("reason") == "transform_missing":
        return _unavailable("transform_missing")
    return None


def _run_compiler(source: str, lang: str) -> dict[str, Any]:
    node_executable = resolve_node_executable()
    if not node_executable:
        return _unavailable("node_missing")
    if not _SCRIPT.is_file() or not _TRANSFORM_PACKAGE.is_file():
        return _unavailable("transform_missing")

    payload = json.dumps({"source": source, "lang": lang}).encode("utf-8")
    try:
        tmp_dir = str(ensure_dir(oxc_validator_tmp_root()))
        env = child_env_without_native_path_secret()
        env["TMPDIR"] = tmp_dir
        env["TMP"] = tmp_dir
        env["TEMP"] = tmp_dir
        # Resolved node's dir first on the child PATH, as the OXC validator does.
        node_bin_dir = os.path.dirname(node_executable)
        if node_bin_dir:
            env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")
        env.pop("NODE_PATH", None)
        proc = subprocess.run(
            [node_executable, str(_SCRIPT)],
            cwd = str(_TOOL_DIR),
            input = payload,
            capture_output = True,
            check = False,
            env = env,
            timeout = COMPILE_TIMEOUT_S,
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child before re-raising.
        logger.warning("React preview compile timed out after %ss", COMPILE_TIMEOUT_S)
        return _unavailable("timeout")
    except (OSError, ValueError) as exc:
        logger.warning("React preview compiler launch failed: %s", exc)
        return _unavailable("failed")

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
        logger.warning("React preview compiler exited with %s: %s", proc.returncode, stderr[:300])
        return _unavailable("failed")
    if len(proc.stdout) > MAX_OUTPUT_BYTES:
        logger.warning("React preview compiler output too large (%s bytes)", len(proc.stdout))
        return _unavailable("failed")
    try:
        raw = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("React preview compiler printed unparseable output")
        return _unavailable("failed")
    result = _normalize_result(raw)
    if result is None:
        logger.warning("React preview compiler printed an unexpected result")
        return _unavailable("failed")
    return result


def compile_react_preview(source: str, lang: str) -> dict[str, Any]:
    """Compile ``source`` (``lang`` is ``"jsx"`` or ``"tsx"``); blocks for up to the timeout."""
    lang = _normalize_lang(lang)
    key = _cache_key(source, lang)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    result = _run_compiler(source, lang)
    _cache_put(key, result)
    return result


async def compile_react_preview_async(source: str, lang: str) -> dict[str, Any]:
    cached = _cache_get(_cache_key(source, _normalize_lang(lang)))
    if cached is not None:
        return cached
    async with _semaphore():
        task = asyncio.ensure_future(asyncio.to_thread(compile_react_preview, source, lang))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The thread keeps its Node child until it exits; hold the slot until then
            # so the cap counts compilers, not requests.
            await asyncio.wait({task})
            raise
