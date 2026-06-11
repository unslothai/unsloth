# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app llama.cpp prebuilt update.

Builds on utils.llama_cpp_freshness (which detects whether a newer prebuilt
release exists) and adds the *apply* half: run install_llama_prebuilt.py to
download the newest bundle for this host and atomically swap it in place, so
the next model load uses it.

Design notes:
- Detection is delegated to check_prebuilt_freshness(). We surface an
  ``update_available`` flag (installed_tag != latest_tag) which is laxer than
  freshness' ``stale`` (which additionally requires the install to be >= 3 days
  old). The UI shows the "Update llama.cpp" affordance on update_available.
- The install is slow (download + extract + validate), so it runs on a daemon
  thread; callers poll get_update_status() for the job state.
- Everything fails open: a missing marker / offline GitHub / source build just
  reports update_available=False and never blocks the app.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import structlog

from utils.llama_cpp_freshness import (
    _INSTALL_MARKER_NAME,
    check_prebuilt_freshness,
    latest_published_release,
    read_install_marker,
    reset_caches,
)

logger = structlog.get_logger(__name__)

DEFAULT_PUBLISHED_REPO = "unslothai/llama.cpp"
_INSTALL_TIMEOUT_SECONDS = 1800  # 30 min ceiling for download + build/validate

# Background job state. Single in-flight update at a time, guarded by _job_lock.
_JOB_IDLE = "idle"
_JOB_RUNNING = "running"
_JOB_SUCCESS = "success"
_JOB_ERROR = "error"

_job_lock = threading.Lock()
_job: dict = {
    "state": _JOB_IDLE,
    "message": "",
    "from_tag": None,
    "to_tag": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _find_binary() -> Optional[str]:
    """Locate the active llama-server binary via the inference backend's own
    resolver, so update targets exactly what Studio runs. Lazy import keeps the
    heavy inference module off this module's import path."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._find_llama_server_binary()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llama update: binary discovery failed", error = str(exc))
        return None


def _install_dir_for(binary_path: Optional[str]) -> Optional[Path]:
    """The directory holding UNSLOTH_PREBUILT_INFO.json -- i.e. the install root
    install_llama_prebuilt.py wrote and the one we re-install into. Walks up from
    the binary the same way read_install_marker() does."""
    if not binary_path:
        return None
    p = Path(binary_path)
    for parent in p.parents[:5]:
        if (parent / _INSTALL_MARKER_NAME).is_file():
            return parent
    return None


def _installer_script() -> Optional[Path]:
    """Locate install_llama_prebuilt.py. Honours UNSLOTH_LLAMA_INSTALLER, then
    searches up from this file for both ``<root>/install_llama_prebuilt.py`` and
    ``<root>/studio/install_llama_prebuilt.py`` so it works in the dev tree and
    in an installed Studio layout."""
    env = os.environ.get("UNSLOTH_LLAMA_INSTALLER")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for up in here.parents:
        for cand in (up / "install_llama_prebuilt.py", up / "studio" / "install_llama_prebuilt.py"):
            if cand.is_file():
                return cand
    return None


# Markerless (source-build) installs have no UNSLOTH_PREBUILT_INFO.json, so we
# ask the installer whether an official prebuilt now exists for this host. Memo
# is 24h; only successful answers are cached so a network blip retries.
_RESOLVE_TTL_SECONDS = 24 * 60 * 60
_resolve_memo: dict = {}


def _resolve_prebuilt_for_host(*, force_refresh: bool = False) -> Optional[dict]:
    """Run install_llama_prebuilt.py --resolve-prebuilt (no download) and return
    {prebuilt_available, repo, release_tag, llama_tag, asset, install_kind} or
    None. Fail-open: any error -> None so a source build never blocks the app."""
    now = time.time()
    if not force_refresh and _resolve_memo:
        if now - _resolve_memo.get("at", 0.0) < _RESOLVE_TTL_SECONDS:
            return _resolve_memo.get("value")
    script = _installer_script()
    if script is None:
        return None
    value: Optional[dict] = None
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--resolve-prebuilt",
                "latest",
                "--output-format",
                "json",
            ],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            parsed = json.loads(out.splitlines()[-1])
            if isinstance(parsed, dict):
                value = parsed
    except Exception as exc:  # pragma: no cover - subprocess/json defensive
        logger.debug("llama update: resolve-prebuilt failed", error = str(exc))
        value = None
    if value is not None:  # cache real answers; let failures retry next poll
        _resolve_memo.update(at = now, value = value)
    return value


def _installed_build_number(binary: Optional[str]) -> Optional[int]:
    """Best-effort build number from ``llama-server --version`` (e.g.
    'version: 9585 (abc)'). None when unparseable or <= 1: a source build with
    no git tags reports 'version: 1', which we treat as unknown (offer update)."""
    if not binary:
        return None
    try:
        proc = subprocess.run([binary, "--version"], capture_output = True, text = True, timeout = 20)
    except Exception:  # pragma: no cover - defensive
        return None
    m = re.search(r"version:\s*(\d+)", (proc.stderr or "") + (proc.stdout or ""))
    if not m:
        return None
    n = int(m.group(1))
    return n if n > 1 else None


def _llama_install_root(binary: Optional[str]) -> Optional[Path]:
    """The llama.cpp install root with or without a marker: marker dir first,
    then UNSLOTH_LLAMA_CPP_PATH, then the nearest 'llama.cpp' ancestor of the
    binary, then studio_root()/llama.cpp (custom) or ~/.unsloth/llama.cpp
    (legacy) -- mirroring _find_llama_server_binary's roots."""
    marked = _install_dir_for(binary)
    if marked is not None:
        return marked
    env = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    if env:
        return Path(env)
    if binary:
        for parent in Path(binary).parents:
            if parent.name == "llama.cpp":
                return parent
    # A pinned LLAMA_SERVER_PATH always wins in _find_llama_server_binary, so a
    # prebuilt installed into the default root would be ignored: don't manage it.
    if os.environ.get("LLAMA_SERVER_PATH"):
        return None
    try:
        from utils.paths.storage_roots import studio_root

        sr = studio_root()
        legacy = Path.home() / ".unsloth" / "studio"
        try:
            is_legacy = sr.resolve() == legacy.resolve()
        except (OSError, ValueError):
            is_legacy = sr == legacy
        return (Path.home() / ".unsloth" / "llama.cpp") if is_legacy else sr / "llama.cpp"
    except Exception:  # pragma: no cover - defensive
        return Path.home() / ".unsloth" / "llama.cpp"


def _source_build_status(binary: str, *, force_refresh: bool) -> Optional[dict]:
    """Update status for a markerless (source-build) install: offer the official
    prebuilt when one exists for this host and is newer than the installed
    binary. None -> caller falls through to the no-marker default (unsupported)."""
    res = _resolve_prebuilt_for_host(force_refresh = force_refresh)
    if not res or not res.get("prebuilt_available"):
        return None
    # llama_tag is the upstream build (bNNNN, what --version reports); release_tag
    # can be a fork wrapper tag, so compare/display against llama_tag.
    latest = res.get("llama_tag") or res.get("release_tag")
    if not latest:
        return None
    # No resolvable install root (e.g. a pinned LLAMA_SERVER_PATH we cannot
    # manage) means an apply would not take effect, so do not offer.
    if _llama_install_root(binary) is None:
        return None
    installed_build = _installed_build_number(binary)
    m = re.search(r"(\d+)", latest)
    latest_build = int(m.group(1)) if m else None
    # Suppress only when the source build is reliably newer/equal; unknown
    # version (the involuntary source-build case) is treated as behind.
    update_available = (
        installed_build is None or latest_build is None or installed_build < latest_build
    )
    with _job_lock:
        job = dict(_job)
    return {
        "supported": True,
        "update_available": update_available,
        "stale": False,
        "installed_tag": (f"b{installed_build}" if installed_build else None),
        "latest_tag": latest,
        "published_repo": res.get("repo"),
        "installed_at_utc": None,
        "age_days": None,
        "source_build": True,
        "job": job,
    }


def get_update_status(*, force_refresh: bool = False) -> dict:
    """Report whether a newer prebuilt exists plus the current job state.

    force_refresh bypasses the 24h release cache for an explicit "check now".
    """
    binary = _find_binary()
    marker = read_install_marker(binary)

    with _job_lock:
        job_running = _job["state"] == _JOB_RUNNING

    # No marker = source build / custom path. Offer the official prebuilt if one
    # now exists for this host (this is why macOS source builds showed no button).
    # Skipped while the updater swaps the tree: each 3s poll would exec the
    # half-replaced binary (on Windows that exec can make the installer's
    # os.replace fail) and the poller only consumes job progress.
    if marker is None and binary is not None and not job_running:
        src = _source_build_status(binary, force_refresh = force_refresh)
        if src is not None:
            return src

    repo = (marker or {}).get("published_repo") or DEFAULT_PUBLISHED_REPO

    if force_refresh and repo:
        # Prime the cache so the freshness read below sees the newest tag.
        try:
            latest_published_release(repo, force_refresh = True)
        except Exception as exc:  # pragma: no cover - network defensive
            logger.debug("llama update: force refresh failed", error = str(exc))

    freshness = check_prebuilt_freshness(binary)
    installed = freshness.get("installed_tag")
    latest = freshness.get("latest_tag")
    update_available = bool(
        freshness.get("has_marker") and installed and latest and installed != latest
    )

    with _job_lock:
        job = dict(_job)

    return {
        "supported": bool(freshness.get("has_marker")),
        "update_available": update_available,
        "stale": bool(freshness.get("stale")),
        "installed_tag": installed,
        "latest_tag": latest,
        "published_repo": freshness.get("published_repo") or repo,
        "installed_at_utc": freshness.get("installed_at_utc"),
        "age_days": freshness.get("age_days"),
        "source_build": False,
        "job": job,
    }


def _rocm_install_args(asset: Optional[str]) -> list[str]:
    """Forward --rocm-gfx/--has-rocm from the marker asset, mirroring setup.sh.
    The installer probe can miss the gfx arch on amd-smi-only hosts; lemonade
    bundles carry the family in the name (rocm-gfx110X), fork bundles only rocm/hip."""
    if not asset:
        return []
    low = asset.lower()
    if "rocm" not in low and "hip" not in low:
        return []
    gfx = re.search(r"-gfx[0-9a-z]+", low)
    if gfx:
        # _normalize_forwarded_gfx accepts the family form (gfx110x -> gfx110X).
        return ["--rocm-gfx", gfx.group(0).lstrip("-")]
    return ["--has-rocm"]


def _run_update(install_dir: Path, repo: str, asset: Optional[str], script: Path) -> None:
    """Worker: put the backend into a maintenance state, run the installer for
    the latest prebuilt, then refresh caches so the next load uses the new build."""
    backend = None
    model_was_active = False
    try:
        # Maintenance state so no load starts a server from the half-swapped binary
        # (and the old binary is freed for the swap). Fails open without a backend.
        try:
            from routes.inference import get_llama_cpp_backend
            backend = get_llama_cpp_backend()
        except Exception as exc:
            logger.debug(
                "llama update: backend unavailable, skipping load coordination", error = str(exc)
            )
            backend = None

        if backend is not None:
            try:
                with backend._serial_load_lock:
                    backend._llama_update_in_progress = True
                    # is_active covers the loading/unhealthy window is_loaded misses
                    # (a live process also locks the exe on Windows during the swap).
                    if getattr(backend, "is_active", False):
                        model_was_active = True
                        backend.unload_model()
            except Exception as exc:
                logger.debug("llama update: load coordination failed", error = str(exc))

        cmd = [
            sys.executable,
            str(script),
            "--install-dir",
            str(install_dir),
            "--llama-tag",
            "latest",
            "--published-repo",
            repo,
        ]
        cmd.extend(_rocm_install_args(asset))
        logger.info("llama update: installing", cmd = " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output = True,
            text = True,
            timeout = _INSTALL_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
            raise RuntimeError(f"installer exited {proc.returncode}: {tail or 'no output'}")

        # New UNSLOTH_PREBUILT_INFO.json is on disk; drop caches so the next
        # status read reflects the freshly installed tag.
        reset_caches()
        new_marker = read_install_marker(_find_binary())
        new_tag = (new_marker or {}).get("tag") or (new_marker or {}).get("release_tag")

        with _job_lock:
            _job.update(
                state = _JOB_SUCCESS,
                message = (
                    f"Updated llama.cpp to {new_tag}."
                    + (" Reload your model to use it." if model_was_active else "")
                ),
                to_tag = new_tag,
                error = None,
                finished_at = _utcnow(),
            )
        logger.info("llama update: success", to_tag = new_tag)
    except Exception as exc:
        logger.warning("llama update: failed", error = str(exc))
        with _job_lock:
            _job.update(
                state = _JOB_ERROR,
                message = "llama.cpp update failed.",
                error = str(exc),
                finished_at = _utcnow(),
            )
    finally:
        # Lift the maintenance state so model loads work again, success or not.
        if backend is not None:
            try:
                backend._llama_update_in_progress = False
            except Exception:  # pragma: no cover - defensive
                pass


def start_update() -> dict:
    """Kick off a background update. Idempotent: a second call while one is
    running returns the in-flight job rather than starting another."""
    binary = _find_binary()
    marker = read_install_marker(binary)
    script = _installer_script()
    if script is None:
        return {
            "started": False,
            "reason": "installer_missing",
            "message": "install_llama_prebuilt.py could not be located.",
            "job": get_update_status()["job"],
        }

    if marker:
        install_dir = _install_dir_for(binary)
        repo = marker.get("published_repo") or DEFAULT_PUBLISHED_REPO
        from_tag = marker.get("tag") or marker.get("release_tag")
        asset = marker.get("asset")
    else:
        # Source build / custom path: install the official prebuilt in place if
        # one exists for this host (asset carries the ROCm gfx for forwarding).
        res = _resolve_prebuilt_for_host()
        if not res or not res.get("prebuilt_available"):
            return {
                "started": False,
                "reason": "no_prebuilt_available",
                "message": (
                    "No official llama.cpp prebuilt is available for this host, "
                    "so the source build cannot be swapped automatically."
                ),
                "job": get_update_status()["job"],
            }
        install_dir = _llama_install_root(binary)
        repo = res.get("repo") or DEFAULT_PUBLISHED_REPO
        from_tag = None
        asset = res.get("asset")

    if install_dir is None:
        return {
            "started": False,
            "reason": "no_install_dir",
            "message": "Could not determine the llama.cpp install directory.",
            "job": get_update_status()["job"],
        }

    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}
        _job.update(
            state = _JOB_RUNNING,
            message = "Downloading and installing the latest llama.cpp prebuilt...",
            from_tag = from_tag,
            to_tag = None,
            error = None,
            started_at = _utcnow(),
            finished_at = None,
        )
        job_snapshot = dict(_job)

    thread = threading.Thread(
        target = _run_update,
        args = (install_dir, repo, asset, script),
        name = "llama-cpp-update",
        daemon = True,
    )
    thread.start()
    return {"started": True, "reason": None, "job": job_snapshot}


def _reset_job_for_tests() -> None:
    """Test-only: return the job tracker to idle."""
    with _job_lock:
        _job.update(
            state = _JOB_IDLE,
            message = "",
            from_tag = None,
            to_tag = None,
            error = None,
            started_at = None,
            finished_at = None,
        )
