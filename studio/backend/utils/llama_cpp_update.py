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

import os
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


def get_update_status(*, force_refresh: bool = False) -> dict:
    """Report whether a newer prebuilt exists plus the current job state.

    force_refresh bypasses the 24h release cache for an explicit "check now".
    """
    binary = _find_binary()
    marker = read_install_marker(binary)
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
        "job": job,
    }


def _run_update(install_dir: Path, repo: str, from_tag: Optional[str], script: Path) -> None:
    """Worker: unload the running model, run the installer, refresh caches."""
    model_was_loaded = False
    try:
        # Release the binary so the atomic swap can replace files in use.
        try:
            from routes.inference import get_llama_cpp_backend

            backend = get_llama_cpp_backend()
            model_was_loaded = bool(getattr(backend, "is_loaded", False))
            if model_was_loaded:
                backend.unload_model()
        except Exception as exc:
            logger.debug("llama update: unload before install failed", error = str(exc))

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
        logger.info("llama update: installing", cmd = " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output = True,
            text = True,
            timeout = _INSTALL_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
            raise RuntimeError(
                f"installer exited {proc.returncode}: {tail or 'no output'}"
            )

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
                    + (" Reload your model to use it." if model_was_loaded else "")
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


def start_update() -> dict:
    """Kick off a background update. Idempotent: a second call while one is
    running returns the in-flight job rather than starting another."""
    binary = _find_binary()
    install_dir = _install_dir_for(binary)
    marker = read_install_marker(binary)
    if install_dir is None or not marker:
        return {
            "started": False,
            "reason": "no_prebuilt_marker",
            "message": (
                "This llama.cpp install was not provisioned from an Unsloth "
                "prebuilt (source build or custom path); in-app update is "
                "unavailable."
            ),
            "job": get_update_status()["job"],
        }
    script = _installer_script()
    if script is None:
        return {
            "started": False,
            "reason": "installer_missing",
            "message": "install_llama_prebuilt.py could not be located.",
            "job": get_update_status()["job"],
        }
    repo = marker.get("published_repo") or DEFAULT_PUBLISHED_REPO
    from_tag = marker.get("tag") or marker.get("release_tag")

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
        args = (install_dir, repo, from_tag, script),
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
