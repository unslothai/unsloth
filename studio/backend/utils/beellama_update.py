# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""beellama.cpp source-build update check + apply.

Replaces the prebuilt llama.cpp upgrade notifier for installs produced by
``install_beellama_source.py`` (which drops ``beellama-source.json`` into the
install dir recording the built commit, ref, repo, backend and FA-quant flags).

Detection: compare the built commit against the HEAD of the configured branch
on GitHub (the ``compare`` API's ``ahead_by`` is "how many new commits since you
built"). Apply: re-run ``install_beellama_source.py`` against the same install
dir with the SAME flags recorded at build time (backend, ref, repo, FA-quant
mode, rocWMMA), then swap the freshly built tree in place.

Installs that are NOT beellama source builds (no ``beellama-source.json``) fall
through to the existing prebuilt updater in ``utils.llama_cpp_update`` so the
upstream path is unchanged. Everything fails open: a missing marker / offline
GitHub reports ``update_available=False`` and never blocks the app.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import structlog

# Reuse the active-binary resolver so we update exactly what Studio runs.
from utils.llama_cpp_update import _find_binary
from utils.process_lifetime import child_popen_kwargs

logger = structlog.get_logger(__name__)

# Written by install_beellama_source.py (its METADATA_FILENAME).
_METADATA_FILENAME = "beellama-source.json"
# Defaults mirror install_beellama_source.py when the marker omits them.
_DEFAULT_REPO = "https://github.com/Anbeeld/beellama.cpp"
_DEFAULT_REF = "v0.3.2"

# Source compiles (especially a CUDA build with the TurboQuant FA kernels) are
# far slower than a prebuilt download -- give the rebuild a generous ceiling.
_INSTALL_TIMEOUT_SECONDS = 3600

# 24h memo keeps the GitHub compare call off the hot path / within rate limits.
_COMPARE_CACHE_TTL_SECONDS = 24 * 60 * 60

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
    "progress": None,
    "started_at": None,
    "finished_at": None,
}

# cmake build progress lines, e.g. "[ 42%] Building CXX object ...".
_BUILD_PROGRESS_RE = re.compile(r"\[\s*(\d+)%\]")

# key "slug|base|head" -> (fetched_at, result|None). In-memory only; a restart
# just re-fetches.
_compare_memo: dict[str, tuple[float, Optional[dict]]] = {}


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short(sha: Optional[str], n: int = 12) -> Optional[str]:
    return sha[:n] if isinstance(sha, str) and sha else None


def _beellama_install(binary: Optional[str]) -> tuple[Optional[Path], Optional[dict]]:
    """Walk up from the active binary to the install dir holding
    ``beellama-source.json``. Returns (install_dir, marker) or (None, None) when
    this is not a beellama source build. (binary lives at
    ``<dir>/build/bin[/Release]/llama-server`` so the marker is 2-4 parents up.)"""
    if not binary:
        return None, None
    p = Path(binary)
    for parent in p.parents[:6]:
        candidate = parent / _METADATA_FILENAME
        if candidate.is_file():
            try:
                return parent, json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("beellama update: marker unreadable", path = str(candidate), error = str(exc))
                return parent, None
    return None, None


def _repo_slug(repo_url: Optional[str]) -> Optional[str]:
    """``https://github.com/Anbeeld/beellama.cpp(.git)`` -> ``Anbeeld/beellama.cpp``."""
    if not repo_url:
        return None
    s = repo_url.strip()
    if s.endswith(".git"):
        s = s[:-4]
    m = re.search(r"github\.com[:/]+([^/]+/[^/]+?)/?$", s)
    return m.group(1) if m else None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "unsloth-studio-beellama-update",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_compare(slug: str, base: str, head: str, *, timeout: float = 6.0) -> Optional[dict]:
    """GitHub compare ``base...head``. Returns {ahead_by, latest_sha, status} or
    None on any failure. ``ahead_by`` = commits on ``head`` not in ``base`` =
    new commits since the build. ``latest_sha`` = head tip (from the commit list;
    capped at GitHub's 250-commit page, which only matters when 250+ behind)."""
    url = f"https://api.github.com/repos/{slug}/compare/{base}...{head}"
    req = urllib.request.Request(url, headers = _github_headers())
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
        logger.debug("beellama update: compare fetch failed", slug = slug, error = str(exc))
        return None
    if not isinstance(data, dict):
        return None
    ahead = data.get("ahead_by")
    commits = data.get("commits") if isinstance(data.get("commits"), list) else []
    latest_sha = base
    if commits:
        tip = commits[-1]
        if isinstance(tip, dict) and isinstance(tip.get("sha"), str):
            latest_sha = tip["sha"]
    return {
        "ahead_by": int(ahead) if isinstance(ahead, int) else None,
        "latest_sha": latest_sha,
        "status": data.get("status"),
    }


def _cached_compare(slug: str, base: str, head: str, *, force_refresh: bool) -> Optional[dict]:
    key = f"{slug}|{base}|{head}"
    now = time.time()
    if not force_refresh:
        memo = _compare_memo.get(key)
        if memo and now - memo[0] < _COMPARE_CACHE_TTL_SECONDS:
            return memo[1]
    result = _github_compare(slug, base, head)
    if result is None:
        memo = _compare_memo.get(key)
        return memo[1] if memo else None
    _compare_memo[key] = (now, result)
    return result


def _empty_status(job: dict) -> dict:
    """A beellama install we recognise but can't compare (missing commit / repo,
    or offline): supported, nothing offered."""
    return {
        "supported": True,
        "update_available": False,
        "stale": False,
        "installed_tag": None,
        "latest_tag": None,
        "published_repo": None,
        "installed_at_utc": None,
        "age_days": None,
        "source_build": True,
        "update_size_bytes": None,
        "commits_behind": None,
        "installed_commit": None,
        "latest_commit": None,
        "branch": None,
        "compare_url": None,
        "job": job,
    }


def _beellama_status(install_dir: Path, marker: dict, *, force_refresh: bool) -> dict:
    with _job_lock:
        job = dict(_job)

    built_commit = marker.get("commit") if isinstance(marker.get("commit"), str) else None
    ref = marker.get("ref") or _DEFAULT_REF
    slug = _repo_slug(marker.get("repo") or _DEFAULT_REPO)

    if not built_commit or not slug:
        out = _empty_status(job)
        out["installed_commit"] = built_commit
        out["installed_tag"] = _short(built_commit)
        out["branch"] = ref
        return out

    compare = _cached_compare(slug, built_commit, ref, force_refresh = force_refresh)
    commits_behind = compare.get("ahead_by") if compare else None
    latest_commit = compare.get("latest_sha") if compare else None
    update_available = bool(commits_behind and commits_behind > 0)
    # Always link the branch HEAD ref so the compare stays correct even if the
    # 250-commit page truncated latest_sha.
    compare_url = f"https://github.com/{slug}/compare/{_short(built_commit)}...{ref}"

    return {
        "supported": True,
        "update_available": update_available,
        "stale": False,
        "installed_tag": _short(built_commit),
        "latest_tag": _short(latest_commit) if update_available else _short(built_commit),
        "published_repo": slug,
        "installed_at_utc": None,
        "age_days": None,
        "source_build": True,
        "update_size_bytes": None,
        "commits_behind": commits_behind,
        "installed_commit": built_commit,
        "latest_commit": latest_commit,
        "branch": ref,
        "compare_url": compare_url,
        "job": job,
    }


def get_update_status(*, force_refresh: bool = False) -> dict:
    """beellama update status when the active install is a beellama source build;
    otherwise delegate to the prebuilt updater so non-beellama installs are
    unchanged."""
    binary = _find_binary()
    install_dir, marker = _beellama_install(binary)
    if marker is None:
        from utils.llama_cpp_update import get_update_status as _prebuilt_status
        return _prebuilt_status(force_refresh = force_refresh)
    return _beellama_status(install_dir, marker, force_refresh = force_refresh)


def get_installed_beellama_version(binary: Optional[str] = None) -> Optional[str]:
    """Display string for a beellama source build, e.g. ``beellama v0.3.2
    (a1b2c3d)``, or None when the active install is not a beellama build.

    Reads only the local ``beellama-source.json`` (no network), so it is safe for
    latency-sensitive surfaces like the About panel. ``binary`` lets the caller
    pass an already-resolved path (so it honours the same resolver/mocking);
    omitted, it resolves the active binary itself. Returns None while a rebuild
    is swapping the tree (the marker may be momentarily absent)."""
    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return None
    if binary is None:
        binary = _find_binary()
    _install_dir, marker = _beellama_install(binary)
    if not marker:
        return None
    commit = _short(marker.get("commit"), 7)
    ref = marker.get("ref") or _DEFAULT_REF
    return f"beellama {ref} ({commit})" if commit else f"beellama {ref}"


def _installer_script() -> Optional[Path]:
    """Locate install_beellama_source.py. Honours UNSLOTH_BEELLAMA_INSTALLER, then
    searches up from this file for ``<root>/install_beellama_source.py`` and
    ``<root>/studio/install_beellama_source.py`` (dev tree + installed layout)."""
    env = os.environ.get("UNSLOTH_BEELLAMA_INSTALLER")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for up in here.parents:
        for cand in (up / "install_beellama_source.py", up / "studio" / "install_beellama_source.py"):
            if cand.is_file():
                return cand
    return None


def _install_env(marker: dict) -> dict[str, str]:
    """Reconstruct the build-time environment from the marker so the rebuild uses
    the SAME flags: FA-quant mode and the opt-in rocWMMA toggle."""
    env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
    fa_quants = marker.get("fa_quants")
    # fa_quants is half/all (turbo4 compiled) or None (not compiled / cpu/metal).
    env["UNSLOTH_BEELLAMA_FA_QUANTS"] = fa_quants if fa_quants in ("half", "all") else "off"
    cmake_args = marker.get("cmake_args") or []
    if isinstance(cmake_args, list) and "-DGGML_HIP_ROCWMMA_FATTN=ON" in cmake_args:
        env["UNSLOTH_BEELLAMA_HIP_ROCWMMA_FATTN"] = "1"
    return env


def _run_update(install_dir: Path, script: Path, marker: dict) -> None:
    """Worker: put the backend into maintenance, re-run the source build with the
    recorded flags, then clear the compare cache so the banner reflects the new
    commit."""
    backend_obj = None
    model_was_active = False
    try:
        try:
            from routes.inference import get_llama_cpp_backend
            backend_obj = get_llama_cpp_backend()
        except Exception as exc:
            logger.debug("beellama update: backend unavailable", error = str(exc))
            backend_obj = None

        if backend_obj is not None:
            try:
                with backend_obj._serial_load_lock:
                    backend_obj._llama_update_in_progress = True
                    # A live server locks the .exe on Windows, which would make
                    # the installer's dir swap fail -- unload first.
                    if getattr(backend_obj, "is_active", False):
                        model_was_active = True
                        backend_obj.unload_model()
            except Exception as exc:
                logger.debug("beellama update: load coordination failed", error = str(exc))

        cmd = [
            sys.executable,
            str(script),
            "--install-dir",
            str(install_dir),
            "--backend",
            str(marker.get("backend") or "auto"),
            "--ref",
            str(marker.get("ref") or _DEFAULT_REF),
            "--repo",
            str(marker.get("repo") or _DEFAULT_REPO),
        ]
        env = _install_env(marker)
        logger.info("beellama update: rebuilding", cmd = " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env = env,
            **child_popen_kwargs(),
        )
        timed_out = threading.Event()

        def _kill_on_timeout() -> None:
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(_INSTALL_TIMEOUT_SECONDS, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()
        tail_lines: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                tail_lines.append(line)
                if len(tail_lines) > 80:
                    del tail_lines[0]
                m = _BUILD_PROGRESS_RE.search(line)
                if m is None:
                    continue
                fraction = min(float(m.group(1)) / 100.0, 1.0)
                with _job_lock:
                    _job["progress"] = max(_job.get("progress") or 0.0, fraction)
            returncode = proc.wait()
        finally:
            watchdog.cancel()
        if timed_out.is_set():
            raise RuntimeError(f"build timed out after {_INSTALL_TIMEOUT_SECONDS}s")
        if returncode != 0:
            tail = "".join(tail_lines).strip()[-1500:]
            raise RuntimeError(f"installer exited {returncode}: {tail or 'no output'}")

        # New beellama-source.json is on disk; drop the compare cache so the next
        # status read recomputes against the freshly built commit (-> 0 behind).
        _compare_memo.clear()
        _new_install, new_marker = _beellama_install(_find_binary())
        new_commit = _short((new_marker or {}).get("commit")) if new_marker else None

        with _job_lock:
            _job.update(
                state = _JOB_SUCCESS,
                message = (
                    f"Rebuilt beellama.cpp at {new_commit or 'the latest commit'}."
                    + (" Reload your model to use it." if model_was_active else "")
                ),
                to_tag = new_commit,
                error = None,
                progress = 1.0,
                finished_at = _utcnow(),
            )
        logger.info("beellama update: success", to_commit = new_commit)
    except Exception as exc:
        logger.warning("beellama update: failed", error = str(exc))
        with _job_lock:
            _job.update(
                state = _JOB_ERROR,
                message = "beellama.cpp rebuild failed.",
                error = str(exc),
                finished_at = _utcnow(),
            )
    finally:
        if backend_obj is not None:
            try:
                backend_obj._llama_update_in_progress = False
            except Exception:  # pragma: no cover - defensive
                pass


def start_update() -> dict:
    """Kick off a background beellama rebuild with the recorded flags. Delegates
    to the prebuilt updater for non-beellama installs. Idempotent: a second call
    while one runs returns the in-flight job."""
    binary = _find_binary()
    install_dir, marker = _beellama_install(binary)
    if marker is None:
        from utils.llama_cpp_update import start_update as _prebuilt_start
        return _prebuilt_start()

    script = _installer_script()
    if script is None:
        return {
            "started": False,
            "reason": "installer_missing",
            "message": "install_beellama_source.py could not be located.",
            "job": get_update_status()["job"],
        }

    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}

    # Fresh check so a stale 24h cache can't wrongly block a real update, and a
    # direct POST can't rebuild when already current.
    status = _beellama_status(install_dir, marker, force_refresh = True)
    if not status.get("update_available"):
        return {
            "started": False,
            "reason": "up_to_date",
            "message": "beellama.cpp is already at the latest commit on this branch.",
            "job": status["job"],
        }
    if install_dir is None:
        return {
            "started": False,
            "reason": "no_install_dir",
            "message": "Could not determine the beellama.cpp install directory.",
            "job": status["job"],
        }

    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}
        _job.update(
            state = _JOB_RUNNING,
            message = "Rebuilding beellama.cpp from source with your build flags...",
            from_tag = status.get("installed_tag"),
            to_tag = None,
            error = None,
            progress = 0.0,
            started_at = _utcnow(),
            finished_at = None,
        )
        job_snapshot = dict(_job)

    thread = threading.Thread(
        target = _run_update,
        args = (install_dir, script, marker),
        name = "beellama-update",
        daemon = True,
    )
    thread.start()
    return {"started": True, "reason": None, "job": job_snapshot}


def _reset_job_for_tests() -> None:
    """Test-only: return the job tracker to idle and clear the compare cache."""
    with _job_lock:
        _job.update(
            state = _JOB_IDLE,
            message = "",
            from_tag = None,
            to_tag = None,
            error = None,
            progress = None,
            started_at = None,
            finished_at = None,
        )
    _compare_memo.clear()
