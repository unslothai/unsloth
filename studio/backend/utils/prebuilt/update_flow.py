# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared mechanics of the llama.cpp / whisper.cpp in-app prebuilt updates.

The component modules (utils.llama_cpp_update / utils.whisper_cpp_update) keep
their public names, job dicts, and update policy (version comparison, pinning,
pre/post install steps); everything mechanical (managed-root resolution,
local-link detection, the resolve probe, the streamed installer run) lives here,
parameterized so the modules' monkeypatch seams keep working.
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
from typing import Callable, Optional

import structlog

from utils.process_lifetime import child_popen_kwargs

logger = structlog.get_logger(__name__)

# Markerless (source-build) resolve answers are memoized for 24h; only
# successful answers are cached so a network blip retries.
RESOLVE_TTL_SECONDS = 24 * 60 * 60

# Matches the installer's download progress lines, e.g.
# "Downloading x.zip:  35.0% (12.3 MiB/35.1 MiB) at 8.2 MiB/s".
PROGRESS_LINE_RE = re.compile(r"(\d+(?:\.\d+)?)%\s*\(")
# The download dominates the update; extract/validate fill the last slice.
DOWNLOAD_PROGRESS_CEILING = 0.95

JOB_IDLE = "idle"
JOB_RUNNING = "running"
JOB_SUCCESS = "success"
JOB_ERROR = "error"

# Per-phase states inside a chained job's "phases" breakdown.
PHASE_PENDING = "pending"
PHASE_RUNNING = "running"
PHASE_SUCCESS = "success"
PHASE_ERROR = "error"
PHASE_SKIPPED = "skipped"

_IDLE_JOB_FIELDS = dict(
    state = JOB_IDLE,
    message = "",
    from_tag = None,
    to_tag = None,
    reload_required = None,
    error = None,
    progress = None,
    started_at = None,
    finished_at = None,
    phases = None,
)


def new_job() -> dict:
    """A fresh idle job-state dict (one per component module)."""
    return dict(_IDLE_JOB_FIELDS)


def reset_job(job: dict, job_lock: threading.Lock) -> None:
    """Return a job tracker to idle (test seam)."""
    with job_lock:
        job.update(_IDLE_JOB_FIELDS)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_under(path: Path, root: Path) -> bool:
    try:
        p, r = path.resolve(), root.resolve()
    except (OSError, ValueError):
        p, r = path, root
    return p == r or r in p.parents


def install_dir_for(binary_path: Optional[str], *, marker_name: str) -> Optional[Path]:
    """The directory holding the install marker: the install root the installer
    wrote and the one we re-install into. Walks up from the binary like the
    freshness marker reader does."""
    if not binary_path:
        return None
    p = Path(binary_path)
    for parent in p.parents[:5]:
        if (parent / marker_name).is_file():
            return parent
    return None


def find_installer_script(*, env_var: str, script_name: str) -> Optional[Path]:
    """Locate the installer script. Honours the env override, then searches up
    from this file for both ``<root>/<script>`` and ``<root>/studio/<script>`` so
    it works in the dev tree and in an installed Unsloth layout."""
    env = os.environ.get(env_var)
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for up in here.parents:
        for cand in (up / script_name, up / "studio" / script_name):
            if cand.is_file():
                return cand
    return None


def resolve_prebuilt_for_host(
    *,
    force_refresh: bool,
    memo: dict,
    installer_script: Callable[[], Optional[Path]],
    log_message: str,
) -> Optional[dict]:
    """Run ``<installer> --resolve-prebuilt latest --output-format json`` (no
    download); return the parsed payload or None. Fail-open: any error -> None so
    a source build never blocks the app."""
    now = time.time()
    if not force_refresh and memo:
        if now - memo.get("at", 0.0) < RESOLVE_TTL_SECONDS:
            return memo.get("value")
    script = installer_script()
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
        logger.debug(log_message, error = str(exc))
        value = None
    if value is not None:  # cache real answers; let failures retry next poll
        memo.update(at = now, value = value)
    return value


def is_external_link(path: Optional[Path]) -> bool:
    """True when ``path`` is a locally-linked component dir: a POSIX symlink or a
    Windows junction / reparse point. Such a link resolves into the user's own
    checkout, so Unsloth must never auto-update it."""
    if path is None:
        return False
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    if os.name == "nt":
        try:
            import stat
            attrs = os.lstat(path).st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, AttributeError):
            return False
    return False


def active_install_is_local_link(binary: Optional[str], *, dir_name: str) -> bool:
    """True when the active server binary resolves through a locally-linked
    component directory. An update would write through that link into the user's
    checkout (or fail), so the install is treated as externally managed: none is
    offered or applied. Checks only up to and including the component dir so a
    symlinked HOME / studio root above it can't trip a false positive."""
    if not binary:
        return False
    for parent in Path(binary).parents:
        if is_external_link(parent):
            return True
        if parent.name == dir_name:
            break
    return False


def managed_install_root(
    binary: Optional[str],
    *,
    marker_root: Optional[Path],
    server_path_var: str,
    cpp_path_var: str,
    dir_name: str,
) -> Optional[Path]:
    """The Unsloth-managed component root the active binary lives under, or None
    when unmanaged. Installing where the active binary is not would not replace
    what discovery runs (a pinned server path, then the custom dir, then a
    component tree), so we refuse rather than install into an inactive or foreign
    tree."""
    if marker_root is not None:
        return marker_root
    if not binary:
        return None
    # The server-path pin is an explicit user choice that wins in discovery; never
    # auto-replace its tree (even the user's own checkout).
    if os.environ.get(server_path_var):
        return None
    p = Path(binary)
    env = os.environ.get(cpp_path_var)
    if env and is_under(p, Path(env)):
        return Path(env)
    for parent in p.parents:
        if parent.name == dir_name:
            return parent
    # PATH / system / custom install: not a managed tree, so do not offer.
    return None


def local_link_status(job: dict, job_lock: threading.Lock) -> dict:
    """Status payload for a local-link install: unmanaged, no update offered."""
    with job_lock:
        snapshot = dict(job)
    return {
        "supported": False,
        "update_available": False,
        "stale": False,
        "installed_tag": None,
        "latest_tag": None,
        "published_repo": None,
        "installed_at_utc": None,
        "age_days": None,
        "source_build": False,
        "local_link": True,
        "update_size_bytes": None,
        "job": snapshot,
    }


def rocm_install_args(asset: Optional[str]) -> list[str]:
    """Forward --rocm-gfx/--has-rocm from the marker asset, mirroring setup.sh.
    The installer probe can miss the gfx arch on amd-smi-only hosts; per-gfx ROCm
    bundles carry the family in the name (rocm-gfx110X), version-tagged bundles
    only rocm/hip."""
    if not asset:
        return []
    low = asset.lower()
    if "rocm" not in low and "hip" not in low:
        return []
    gfx = re.search(r"-gfx[0-9a-z]+", low)
    if gfx:
        return ["--rocm-gfx", gfx.group(0).lstrip("-")]
    return ["--has-rocm"]


def stream_installer(
    cmd: list[str],
    env: dict[str, str],
    *,
    timeout_seconds: int,
    job: Optional[dict] = None,
    job_lock: Optional[threading.Lock] = None,
    set_progress: Optional[Callable[[float], None]] = None,
) -> None:
    """Run the installer, streaming its progress lines into job["progress"]
    (or through set_progress when given, e.g. a chained-phase progress window).
    Raises RuntimeError on timeout or a nonzero exit (with an output tail)."""
    if set_progress is None:
        assert job is not None and job_lock is not None

        def set_progress(fraction: float) -> None:
            with job_lock:
                job["progress"] = max(job.get("progress") or 0.0, fraction)

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

    watchdog = threading.Timer(timeout_seconds, _kill_on_timeout)
    watchdog.daemon = True
    watchdog.start()
    tail_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail_lines.append(line)
            if len(tail_lines) > 80:
                del tail_lines[0]
            m = PROGRESS_LINE_RE.search(line)
            if m is None:
                continue
            set_progress(min(float(m.group(1)) / 100.0, 1.0) * DOWNLOAD_PROGRESS_CEILING)
        returncode = proc.wait()
    finally:
        watchdog.cancel()
    if timed_out.is_set():
        raise RuntimeError(f"installer timed out after {timeout_seconds}s")
    if returncode != 0:
        tail = "".join(tail_lines).strip()[-1500:]
        raise RuntimeError(f"installer exited {returncode}: {tail or 'no output'}")


def _new_phase_record(spec: dict) -> dict:
    """Initial breakdown entry for one phase of a chained job."""
    runnable = spec.get("run") is not None
    return {
        "state": PHASE_PENDING if runnable else PHASE_SKIPPED,
        "reason": None if runnable else spec.get("skip_reason"),
        "progress": None,
        "to_tag": None,
        "reload_required": None,
        "message": "",
        "error": None,
    }


def run_chained_update(phases: list[dict], *, job: dict, job_lock: threading.Lock) -> None:
    """Run update phases in order into one shared job dict (the worker of a
    combined llama+whisper apply).

    Each phase spec: ``name`` (breakdown key), ``weight`` (progress slice,
    normalized over runnable phases), ``run`` (callable(set_progress) -> result
    dict with to_tag/reload_required/message, raises on failure; None = skipped)
    and ``skip_reason`` / ``failure_message``. A failing phase aborts the chain:
    later phases are marked skipped (reason "aborted") and the job goes to error,
    keeping the reload_required and messages of already-succeeded phases so a
    partial success stays visible."""
    runnable = [p for p in phases if p.get("run") is not None]
    total_weight = sum(float(p.get("weight") or 1.0) for p in runnable) or 1.0
    with job_lock:
        job["phases"] = {p["name"]: _new_phase_record(p) for p in phases}

    offset = 0.0
    done_messages: list[str] = []
    reload_required = False
    primary_to_tag: Optional[str] = None
    for index, phase in enumerate(phases):
        if phase.get("run") is None:
            continue
        name = phase["name"]
        weight = float(phase.get("weight") or 1.0) / total_weight
        with job_lock:
            job["phases"][name].update(state = PHASE_RUNNING, progress = 0.0)

        def set_progress(
            fraction: float,
            *,
            _name: str = name,
            _base: float = offset,
            _slice: float = weight,
        ) -> None:
            f = max(0.0, min(float(fraction), 1.0))
            with job_lock:
                record = job["phases"][_name]
                record["progress"] = max(record.get("progress") or 0.0, f)
                job["progress"] = max(job.get("progress") or 0.0, _base + f * _slice)

        try:
            result = phase["run"](set_progress) or {}
        except Exception as exc:
            failure = phase.get("failure_message") or f"{name} update failed."
            with job_lock:
                job["phases"][name].update(state = PHASE_ERROR, error = str(exc))
                for later in phases[index + 1 :]:
                    if later.get("run") is not None:
                        job["phases"][later["name"]].update(state = PHASE_SKIPPED, reason = "aborted")
                # A partial success keeps its messages and reload_required so the
                # caller sees the earlier phase did land.
                job.update(
                    state = JOB_ERROR,
                    message = " ".join(done_messages + [failure]),
                    error = str(exc),
                    finished_at = utcnow(),
                )
                if done_messages:
                    job["reload_required"] = reload_required
            return
        set_progress(1.0)
        offset += weight
        with job_lock:
            job["phases"][name].update(
                state = PHASE_SUCCESS,
                to_tag = result.get("to_tag"),
                reload_required = result.get("reload_required"),
                message = result.get("message") or "",
            )
        if result.get("message"):
            done_messages.append(result["message"])
        # Only phases affecting the primary (llama) server may raise the job-level
        # reload flag: the frontend resyncs chat model state off it, and a
        # whisper-only sidecar reload must not clear the chat checkpoint. Per-phase
        # reload_required stays visible under job["phases"].
        if phase.get("affects_job_reload", True):
            reload_required = reload_required or bool(result.get("reload_required"))
        if primary_to_tag is None:
            primary_to_tag = result.get("to_tag")

    with job_lock:
        job.update(
            state = JOB_SUCCESS,
            message = " ".join(done_messages) or "Already up to date.",
            to_tag = primary_to_tag,
            reload_required = reload_required,
            error = None,
            progress = 1.0,
            finished_at = utcnow(),
        )
