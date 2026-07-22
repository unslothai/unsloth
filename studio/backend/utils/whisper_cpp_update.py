# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app whisper.cpp prebuilt update.

Builds on utils.whisper_cpp_freshness (which detects whether a newer prebuilt
release exists) and adds the *apply* half: run install_whisper_prebuilt.py to
download the newest bundle for this host and atomically swap it in place, so
the next model load uses it.

Design notes:
- Detection is delegated to check_prebuilt_freshness(). We surface an
  ``update_available`` flag (installed_tag != latest_tag) which is laxer than
  freshness' ``stale`` (which additionally requires the install to be >= 3 days
  old). The UI shows the "Update whisper.cpp" affordance on update_available.
- The install is slow (download + extract + validate), so it runs on a daemon
  thread; callers poll get_update_status() for the job state.
- Everything fails open: a missing marker / offline GitHub / source build just
  reports update_available=False and never blocks the app.
- The mechanics (managed-root resolution, local-link detection, the resolve
  probe, the streamed installer run) live in utils.prebuilt.update_flow; this
  module keeps the whisper policy and the job dict its callers poll.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import structlog

from utils.prebuilt import update_flow as _flow
from utils.whisper_cpp_freshness import (
    _INSTALL_MARKER_NAME,
    check_prebuilt_freshness,
    latest_published_release,
    latest_release_assets,
    parse_release_version,
    read_install_marker,
    reset_caches,
    update_download_size_bytes,
)

logger = structlog.get_logger(__name__)

DEFAULT_PUBLISHED_REPO = "unslothai/whisper.cpp"
_INSTALL_TIMEOUT_SECONDS = 1800  # 30 min ceiling for download + extract/validate

# Background job state. Single in-flight update at a time, guarded by _job_lock.
_JOB_IDLE = _flow.JOB_IDLE
_JOB_RUNNING = _flow.JOB_RUNNING
_JOB_SUCCESS = _flow.JOB_SUCCESS
_JOB_ERROR = _flow.JOB_ERROR

_job_lock = threading.Lock()
_job: dict = _flow.new_job()

_utcnow = _flow.utcnow
_is_under = _flow.is_under
_is_external_link = _flow.is_external_link
_rocm_install_args = _flow.rocm_install_args


def _find_binary() -> Optional[str]:
    """Locate the active whisper-server binary via the STT sidecar's own
    resolver, so update targets exactly what Unsloth runs. Lazy import keeps the
    heavy inference module off this module's import path."""
    try:
        from core.inference.stt_ggml_sidecar import find_whisper_server_binary
        return find_whisper_server_binary()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("whisper update: binary discovery failed", error = str(exc))
        return None


def _install_dir_for(binary_path: Optional[str]) -> Optional[Path]:
    """The directory holding UNSLOTH_WHISPER_PREBUILT_INFO.json -- i.e. the
    install root install_whisper_prebuilt.py wrote (the ``<install-dir>`` whose
    canonical server is ``build/bin/whisper-server``) and the one we re-install
    into."""
    return _flow.install_dir_for(binary_path, marker_name = _INSTALL_MARKER_NAME)


def _installer_script() -> Optional[Path]:
    """Locate install_whisper_prebuilt.py (UNSLOTH_WHISPER_INSTALLER wins)."""
    return _flow.find_installer_script(
        env_var = "UNSLOTH_WHISPER_INSTALLER", script_name = "install_whisper_prebuilt.py"
    )


# Markerless (source-build) installs have no UNSLOTH_WHISPER_PREBUILT_INFO.json, so
# we ask the installer whether an official prebuilt now exists for this host.
_resolve_memo: dict = {}


def _resolve_prebuilt_for_host(*, force_refresh: bool = False) -> Optional[dict]:
    """Run install_whisper_prebuilt.py --resolve-prebuilt (no download) and return
    {prebuilt_available, repo, release_tag, upstream_tag, backend, asset, os,
    arch, ...} or None. Fail-open: any error -> None so a source build never
    blocks the app."""
    return _flow.resolve_prebuilt_for_host(
        force_refresh = force_refresh,
        memo = _resolve_memo,
        installer_script = lambda: _installer_script(),
        log_message = "whisper update: resolve-prebuilt failed",
    )


def _installed_whisper_version(binary: Optional[str]) -> Optional[str]:
    """Best-effort ``v<A.B.C>`` from ``whisper-server --version``. None when the
    binary is missing, does not report a version, or cannot be run. Used only for
    the markerless source-build downgrade guard, so it fails open to None."""
    if not binary:
        return None
    try:
        proc = subprocess.run([binary, "--version"], capture_output = True, text = True, timeout = 20)
    except Exception:  # pragma: no cover - defensive
        return None
    m = re.search(r"v?(\d+\.\d+\.\d+)", (proc.stderr or "") + (proc.stdout or ""))
    if not m:
        return None
    return f"v{m.group(1)}"


def get_installed_whisper_version() -> Optional[str]:
    """Display string for the active whisper.cpp install (e.g. 'v1.9.1-unsloth.2'),
    or None.

    Prefers the install marker's release_tag -- the full unsloth release identity,
    the same field the update banner compares as installed -- so a mix/serial
    build reads back in full. Last resort is ``v<A.B.C>`` parsed from
    ``whisper-server --version`` for source/custom builds that have no marker.

    Lightweight: reads the local marker and at most runs ``--version``. Does no
    network or release-freshness work (unlike get_update_status), so it is safe
    to call from latency-sensitive paths like the About panel.
    """
    binary = _find_binary()
    marker = read_install_marker(binary)
    if marker:
        tag = marker.get("release_tag")
        if tag:
            return tag
    # Markerless/source build: the fallback execs ``whisper-server --version``.
    # Skip it while an update is swapping the tree -- on Windows that exec can
    # make the installer's os.replace fail (the same race get_update_status's
    # source-build probe guards against). The panel just omits the row.
    with _job_lock:
        job_running = _job["state"] == _JOB_RUNNING
    if job_running:
        return None
    return _installed_whisper_version(binary)


def _whisper_install_root(binary: Optional[str]) -> Optional[Path]:
    """The Unsloth-managed whisper.cpp root the active binary lives under, or None
    when the binary is unmanaged (see update_flow.managed_install_root)."""
    return _flow.managed_install_root(
        binary,
        marker_root = _install_dir_for(binary),
        server_path_var = "WHISPER_SERVER_PATH",
        cpp_path_var = "UNSLOTH_WHISPER_CPP_PATH",
        dir_name = "whisper.cpp",
    )


def _source_build_status(binary: str, *, force_refresh: bool) -> Optional[dict]:
    """Update status for a markerless (source-build) install: offer the official
    prebuilt when one exists for this host and is newer than the installed
    binary. None -> caller falls through to the no-marker default (unsupported)."""
    res = _resolve_prebuilt_for_host(force_refresh = force_refresh)
    if not res or not res.get("prebuilt_available"):
        return None
    release_tag = res.get("release_tag")
    if not release_tag:
        return None
    # No resolvable install root (e.g. a pinned WHISPER_SERVER_PATH we cannot
    # manage) means an apply would not take effect, so do not offer.
    if _whisper_install_root(binary) is None:
        return None
    installed_tag = _installed_whisper_version(binary)
    installed_key = parse_release_version(installed_tag) if installed_tag else None
    latest_key = parse_release_version(release_tag)
    if installed_key is None or latest_key is None:
        # Unknown installed/latest version (the involuntary source-build case):
        # treat as behind so we still offer the prebuilt.
        update_available = True
    else:
        # Same downgrade guard as is_behind: only a strictly newer key is behind.
        update_available = latest_key > installed_key
    latest = release_tag
    # Size of the resolved prebuilt, so source builds show it like the marker
    # path. Fails open to None (offline / asset absent from the release).
    update_size_bytes = None
    if update_available:
        asset_name = res.get("asset")
        if isinstance(asset_name, str) and asset_name:
            try:
                assets = latest_release_assets(res.get("repo"), force_refresh = force_refresh)
                if assets:
                    update_size_bytes = assets.get(asset_name)
            except Exception as exc:  # pragma: no cover - network defensive
                logger.debug("whisper update: source-build size lookup failed", error = str(exc))
    with _job_lock:
        job = dict(_job)
    return {
        "supported": True,
        "update_available": update_available,
        "stale": False,
        "installed_tag": installed_tag,
        "latest_tag": latest,
        "published_repo": res.get("repo"),
        "installed_at_utc": None,
        "age_days": None,
        "source_build": True,
        "update_size_bytes": update_size_bytes,
        "job": job,
    }


def _active_install_is_local_link(binary: Optional[str]) -> bool:
    """True when the active whisper-server resolves through a locally-linked
    whisper.cpp directory (see update_flow.active_install_is_local_link)."""
    return _flow.active_install_is_local_link(binary, dir_name = "whisper.cpp")


def _local_link_status() -> dict:
    """Status payload for a local-link install: unmanaged, no update offered."""
    return _flow.local_link_status(_job, _job_lock)


def get_update_status(*, force_refresh: bool = False) -> dict:
    """Report whether a newer prebuilt exists plus the current job state.

    force_refresh bypasses the 24h release cache for an explicit "check now".
    """
    binary = _find_binary()
    # A locally-linked whisper.cpp dir is the user's own tree; never offer to
    # replace it. Bail before any network/freshness work.
    if _active_install_is_local_link(binary):
        return _local_link_status()
    marker = read_install_marker(binary)

    with _job_lock:
        job_running = _job["state"] == _JOB_RUNNING

    # No marker = source build / custom path. Offer the official prebuilt if one
    # now exists for this host. Skipped while the updater swaps the tree: each
    # poll would exec the half-replaced binary (on Windows that exec can make the
    # installer's os.replace fail) and the poller only consumes job progress.
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
            logger.debug("whisper update: force refresh failed", error = str(exc))

    freshness = check_prebuilt_freshness(binary)
    installed = freshness.get("installed_tag")
    latest = freshness.get("latest_tag")
    # `behind` compares the release version with a downgrade guard, so a lagging
    # /releases/latest or a lower published tag can't show a false update
    # (see whisper_cpp_freshness.is_behind).
    update_available = bool(freshness.get("has_marker") and freshness.get("behind"))

    # Size of the prebuilt that Update would download, for the banner. Only when
    # an update is offered; fails open to None (offline / no matching asset).
    update_size_bytes = None
    if update_available:
        try:
            update_size_bytes = update_download_size_bytes(
                marker,
                latest,
                freshness.get("published_repo") or repo,
                force_refresh = force_refresh,
            )
        except Exception as exc:  # pragma: no cover - network defensive
            logger.debug("whisper update: size lookup failed", error = str(exc))

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
        "update_size_bytes": update_size_bytes,
        "job": job,
    }


def _run_update(
    install_dir: Path, repo: str, asset: Optional[str], backend: Optional[str], script: Path
) -> None:
    """Worker: unload the warm whisper sidecar, run the installer for the latest
    prebuilt, then refresh caches so the next load uses the new build."""
    model_was_active = False
    try:
        # Free the binary while the installer swaps it. whisper.cpp is served by a
        # single sidecar subprocess (not a multi-backend registry like llama), so
        # a single unload is enough; reload_required reflects whether a model was
        # actually loaded before we tore it down.
        try:
            from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar

            sidecar = get_ggml_stt_sidecar()
            model_was_active = bool(sidecar.loaded_model)
            sidecar.unload()
        except Exception as exc:
            logger.debug("whisper update: sidecar unload failed", error = str(exc))

        cmd = [
            sys.executable,
            str(script),
            "--install-dir",
            str(install_dir),
            "--whisper-tag",
            "latest",
            "--published-repo",
            repo,
        ]
        # Preserve the installed accelerator across updates. Left unpinned the
        # installer re-detects the host, which is fine on unchanged hardware but
        # can reroute a deliberate choice (e.g. cpu on a GPU box); forwarding the
        # marker's backend keeps the same slice. No --published-release-tag:
        # the installer resolves the newest published release itself.
        if isinstance(backend, str) and backend:
            cmd.extend(["--backend", backend])
        cmd.extend(_rocm_install_args(asset))
        logger.info("whisper update: installing", cmd = " ".join(cmd))
        env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
        _flow.stream_installer(
            cmd,
            env,
            job = _job,
            job_lock = _job_lock,
            timeout_seconds = _INSTALL_TIMEOUT_SECONDS,
        )

        # Drop stale caches so the banner re-checks the swapped marker.
        # If GitHub is offline, latest stays unknown and the banner fails open.
        reset_caches(drop_disk = True)
        try:
            latest_published_release(repo, force_refresh = True)
        except Exception as exc:  # pragma: no cover - network defensive
            logger.debug("whisper update: post-install freshness refresh failed", error = str(exc))
        new_marker = read_install_marker(_find_binary())
        new_tag = (new_marker or {}).get("release_tag")

        with _job_lock:
            _job.update(
                state = _JOB_SUCCESS,
                message = (
                    f"Updated whisper.cpp to {new_tag}."
                    + (" Reload your model to use it." if model_was_active else "")
                ),
                to_tag = new_tag,
                reload_required = model_was_active,
                error = None,
                progress = 1.0,
                finished_at = _utcnow(),
            )
        logger.info("whisper update: success", to_tag = new_tag)
    except Exception as exc:
        logger.warning("whisper update: failed", error = str(exc))
        with _job_lock:
            _job.update(
                state = _JOB_ERROR,
                message = "whisper.cpp update failed.",
                error = str(exc),
                finished_at = _utcnow(),
            )


def start_update() -> dict:
    """Kick off a background update. Idempotent: a second call while one is
    running returns the in-flight job rather than starting another."""
    binary = _find_binary()
    # Refuse to update a locally-linked whisper.cpp dir: installing a prebuilt
    # here would write through the link into the user's own checkout (or fail)
    # and silently drop the link.
    if _active_install_is_local_link(binary):
        return {
            "started": False,
            "reason": "local_link",
            "message": (
                "whisper.cpp is a local directory linked into the managed path; "
                "Unsloth won't replace it. Update your own whisper.cpp checkout instead."
            ),
            "job": get_update_status()["job"],
        }
    marker = read_install_marker(binary)
    script = _installer_script()
    if script is None:
        return {
            "started": False,
            "reason": "installer_missing",
            "message": "install_whisper_prebuilt.py could not be located.",
            "job": get_update_status()["job"],
        }

    # A job already in flight wins over any freshness re-check below (and skips
    # its network call). The final lock block re-checks to close the TOCTOU.
    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}

    if marker:
        # Mirror the detection guard: a direct POST or a stale banner must not
        # start an install when the latest is not actually newer (force a fresh
        # check so a stale 24h cache can't wrongly block a real update either).
        status = get_update_status(force_refresh = True)
        if not status.get("update_available"):
            return {
                "started": False,
                "reason": "up_to_date",
                "message": "The installed whisper.cpp build is already at the latest prebuilt.",
                "job": status["job"],
            }
        install_dir = _install_dir_for(binary)
        repo = marker.get("published_repo") or DEFAULT_PUBLISHED_REPO
        from_tag = marker.get("release_tag")
        asset = marker.get("asset")
        backend = marker.get("backend")
    else:
        # Source build / custom path: only proceed when the same detection logic
        # would offer the update (prebuilt exists, install is behind, root is
        # manageable), so a direct POST cannot downgrade a newer source build.
        src = _source_build_status(binary, force_refresh = True) if binary else None
        if src is None:
            return {
                "started": False,
                "reason": "no_prebuilt_available",
                "message": (
                    "No official whisper.cpp prebuilt is available for this host, "
                    "so the source build cannot be swapped automatically."
                ),
                "job": get_update_status()["job"],
            }
        if not src.get("update_available"):
            return {
                "started": False,
                "reason": "up_to_date",
                "message": "The installed whisper.cpp build is already at or newer than the latest prebuilt.",
                "job": get_update_status()["job"],
            }
        res = _resolve_prebuilt_for_host()
        install_dir = _whisper_install_root(binary)
        repo = (res or {}).get("repo") or DEFAULT_PUBLISHED_REPO
        from_tag = None
        asset = (res or {}).get("asset")
        backend = (res or {}).get("backend")

    if install_dir is None:
        return {
            "started": False,
            "reason": "no_install_dir",
            "message": "Could not determine the whisper.cpp install directory.",
            "job": get_update_status()["job"],
        }

    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}
        _job.update(
            state = _JOB_RUNNING,
            message = "Downloading and installing the latest whisper.cpp prebuilt...",
            from_tag = from_tag,
            to_tag = None,
            reload_required = None,
            error = None,
            progress = 0.0,
            started_at = _utcnow(),
            finished_at = None,
        )
        job_snapshot = dict(_job)

    thread = threading.Thread(
        target = _run_update,
        args = (install_dir, repo, asset, backend, script),
        name = "whisper-cpp-update",
        daemon = True,
    )
    thread.start()
    return {"started": True, "reason": None, "job": job_snapshot}


def _reset_job_for_tests() -> None:
    """Test-only: return the job tracker to idle."""
    _flow.reset_job(_job, _job_lock)
