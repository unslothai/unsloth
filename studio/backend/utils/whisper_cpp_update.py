# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app whisper.cpp prebuilt update.

Builds on utils.whisper_cpp_freshness (which detects whether a newer prebuilt
release exists) and adds the *apply* half: run install_whisper_prebuilt.py to
download the newest bundle for this host and atomically swap it in place, so
the next model load uses it. Applies only run as the whisper phase of the
combined llama+whisper update (utils.llama_cpp_update.start_update chains
run_chained_phase); there is no standalone whisper update trigger.

Design notes:
- Detection is delegated to check_prebuilt_freshness(). We surface an
  ``update_available`` flag (installed_tag != latest_tag) which is laxer than
  freshness' ``stale`` (which additionally requires the install to be >= 3 days
  old). The UI shows the single main update item on update_available.
- Everything fails open: a missing marker / offline GitHub / source build just
  reports update_available=False and never blocks the app.
- The mechanics (managed-root resolution, local-link detection, the resolve
  probe, the streamed installer run) live in utils.prebuilt.update_flow; this
  module keeps the whisper policy. The ``job`` dict in the status payload is
  kept for response-shape stability but stays idle: chained applies report
  progress through the llama job.
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

# Always-idle job payload: whisper applies run inside the chained llama job, so
# nothing ever flips this to running. Kept so status payload shapes are stable.
_job_lock = threading.Lock()
_job: dict = _flow.new_job()

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

    # No marker = source build / custom path. Offer the official prebuilt if one
    # now exists for this host.
    if marker is None and binary is not None:
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


def _install_latest(
    install_dir: Path,
    repo: str,
    asset: Optional[str],
    backend: Optional[str],
    script: Path,
    set_progress,
) -> dict:
    """Unload the warm whisper sidecar, run the installer for the latest
    prebuilt, then refresh caches so the next load uses the new build. Runs as
    the whisper phase of the chained llama+whisper update. Returns
    {to_tag, reload_required, message}; raises on failure."""
    # Free the binary while the installer swaps it. whisper.cpp is served by a
    # single sidecar subprocess (not a multi-backend registry like llama), so
    # a single unload is enough; reload_required reflects whether a model was
    # actually loaded before we tore it down.
    model_was_active = False
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
        set_progress = set_progress,
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
    logger.info("whisper update: success", to_tag = new_tag)
    return {
        "to_tag": new_tag,
        "reload_required": model_was_active,
        "message": (
            f"Updated whisper.cpp to {new_tag}."
            + (" Reload your model to use it." if model_was_active else "")
        ),
    }


def chained_phase_plan(*, force_refresh: bool = False) -> dict:
    """Whisper's side of the combined llama+whisper update item.

    Returns {status, update_available, skip_reason, phase}: `status` is the
    marker-path status dict (or a minimal one when whisper is skipped),
    `update_available` says the chained apply would actually run a whisper
    phase, and `phase` carries what run_chained_phase needs. Only marker-managed
    installs are chained: local links, source builds and unmanaged/pinned paths
    are silently skipped so whisper can never block a llama update. Never
    raises; failures degrade to a skip."""
    binary = _find_binary()
    if _active_install_is_local_link(binary):
        return {
            "status": _local_link_status(),
            "update_available": False,
            "skip_reason": "local_link",
            "phase": None,
        }
    marker = read_install_marker(binary)
    if marker is None:
        # No marker: whisper is absent or a source/custom build. The standalone
        # utils API can still update source builds; the chain does not.
        return {
            "status": None,
            "update_available": False,
            "skip_reason": "source_build" if binary else "not_installed",
            "phase": None,
        }
    status = get_update_status(force_refresh = force_refresh)
    plan: dict = {"status": status, "update_available": False, "skip_reason": None, "phase": None}
    if not status.get("update_available"):
        # Skew note: when llama just updated but whisper is already latest, a
        # slim install keeps hardlinks to the OLD llama ggml inodes -- still the
        # exact build whisper was installed against, so skipping is correct by
        # design and needs no re-wiring. A whisper phase that does run re-wires
        # via the installer (prepare_runtime_payload).
        plan["skip_reason"] = "up_to_date"
        return plan
    script = _installer_script()
    if script is None:
        plan["skip_reason"] = "installer_missing"
        return plan
    install_dir = _install_dir_for(binary)
    if install_dir is None:
        plan["skip_reason"] = "no_install_dir"
        return plan
    plan["update_available"] = True
    plan["phase"] = {
        "install_dir": install_dir,
        "repo": marker.get("published_repo") or DEFAULT_PUBLISHED_REPO,
        "asset": marker.get("asset"),
        "backend": marker.get("backend"),
        "script": script,
    }
    return plan


def run_chained_phase(phase: dict, set_progress) -> dict:
    """Run the whisper phase of a combined update (spec from chained_phase_plan):
    same unload/install/cache-refresh path as the standalone job, reporting
    progress through the chained job's window instead of whisper's own job."""
    return _install_latest(
        phase["install_dir"],
        phase["repo"],
        phase["asset"],
        phase["backend"],
        phase["script"],
        set_progress,
    )
