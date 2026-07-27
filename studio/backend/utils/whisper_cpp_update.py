# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app whisper.cpp prebuilt update.

Builds on utils.whisper_cpp_freshness (which detects whether a newer prebuilt
release exists) and adds the *apply* half: run install_whisper_prebuilt.py to
download the newest bundle for this host and atomically swap it in, so the next
model load uses it. Applies only run as the whisper phase of the combined
llama+whisper update (utils.llama_cpp_update.start_update chains
run_chained_phase); there is no standalone whisper update trigger.

Design notes:
- Detection is delegated to check_prebuilt_freshness(). We surface an
  ``update_available`` flag (installed_tag != latest_tag), laxer than freshness'
  ``stale`` (which also requires the install to be >= 3 days old). The UI shows
  the single main update item on update_available.
- Everything fails open: a missing marker / offline GitHub / source build just
  reports update_available=False and never blocks the app.
- The mechanics (managed-root resolution, local-link detection, the resolve
  probe, the streamed installer run) live in utils.prebuilt.update_flow; this
  module keeps the whisper policy. The ``job`` dict in the status payload stays
  idle (kept for response-shape stability): chained applies report progress
  through the llama job.
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
from utils.prebuilt.whisper_layout import canonical_install_root
from utils.whisper_cpp_freshness import (
    _INSTALL_MARKER_NAME,
    check_prebuilt_freshness,
    is_behind,
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
# nothing flips this to running. Kept so status payload shapes stay stable.
_job_lock = threading.Lock()
_job: dict = _flow.new_job()

_rocm_install_args = _flow.rocm_install_args


def _find_binary() -> Optional[str]:
    """Locate the active whisper-server binary via the STT sidecar's own resolver
    so update targets exactly what Unsloth runs. Lazy import keeps the heavy
    inference module off this module's import path."""
    try:
        from core.inference.stt_ggml_sidecar import find_whisper_server_binary
        return find_whisper_server_binary()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("whisper update: binary discovery failed", error = str(exc))
        return None


def _install_dir_for(binary_path: Optional[str]) -> Optional[Path]:
    """The directory holding UNSLOTH_WHISPER_PREBUILT_INFO.json: the install root
    install_whisper_prebuilt.py wrote (``<install-dir>`` whose canonical server is
    ``build/bin/whisper-server``) and the one we re-install into."""
    root = canonical_install_root(binary_path)
    if root is not None and (root / _INSTALL_MARKER_NAME).is_file():
        return root
    return _flow.install_dir_for(binary_path, marker_name = _INSTALL_MARKER_NAME)


def _installer_script() -> Optional[Path]:
    """Locate install_whisper_prebuilt.py (UNSLOTH_WHISPER_INSTALLER wins)."""
    return _flow.find_installer_script(
        env_var = "UNSLOTH_WHISPER_INSTALLER", script_name = "install_whisper_prebuilt.py"
    )


# Markerless (source-build) installs have no UNSLOTH_WHISPER_PREBUILT_INFO.json,
# so we ask the installer whether an official prebuilt now exists for this host.
_resolve_memo: dict = {}


def _resolve_prebuilt_for_host(
    *, force_refresh: bool = False, backend: Optional[str] = None
) -> Optional[dict]:
    """Run install_whisper_prebuilt.py --resolve-prebuilt (no download); return
    {prebuilt_available, repo, release_tag, upstream_tag, backend, asset, os,
    arch, ...} or None. Fail-open: any error -> None so a source build never
    blocks the app."""
    extra_args = ("--backend", backend) if backend else ()
    return _flow.resolve_prebuilt_for_host(
        force_refresh = force_refresh,
        memo = _resolve_memo,
        installer_script = lambda: _installer_script(),
        log_message = "whisper update: resolve-prebuilt failed",
        extra_args = extra_args,
    )


def _installed_whisper_version(binary: Optional[str]) -> Optional[str]:
    """Best-effort ``v<A.B.C>`` from ``whisper-server --version``. None when the
    binary is missing, reports no version, or cannot run. Used only for the
    markerless source-build downgrade guard, so it fails open to None."""
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
    prebuilt when one exists for this host and is newer than the installed binary.
    None -> caller falls through to the no-marker default (unsupported)."""
    res = _resolve_prebuilt_for_host(force_refresh = force_refresh)
    if not res or not res.get("prebuilt_available"):
        return None
    release_tag = res.get("release_tag")
    if not release_tag:
        return None
    # No resolvable install root (e.g. a pinned WHISPER_SERVER_PATH we cannot
    # manage) means an apply would not take effect, so do not offer it.
    if _whisper_install_root(binary) is None:
        return None
    installed_tag = _installed_whisper_version(binary)
    installed_key = parse_release_version(installed_tag) if installed_tag else None
    latest_key = parse_release_version(release_tag)
    if installed_key is None or latest_key is None:
        # Unknown installed/latest version (involuntary source-build case): treat
        # as behind so we still offer the prebuilt.
        update_available = True
    else:
        # Same downgrade guard as is_behind: only a strictly newer key is behind.
        update_available = latest_key > installed_key
    latest = release_tag
    # Size of the resolved prebuilt, so source builds show it like the marker
    # path. Fails open to None (offline / asset absent from release).
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
    # exists for this host.
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
    compatible_override = False
    if sys.platform == "darwin" and marker is not None:
        # The newest published release may require a newer macOS. Ask the same
        # host-aware resolver the installer uses so the banner compares against
        # the newest release this host can actually install, avoiding a repeated
        # offer of an incompatible release after walkback.
        resolved = _resolve_prebuilt_for_host(
            force_refresh = force_refresh,
            backend = marker.get("backend") if isinstance(marker.get("backend"), str) else None,
        )
        compatible_latest = (resolved or {}).get("release_tag")
        if (resolved or {}).get("prebuilt_available") and isinstance(compatible_latest, str):
            compatible_override = compatible_latest != latest
            latest = compatible_latest
    # `behind` compares the release version with a downgrade guard, so a lagging
    # /releases/latest or lower published tag can't show a false update
    # (whisper_cpp_freshness.is_behind).
    update_available = bool(
        freshness.get("has_marker")
        and (
            is_behind(installed, latest)
            if sys.platform == "darwin" and marker is not None
            else freshness.get("behind")
        )
    )

    # Size of the prebuilt Update would download, for the banner. Only when an
    # update is offered; fails open to None (offline / no matching asset).
    update_size_bytes = None
    if update_available and not compatible_override:
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
        "stale": bool(update_available and freshness.get("stale")),
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
    pin_release_tag: Optional[str] = None,
) -> dict:
    """Replace whisper.cpp while the sidecar blocks every new load."""
    try:
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        sidecar = get_ggml_stt_sidecar()
    except Exception as exc:
        # Replacing the tree without the singleton's maintenance barrier would
        # reopen the Windows executable-lock and stale-process races. Fail closed.
        raise RuntimeError("could not coordinate the whisper.cpp sidecar update") from exc

    # update_maintenance publishes its guard before waiting for an existing
    # transcription, unloads the warm server, and holds the sidecar lock across
    # the complete atomic install. No new process can relock or outlive the tree.
    with sidecar.update_maintenance() as model_was_active:
        return _install_latest_while_blocked(
            install_dir,
            repo,
            asset,
            backend,
            script,
            set_progress,
            pin_release_tag = pin_release_tag,
            model_was_active = model_was_active,
        )


def _install_latest_while_blocked(
    install_dir: Path,
    repo: str,
    asset: Optional[str],
    backend: Optional[str],
    script: Path,
    set_progress,
    *,
    pin_release_tag: Optional[str],
    model_was_active: bool,
) -> dict:
    """Run the installer with the sidecar already in update maintenance."""

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
    # installer re-detects the host, fine on unchanged hardware but able to
    # reroute a deliberate choice (e.g. cpu on a GPU box); forwarding the marker's
    # backend keeps the same slice.
    if isinstance(backend, str) and backend:
        cmd.extend(["--backend", backend])
    if pin_release_tag:
        cmd.extend(["--published-release-tag", pin_release_tag])
    cmd.extend(_rocm_install_args(asset))
    logger.info("whisper update: installing", cmd = " ".join(cmd))
    env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
    # Every nonzero exit is a failed phase. In particular, exit 2 means the
    # requested release was incompatible and no install happened, so reporting
    # success would hide the banner and toast an update that never landed.
    _flow.stream_installer(
        cmd,
        env,
        set_progress = set_progress,
        timeout_seconds = _INSTALL_TIMEOUT_SECONDS,
    )

    # Drop stale caches so the banner re-checks the swapped marker. If GitHub is
    # offline, latest stays unknown and the banner fails open.
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


def chained_phase_plan(
    *, force_refresh: bool = False, paired_llama_will_update: bool = False
) -> dict:
    """Whisper's side of the combined llama+whisper update item.

    Returns {status, update_available, skip_reason, phase}: `status` is the
    marker-path status dict (or a minimal one when whisper is skipped),
    `update_available` says the chained apply would run a whisper phase, and
    `phase` carries what run_chained_phase needs. Only marker-managed installs are
    chained: local links, source builds and unmanaged/pinned paths are silently
    skipped so whisper can never block a llama update. Never raises; failures
    degrade to a skip."""
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
        # utils API can update source builds; the chain does not.
        return {
            "status": None,
            "update_available": False,
            "skip_reason": "source_build" if binary else "not_installed",
            "phase": None,
        }
    status = get_update_status(force_refresh = force_refresh)
    plan: dict = {"status": status, "update_available": False, "skip_reason": None, "phase": None}
    if not status.get("update_available"):
        # Skew note: when llama just updated but whisper is already latest, a slim
        # install keeps hardlinks to the OLD llama ggml inodes -- still the exact
        # build whisper was installed against, so skipping is correct and needs no
        # re-wiring. A whisper phase that does run re-wires via the installer
        # (prepare_runtime_payload).
        plan["skip_reason"] = "up_to_date"
        return plan
    if marker.get("install_kind") == "slim" and not paired_llama_will_update:
        # A slim install can only be refreshed from a completed managed llama
        # prebuilt. Ask the installer through its read-only resolver so local
        # links, markerless/current source builds, and incomplete managed trees
        # never produce an Update button that can only fail. When the llama
        # phase will run first, it supplies the repaired pairing instead.
        resolved = _resolve_prebuilt_for_host(
            force_refresh = force_refresh,
            backend = marker.get("backend") if isinstance(marker.get("backend"), str) else None,
        )
        if not (resolved or {}).get("prebuilt_available"):
            plan["skip_reason"] = "paired_llama_unavailable"
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
        # Install exactly the release the check offered: the installer's unpinned
        # "latest" prefers the download-host /releases/latest pointer, which sorts
        # by commit date and can lag the published_at pick the freshness check
        # used, reinstalling an older build in a loop (the #6219 class the llama
        # phase pins against). Not on macOS: the llama phase is unpinned there
        # (walk-back to an os-compatible release), so pinning whisper to the
        # newest tag could be an impossible pairing (min_os / requires_llama_tag)
        # on every retry.
        "pin_release_tag": None if sys.platform == "darwin" else status.get("latest_tag"),
    }
    return plan


def run_chained_phase(phase: dict, set_progress) -> dict:
    """Run the whisper phase of a combined update (spec from chained_phase_plan):
    same unload/install/cache-refresh path as the standalone job, reporting
    progress through the chained job's window rather than whisper's own."""
    return _install_latest(
        phase["install_dir"],
        phase["repo"],
        phase["asset"],
        phase["backend"],
        phase["script"],
        set_progress,
        pin_release_tag = phase.get("pin_release_tag"),
    )
