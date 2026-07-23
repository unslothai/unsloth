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
- The mechanics (managed-root resolution, local-link detection, the resolve
  probe, the streamed installer run) live in utils.prebuilt.update_flow; this
  module keeps the llama policy and the job dict its callers poll.
- This is the single main update item: whisper.cpp piggybacks on it. Status
  folds in a whisper sub-status (update_available becomes the union) and apply
  chains a whisper phase after the llama phase when whisper is behind (see
  update_flow.run_chained_update and whisper_cpp_update.chained_phase_plan).
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

from utils.llama_cpp_freshness import (
    _INSTALL_MARKER_NAME,
    check_prebuilt_freshness,
    latest_published_release,
    latest_release_assets,
    parse_base_build,
    read_install_marker,
    reset_caches,
    update_download_size_bytes,
)
from utils.prebuilt import update_flow as _flow

logger = structlog.get_logger(__name__)

DEFAULT_PUBLISHED_REPO = "unslothai/llama.cpp"
_INSTALL_TIMEOUT_SECONDS = 1800  # 30 min ceiling for download + build/validate

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
    """Locate the active llama-server binary via the inference backend's own
    resolver, so update targets exactly what Unsloth runs. Lazy import keeps the
    heavy inference module off this module's import path."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._find_llama_server_binary()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llama update: binary discovery failed", error = str(exc))
        return None


def _install_dir_for(binary_path: Optional[str]) -> Optional[Path]:
    """The directory holding UNSLOTH_PREBUILT_INFO.json -- i.e. the install root
    install_llama_prebuilt.py wrote and the one we re-install into."""
    return _flow.install_dir_for(binary_path, marker_name = _INSTALL_MARKER_NAME)


def _installer_script() -> Optional[Path]:
    """Locate install_llama_prebuilt.py (UNSLOTH_LLAMA_INSTALLER wins)."""
    return _flow.find_installer_script(
        env_var = "UNSLOTH_LLAMA_INSTALLER", script_name = "install_llama_prebuilt.py"
    )


# Markerless (source-build) installs have no UNSLOTH_PREBUILT_INFO.json, so we
# ask the installer whether an official prebuilt now exists for this host.
_resolve_memo: dict = {}


def _resolve_prebuilt_for_host(*, force_refresh: bool = False) -> Optional[dict]:
    """Run install_llama_prebuilt.py --resolve-prebuilt (no download) and return
    {prebuilt_available, repo, release_tag, llama_tag, asset, install_kind} or
    None. Fail-open: any error -> None so a source build never blocks the app."""
    return _flow.resolve_prebuilt_for_host(
        force_refresh = force_refresh,
        memo = _resolve_memo,
        installer_script = lambda: _installer_script(),
        log_message = "llama update: resolve-prebuilt failed",
    )


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


def get_installed_llama_version() -> Optional[str]:
    """Display string for the active llama.cpp install (e.g. 'b9585' or
    'b9601-mix-a0e2906'), or None.

    Prefers the install marker's release_tag -- the full unsloth release
    identity, the same field the update banner compares as installed (see
    #6219) -- so a 'b9601-mix-a0e2906' build reads back in full rather than
    collapsing to its base 'b9601'. The marker's bare ``tag`` is only the
    upstream llama.cpp build (no '-mix-<commit>' suffix), so it's the fallback.
    Last resort is ``b<build>`` parsed from ``llama-server --version`` for
    source/custom builds that have no marker.

    Lightweight: reads the local marker and at most runs ``--version``. Does no
    network or release-freshness work (unlike get_update_status), so it is safe
    to call from latency-sensitive paths like the About panel.
    """
    binary = _find_binary()
    marker = read_install_marker(binary)
    if marker:
        tag = marker.get("release_tag") or marker.get("tag")
        if tag:
            return tag
    # Markerless/source build: the fallback execs ``llama-server --version``.
    # Skip it while an update is swapping the tree -- on Windows that exec can
    # make the installer's os.replace fail (the same race get_update_status's
    # source-build probe guards against). The panel just omits the row.
    with _job_lock:
        job_running = _job["state"] == _JOB_RUNNING
    if job_running:
        return None
    n = _installed_build_number(binary)
    return f"b{n}" if n is not None else None


def _llama_install_root(binary: Optional[str]) -> Optional[Path]:
    """The Unsloth-managed llama.cpp root the active binary lives under, or None
    when the binary is unmanaged (see update_flow.managed_install_root)."""
    return _flow.managed_install_root(
        binary,
        marker_root = _install_dir_for(binary),
        server_path_var = "LLAMA_SERVER_PATH",
        cpp_path_var = "UNSLOTH_LLAMA_CPP_PATH",
        dir_name = "llama.cpp",
    )


def _source_build_status(binary: str, *, force_refresh: bool) -> Optional[dict]:
    """Update status for a markerless (source-build) install: offer the official
    prebuilt when one exists for this host and is newer than the installed
    binary. None -> caller falls through to the no-marker default (unsupported)."""
    res = _resolve_prebuilt_for_host(force_refresh = force_refresh)
    if not res or not res.get("prebuilt_available"):
        return None
    # llama_tag is the upstream base (bNNNN, what --version reports); release_tag
    # is the full tag, either a same-base mix (bNNNN-mix-<sha>) or a fork wrapper
    # (e.g. v1.0). Compare the numeric base against llama_tag.
    base_tag = res.get("llama_tag") or res.get("release_tag")
    release_tag = res.get("release_tag")
    if not base_tag:
        return None
    # No resolvable install root (e.g. a pinned LLAMA_SERVER_PATH we cannot
    # manage) means an apply would not take effect, so do not offer.
    if _llama_install_root(binary) is None:
        return None
    installed_build = _installed_build_number(binary)
    latest_build = parse_base_build(base_tag)
    # A same-base mix adds patches the bare base lacks, so it is newer even at an
    # unchanged build number (the marker path's is_behind already does this). The
    # bNNNN anchor keeps a fork wrapper tag from being read as a mix.
    latest_is_mix = (
        isinstance(release_tag, str)
        and latest_build is not None
        and parse_base_build(release_tag) == latest_build
        and release_tag.strip() != f"b{latest_build}"
    )
    if installed_build is None or latest_build is None:
        # Unknown installed/latest version (the involuntary source-build case):
        # treat as behind so we still offer the prebuilt.
        update_available = True
    elif installed_build < latest_build:
        update_available = True
    elif installed_build == latest_build:
        # Same upstream base: offer the extra-patch mix, never a bare rebuild.
        update_available = latest_is_mix
    else:
        # Source build newer than the latest prebuilt: downgrade guard.
        update_available = False
    # Display the mix tag when that's what makes it newer; otherwise the base.
    latest = release_tag if latest_is_mix else base_tag
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
                logger.debug("llama update: source-build size lookup failed", error = str(exc))
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
        "update_size_bytes": update_size_bytes,
        "job": job,
    }


def _active_install_is_local_link(binary: Optional[str]) -> bool:
    """True when the active llama-server resolves through a --with-llama-cpp-dir
    local link at the canonical llama.cpp directory (see
    update_flow.active_install_is_local_link)."""
    return _flow.active_install_is_local_link(binary, dir_name = "llama.cpp")


def _local_link_status() -> dict:
    """Status payload for a local-link install: unmanaged, no update offered."""
    return _flow.local_link_status(_job, _job_lock)


def _whisper_chain_status(
    *, force_refresh: bool = False, paired_llama_will_update: bool = False
) -> Optional[dict]:
    """Whisper's piggyback plan for the combined update item (see
    whisper_cpp_update.chained_phase_plan). None disables the piggyback --
    fail-open so whisper can never break the llama status or apply."""
    try:
        from utils import whisper_cpp_update
        return whisper_cpp_update.chained_phase_plan(
            force_refresh = force_refresh,
            paired_llama_will_update = paired_llama_will_update,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llama update: whisper piggyback probe failed", error = str(exc))
        return None


def _merge_whisper_status(status: dict, *, force_refresh: bool = False) -> dict:
    """Fold the whisper sub-status into the llama status payload: the llama
    update item is the single UI surface, so update_available becomes the union
    (llama behind OR whisper behind) while llama_update_available keeps the
    llama-only flag. All pre-existing top-level fields are preserved."""
    status["llama_update_available"] = bool(status.get("update_available"))
    plan = _whisper_chain_status(
        force_refresh = force_refresh,
        paired_llama_will_update = status["llama_update_available"],
    )
    if plan is None:
        status["whisper"] = None
        status["update_component"] = "llama" if status["llama_update_available"] else None
        return status
    sub = plan.get("status") or {}
    status["whisper"] = {
        "update_available": bool(plan.get("update_available")),
        "installed_tag": sub.get("installed_tag"),
        "latest_tag": sub.get("latest_tag"),
        "update_size_bytes": sub.get("update_size_bytes"),
        "skip_reason": plan.get("skip_reason"),
    }
    whisper_update_available = bool(plan.get("update_available"))
    if whisper_update_available:
        status["update_available"] = True
    status["update_component"] = (
        "llama"
        if status["llama_update_available"]
        else "whisper"
        if whisper_update_available
        else None
    )
    return status


def get_update_status(*, force_refresh: bool = False) -> dict:
    """Report whether an update is available plus the current job state.

    This is the single main update item: llama.cpp drives it and the whisper
    piggyback is folded in (see _merge_whisper_status). force_refresh bypasses
    the 24h release cache for an explicit "check now".
    """
    status = _llama_only_status(force_refresh = force_refresh)
    return _merge_whisper_status(status, force_refresh = force_refresh)


def _llama_only_status(*, force_refresh: bool = False) -> dict:
    """The llama.cpp half of get_update_status (no whisper sub-status)."""
    binary = _find_binary()
    # A --with-llama-cpp-dir local link is the user's own tree; never offer to
    # replace it. Bail before any network/freshness work.
    if _active_install_is_local_link(binary):
        return _local_link_status()
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
    # `behind` compares the full release identity with a base-build guard, so a
    # lagging /releases/latest or a mix-tagged latest can't show a false update
    # (see llama_cpp_freshness.is_behind).
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
            logger.debug("llama update: size lookup failed", error = str(exc))

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


def _run_llama_phase(
    install_dir: Path,
    repo: str,
    asset: Optional[str],
    script: Path,
    pin_release_tag: Optional[str],
    set_progress,
    force_cpu: bool = False,
    llama_backend: Optional[str] = None,
) -> dict:
    """The llama phase of a chained update: put the backend into a maintenance
    state, run the installer for the latest prebuilt, then refresh caches so the
    next load uses the new build. Returns {to_tag, reload_required, message};
    raises on failure.

    pin_release_tag pins the installer to that exact published release instead
    of letting it re-resolve "latest" itself (see start_update for why)."""
    backend = None
    model_was_active = False
    try:
        # Block loads and free the binary while the installer swaps it.
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
                    # Active processes can lock the exe on Windows.
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
        if pin_release_tag:
            cmd.extend(["--published-release-tag", pin_release_tag])
        cmd.extend(_rocm_install_args(asset))
        # Re-assert a deliberate CPU install (--force-cpu) so detect_host on a GPU host
        # does not re-route to a GPU/Vulkan bundle and revive the crash (#7213). --force-cpu
        # (not --cpu-fallback) also re-persists force_cpu, keeping the choice across future
        # updates. A natural fallback (or a legacy marker without the flag) heals to GPU (#6097).
        if force_cpu:
            cmd.append("--force-cpu")
        if llama_backend == "vulkan":
            cmd.extend(["--llama-backend", "vulkan"])
        logger.info("llama update: installing", cmd = " ".join(cmd))
        env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
        # Preserve a Vulkan install across updates: detect_host on a CUDA/ROCm
        # box would otherwise re-route and silently replace the Vulkan build.
        # Re-assert it via the same env/CLI flags setup uses (mirrors
        # _rocm_install_args).
        if llama_backend == "vulkan" or (asset and "vulkan" in asset.lower()):
            env["UNSLOTH_FORCE_VULKAN"] = "1"
            env["UNSLOTH_LLAMA_BACKEND"] = "vulkan"
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
            logger.debug("llama update: post-install freshness refresh failed", error = str(exc))
        new_marker = read_install_marker(_find_binary())
        new_tag = (new_marker or {}).get("release_tag") or (new_marker or {}).get("tag")

        # Pinned install must land on that exact release; a same-repo mismatch
        # means the pin was ignored (Vulkan/Intel reroute to another repo is fine).
        if (
            pin_release_tag
            and new_tag
            and (new_marker or {}).get("published_repo") == repo
            and new_tag != pin_release_tag
        ):
            raise RuntimeError(f"pinned release {pin_release_tag} but installer produced {new_tag}")

        logger.info("llama update: success", to_tag = new_tag)
        return {
            "to_tag": new_tag,
            "reload_required": model_was_active,
            "message": (
                f"Updated llama.cpp to {new_tag}."
                + (" Reload your model to use it." if model_was_active else "")
            ),
        }
    except Exception as exc:
        logger.warning("llama update: failed", error = str(exc))
        raise
    finally:
        # Always clear maintenance state.
        if backend is not None:
            try:
                backend._llama_update_in_progress = False
            except Exception:  # pragma: no cover - defensive
                pass


# Combined-job progress split when both phases run (download sizes: the llama
# bundle dwarfs the whisper one); normalized to 0..1 when a phase is skipped.
_LLAMA_PHASE_WEIGHT = 0.7
_WHISPER_PHASE_WEIGHT = 0.3


def _plan_llama_phase() -> dict:
    """Decide how the llama phase of a combined update runs. Returns {"spec"}
    when llama should install, else {"skip_reason", "refusal"}: skip_reason
    marks the phase skipped inside a chained job, refusal is the started=False
    response when the whisper phase has nothing to run either."""
    binary = _find_binary()
    # Refuse to update a --with-llama-cpp-dir local link: installing a prebuilt
    # here would write through the link into the user's own checkout (or fail)
    # and silently drop the link the flag created.
    if _active_install_is_local_link(binary):
        return {
            "skip_reason": "local_link",
            "refusal": {
                "started": False,
                "reason": "local_link",
                "message": (
                    "llama.cpp is a local directory linked with --with-llama-cpp-dir; "
                    "Unsloth won't replace it. Update your own llama.cpp checkout instead."
                ),
            },
        }
    marker = read_install_marker(binary)
    script = _installer_script()
    if script is None:
        return {
            "skip_reason": "installer_missing",
            "refusal": {
                "started": False,
                "reason": "installer_missing",
                "message": "install_llama_prebuilt.py could not be located.",
            },
        }

    if marker:
        # Mirror the detection guard: a direct POST or a stale banner must not
        # start an install when the latest is not actually newer (force a fresh
        # check so a stale 24h cache can't wrongly block a real update either).
        status = _llama_only_status(force_refresh = True)
        if not status.get("update_available"):
            return {
                "skip_reason": "up_to_date",
                "refusal": {
                    "started": False,
                    "reason": "up_to_date",
                    "message": "The installed llama.cpp build is already at the latest prebuilt.",
                },
            }
        install_dir = _install_dir_for(binary)
        repo = marker.get("published_repo") or DEFAULT_PUBLISHED_REPO
        from_tag = marker.get("tag") or marker.get("release_tag")
        asset = marker.get("asset")
        force_cpu = bool(marker.get("force_cpu"))
        llama_backend = marker.get("llama_backend")
        if llama_backend == "vulkan" or (asset and "vulkan" in str(asset).lower()):
            llama_backend = "vulkan"
        # Install exactly the release the banner offered: the installer's own
        # "latest" is commit-date ordered and can lag the published_at pick
        # above, reinstalling the current build in a loop (the #6219 class).
        # Not on macOS, which needs the older-release walk-back a pin disables
        # (skipping too-new prebuilts); elsewhere an unusable latest now fails
        # the job loudly (retryable) instead of walking back.
        pin_release_tag = None if sys.platform == "darwin" else status.get("latest_tag")
    else:
        # Source build / custom path: only proceed when the same detection logic
        # would offer the update (prebuilt exists, install is behind, root is
        # manageable), so a direct POST cannot downgrade a newer source build.
        src = _source_build_status(binary, force_refresh = True) if binary else None
        if src is None:
            return {
                "skip_reason": "no_prebuilt_available",
                "refusal": {
                    "started": False,
                    "reason": "no_prebuilt_available",
                    "message": (
                        "No official llama.cpp prebuilt is available for this host, "
                        "so the source build cannot be swapped automatically."
                    ),
                },
            }
        if not src.get("update_available"):
            return {
                "skip_reason": "up_to_date",
                "refusal": {
                    "started": False,
                    "reason": "up_to_date",
                    "message": (
                        "The installed llama.cpp build is already at or newer than the "
                        "latest prebuilt."
                    ),
                },
            }
        res = _resolve_prebuilt_for_host()
        install_dir = _llama_install_root(binary)
        repo = (res or {}).get("repo") or DEFAULT_PUBLISHED_REPO
        from_tag = None
        asset = (res or {}).get("asset")
        # Source builds carry no forced-CPU marker, so nothing to preserve here.
        force_cpu = False
        llama_backend = None
        # No pin: source-build detection resolves via --resolve-prebuilt latest,
        # the same resolver the unpinned apply uses, so the two already agree.
        pin_release_tag = None

    if install_dir is None:
        return {
            "skip_reason": "no_install_dir",
            "refusal": {
                "started": False,
                "reason": "no_install_dir",
                "message": "Could not determine the llama.cpp install directory.",
            },
        }
    return {
        "spec": {
            "install_dir": install_dir,
            "repo": repo,
            "asset": asset,
            "script": script,
            "pin_release_tag": pin_release_tag,
            "from_tag": from_tag,
            "force_cpu": force_cpu,
            "llama_backend": llama_backend,
        }
    }


def start_update() -> dict:
    """Kick off a background update job. The job chains the llama phase (the
    existing flow) with a whisper phase that runs only when whisper is actually
    behind; either phase no-ops cleanly when its component is current or
    unmanaged. Idempotent: a second call while one is running returns the
    in-flight job rather than starting another."""
    # A job already in flight wins over any freshness re-check below (and skips
    # its network calls). The final lock block re-checks to close the TOCTOU.
    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}

    llama_plan = _plan_llama_phase()
    llama_spec = llama_plan.get("spec")
    whisper_plan = _whisper_chain_status(
        force_refresh = True,
        paired_llama_will_update = llama_spec is not None,
    )
    whisper_spec = (whisper_plan or {}).get("phase")
    if llama_spec is None and whisper_spec is None:
        # Nothing to run in either phase: answer with the llama refusal so the
        # existing reasons (local_link / up_to_date / ...) keep their meaning.
        refusal = dict(llama_plan["refusal"])
        with _job_lock:
            refusal["job"] = dict(_job)
        return refusal

    whisper_run = None
    if whisper_spec is not None:
        from utils import whisper_cpp_update as _whisper
        whisper_run = lambda set_progress: _whisper.run_chained_phase(whisper_spec, set_progress)

    phases = [
        {
            "name": "llama",
            "weight": _LLAMA_PHASE_WEIGHT,
            "failure_message": "llama.cpp update failed.",
            "skip_reason": llama_plan.get("skip_reason"),
            "run": (
                (
                    lambda set_progress: _run_llama_phase(
                        llama_spec["install_dir"],
                        llama_spec["repo"],
                        llama_spec["asset"],
                        llama_spec["script"],
                        llama_spec["pin_release_tag"],
                        set_progress,
                        force_cpu = llama_spec.get("force_cpu", False),
                        llama_backend = llama_spec.get("llama_backend"),
                    )
                )
                if llama_spec
                else None
            ),
        },
        {
            "name": "whisper",
            "weight": _WHISPER_PHASE_WEIGHT,
            "failure_message": "whisper.cpp update failed.",
            # The sidecar reload is whisper-internal; it must not trip the
            # job-level reload flag the chat frontend resyncs on.
            "affects_job_reload": False,
            "skip_reason": (whisper_plan or {}).get("skip_reason") or "unavailable",
            "run": whisper_run,
        },
    ]
    running = " + ".join(
        name for name, spec in (("llama.cpp", llama_spec), ("whisper.cpp", whisper_spec)) if spec
    )

    with _job_lock:
        if _job["state"] == _JOB_RUNNING:
            return {"started": False, "reason": "already_running", "job": dict(_job)}
        _job.update(
            state = _JOB_RUNNING,
            message = f"Downloading and installing the latest {running} prebuilt...",
            from_tag = (llama_spec or {}).get("from_tag"),
            to_tag = None,
            reload_required = None,
            error = None,
            progress = 0.0,
            started_at = _utcnow(),
            finished_at = None,
            phases = None,
        )
        job_snapshot = dict(_job)

    thread = threading.Thread(
        target = _flow.run_chained_update,
        args = (phases,),
        kwargs = {"job": _job, "job_lock": _job_lock},
        name = "llama-cpp-update",
        daemon = True,
    )
    thread.start()
    return {"started": True, "reason": None, "job": job_snapshot}


def _reset_job_for_tests() -> None:
    """Test-only: return the job tracker to idle."""
    _flow.reset_job(_job, _job_lock)
