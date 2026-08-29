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
from contextlib import ExitStack
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
from utils.prebuilt.llama_backend import (
    REQUESTABLE_BACKENDS,
    environment_backend_override,
    marker_backend,
    marker_backend_request,
    normalize_backend,
)

logger = structlog.get_logger(__name__)

DEFAULT_PUBLISHED_REPO = "unslothai/llama.cpp"
_INSTALL_TIMEOUT_SECONDS = 1800  # 30 min ceiling for download + build/validate
# install_llama_prebuilt.py EXIT_NO_SPACE: out of disk, retrying will not help.
_EXIT_NO_SPACE = 4
# A concrete backend selection could not be satisfied.
_EXIT_BACKEND_UNAVAILABLE = 5
# Prebuilt path failed; setup scripts source-build, but the in-app updater cannot.
_EXIT_FALLBACK = 2


class _LlamaPhaseError(RuntimeError):
    """A llama phase failed after possibly unloading the active runtime."""

    def __init__(self, message: str, *, reload_required: bool):
        super().__init__(message)
        self.reload_required = reload_required


# Background job state. Single in-flight update at a time, guarded by _job_lock.
_JOB_IDLE = _flow.JOB_IDLE
_JOB_RUNNING = _flow.JOB_RUNNING
_JOB_SUCCESS = _flow.JOB_SUCCESS
_JOB_ERROR = _flow.JOB_ERROR

_job_lock = threading.Lock()
# Covers the complete operation, including release/backend resolution before the
# worker starts. _job_lock only protects the status dictionary and must never be
# held across network work.
_operation_lock = threading.Lock()
_job: dict = _flow.new_job()
_ALREADY_RUNNING_MESSAGE = "Another llama.cpp install is already running."

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
    """Best-effort build number from ``llama-server --version``.

    Current llama.cpp reports a semantic version followed by ``build NNNN``;
    older binaries put the build directly after ``version:``. None when
    unparseable or <= 1: a source build with no git tags reports build 1,
    which we treat as unknown (offer update).
    """
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 20,
        )
    except Exception:  # pragma: no cover - defensive
        return None
    output = "\n".join((proc.stderr or "", proc.stdout or ""))
    m = re.search(r"version:[^\r\n]*\bbuild\s+(\d+)\b", output)
    if not m:
        m = re.search(r"version:\s*(\d+)\b(?!\.)", output)
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
    # llama_tag is the upstream bNNNN base whose numeric part matches the build
    # field in --version; release_tag is the full tag, either a same-base mix
    # (bNNNN-mix-<sha>) or a fork wrapper (e.g. v1.0). Compare the numeric base
    # against llama_tag.
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


def _studio_custom_path_active() -> bool:
    """True when Settings, rather than the managed installer, owns the runtime."""
    try:
        from utils.llama_cpp_path_settings import custom_llama_cpp_path_source
        return custom_llama_cpp_path_source() == "studio"
    except Exception:
        return False


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


def _llama_only_status(
    *, force_refresh: bool = False, allow_source_probe_while_running: bool = False
) -> dict:
    """The llama.cpp half of get_update_status (no whisper sub-status)."""
    binary = _find_binary()
    # A path selected in Settings is user-managed even if its folder happens to
    # contain an Unsloth prebuilt marker. Never offer to replace that tree.
    if _studio_custom_path_active():
        return _local_link_status()
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
    if (
        marker is None
        and binary is not None
        and (not job_running or allow_source_probe_while_running)
    ):
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


# Cache resolver results between polls and clear them after an install.
_backends_memo: dict = {}


def _resolve_backends_for_host(
    install_dir: Optional[Path],
    *,
    force_refresh: bool = False,
    published_repo: Optional[str] = None,
) -> Optional[dict]:
    """Ask the installer which backends it could install here. None on any failure."""
    args: list[str] = []
    if install_dir is not None:
        args.extend(("--install-dir", str(install_dir)))
    if published_repo:
        args.extend(("--published-repo", published_repo))
    return _flow.resolve_prebuilt_for_host(
        force_refresh = force_refresh,
        memo = _backends_memo,
        installer_script = lambda: _installer_script(),
        log_message = "llama backend: resolve-backends failed",
        mode = ("--resolve-backends", "latest"),
        extra_args = tuple(args),
    )


def _switch_support(binary: Optional[str], marker: Optional[dict]) -> Optional[str]:
    """Return why this install cannot be switched, or None if it can."""
    if _studio_custom_path_active():
        return "custom_path"
    if binary is None:
        return "not_installed"
    if _active_install_is_local_link(binary):
        return "local_link"
    if marker is None:
        return "source_build"
    if _install_dir_for(binary) is None:
        return "no_install_dir"
    return None


def _env_backend_override() -> Optional[str]:
    """Return the environment override that outranks the picker."""
    return environment_backend_override(
        os.environ.get("UNSLOTH_LLAMA_CPP_BACKEND"),
        os.environ.get("UNSLOTH_FORCE_VULKAN"),
    )


def _backend_options(resolved: Optional[dict], assets: Optional[dict] = None) -> list[dict]:
    """Normalize a resolver payload into the options the picker and the switch
    planner both read, so both judge a request against the same list."""
    options = []
    for entry in (resolved or {}).get("backends") or []:
        backend = normalize_backend(entry.get("backend"))
        if backend is None:
            continue
        asset = entry.get("asset")
        options.append(
            {
                "backend": backend,
                "available": bool(entry.get("available")),
                "unavailable_reason": entry.get("reason"),
                # What "auto" would pick right now, so it can be labelled with it.
                "resolved_backend": normalize_backend(entry.get("resolved_backend")),
                "release_tag": entry.get("release_tag"),
                "download_size_bytes": (assets or {}).get(asset) if asset else None,
            }
        )
    return options


def _selection_applied(
    backend_request: str, installed_backend: Optional[str], options: list[dict]
) -> bool:
    """Whether the recorded choice still describes the installed backend.

    A concrete choice is applied by definition: the installer records it only on
    an install that honoured it, so it can never disagree with what is on disk.
    ``auto`` is the one that drifts -- a GPU or driver that appears after an
    automatic CPU install makes detection resolve somewhere else -- and applying
    it again is offered exactly then.
    """
    if backend_request != "auto":
        return True
    auto = next((option for option in options if option["backend"] == "auto"), None)
    if not auto or not auto.get("available"):
        return True
    resolved = auto.get("resolved_backend")
    return resolved is None or installed_backend is None or resolved == installed_backend


def get_backend_status(*, force_refresh: bool = False) -> dict:
    """Return the installed backend and host-compatible alternatives."""
    binary = _find_binary()
    marker = read_install_marker(binary)
    unsupported = _switch_support(binary, marker)
    with _job_lock:
        job = dict(_job)
    status: dict = {
        "supported": unsupported is None,
        "reason": unsupported,
        "env_backend": _env_backend_override(),
        "backend": marker_backend(marker),
        "backend_request": marker_backend_request(marker),
        # Only the resolver can tell that an automatic choice has drifted, so an
        # unresolved status must not invite an apply.
        "selection_applied": True,
        "installed_tag": (marker or {}).get("release_tag") or (marker or {}).get("tag"),
        "options": [],
        "job": job,
    }
    if unsupported is not None:
        return status
    # Do not resolve options while the shared job is replacing the install.
    if job["state"] == _JOB_RUNNING:
        return status
    repo = (marker or {}).get("published_repo") or DEFAULT_PUBLISHED_REPO
    resolved = _resolve_backends_for_host(
        _install_dir_for(binary), force_refresh = force_refresh, published_repo = repo
    )
    if not resolved:
        # Keep showing the installed backend without guessing alternatives.
        status["reason"] = "unresolved"
        status["supported"] = False
        return status
    assets = None
    try:
        assets = latest_release_assets(repo, force_refresh = force_refresh)
    except Exception as exc:  # pragma: no cover - network defensive
        logger.debug("llama backend: asset size lookup failed", error = str(exc))
    options = _backend_options(resolved, assets)
    status["options"] = options
    status["selection_applied"] = _selection_applied(
        status["backend_request"], status["backend"], options
    )
    return status


def _run_llama_phase(
    install_dir: Path,
    repo: str,
    asset: Optional[str],
    script: Path,
    pin_release_tag: Optional[str],
    set_progress,
    llama_backend: Optional[str] = None,
    rocm_gfx: Optional[str] = None,
    backend_request: Optional[str] = None,
) -> dict:
    """The llama phase of a chained update: put the backend into a maintenance
    state, run the installer for the latest prebuilt, then refresh caches so the
    next load uses the new build. Returns {to_tag, reload_required, message};
    raises on failure.

    pin_release_tag pins the installer to that exact published release instead
    of letting it re-resolve "latest" itself (see start_update for why)."""
    backend = None
    model_was_active = False
    mtmd_guard = ExitStack()
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

        # The mtmd dictation sidecar serves Qwen3-ASR from this same llama-server
        # out of this same tree, so a live one locks the exe on Windows and a
        # concurrent load would start against a half-swapped install.
        model_was_active = _block_mtmd_sidecar(mtmd_guard) or model_was_active

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
        # Switches name a backend. Updates preserve the marker's recorded choice.
        if backend_request is not None:
            cmd.extend(["--llama-backend", backend_request])
        logger.info("llama update: installing", cmd = " ".join(cmd))
        env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
        if backend_request is not None:
            # Ensure an inherited override cannot defeat the requested switch.
            env.pop("UNSLOTH_FORCE_VULKAN", None)
            env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend_request
        # A Vulkan asset name carries no arch, so the marker is the only record of the
        # gfx an automatic AMD route used. Automatic only: elsewhere the asset has the
        # arch, and replaying it would assert ROCm on a host whose AMD GPU is gone.
        # Advisory even then, applied only if this host's own probe finds none.
        if rocm_gfx and llama_backend == "auto":
            env["UNSLOTH_ROCM_GFX_REMEMBERED"] = rocm_gfx
        _flow.stream_installer(
            cmd,
            env,
            set_progress = set_progress,
            timeout_seconds = _INSTALL_TIMEOUT_SECONDS,
        )

        # Drop stale caches so the banner re-checks the swapped marker.
        # If GitHub is offline, latest stays unknown and the banner fails open.
        reset_caches(drop_disk = True)
        # The cached options describe the previous install.
        _backends_memo.clear()
        try:
            latest_published_release(repo, force_refresh = True)
        except Exception as exc:  # pragma: no cover - network defensive
            logger.debug("llama update: post-install freshness refresh failed", error = str(exc))
        new_marker = read_install_marker(_find_binary())
        new_tag = (new_marker or {}).get("release_tag") or (new_marker or {}).get("tag")
        new_backend = marker_backend(new_marker)
        new_backend_request = marker_backend_request(new_marker)

        new_repo = (new_marker or {}).get("published_repo")
        if pin_release_tag and backend_request is not None:
            if new_repo != repo or new_tag != pin_release_tag:
                raise RuntimeError(
                    "backend switch must preserve "
                    f"{repo}@{pin_release_tag}, but installer produced "
                    f"{new_repo or 'an unknown repository'}@{new_tag or 'an unknown release'}"
                )
        elif pin_release_tag and new_tag and new_repo == repo and new_tag != pin_release_tag:
            raise RuntimeError(f"pinned release {pin_release_tag} but installer produced {new_tag}")

        if backend_request is not None:
            if new_backend is None:
                raise RuntimeError(
                    f"requested {backend_request} but the installed backend is unknown"
                )
            if new_backend_request != backend_request:
                raise RuntimeError(
                    f"requested {backend_request} but the installer recorded "
                    f"{new_backend_request or 'an unknown selection'}"
                )
            if backend_request != "auto" and new_backend != backend_request:
                raise RuntimeError(
                    f"requested {backend_request} but the installer produced "
                    f"{new_backend or 'an unknown backend'}"
                )

        logger.info("llama update: success", to_tag = new_tag, backend = new_backend)
        reload_hint = " Reload your model to use it." if model_was_active else ""
        return {
            "to_tag": new_tag,
            "backend": new_backend,
            "reload_required": model_was_active,
            "message": (
                f"llama.cpp is now running on {new_backend or backend_request}.{reload_hint}"
                if backend_request is not None
                else f"Updated llama.cpp to {new_tag}.{reload_hint}"
            ),
        }
    except _flow.InstallerExit as exc:
        # Raw "installer exited 4: <log tail>" says nothing actionable in the UI.
        if exc.returncode == _EXIT_NO_SPACE:
            logger.warning("llama update: out of disk space")
            raise _LlamaPhaseError(
                "Not enough disk space to install llama.cpp. Free up space or point "
                "UNSLOTH_STUDIO_HOME/TMPDIR at a larger volume, then retry.",
                reload_required = model_was_active,
            ) from exc
        if exc.returncode == _EXIT_BACKEND_UNAVAILABLE:
            # This can race hardware or release changes after option resolution.
            failed_backend = backend_request or _env_backend_override()
            backend_label = (
                failed_backend
                if failed_backend is not None and failed_backend != "auto"
                else "requested"
            )
            logger.warning("llama update: backend unavailable", backend = failed_backend)
            raise _LlamaPhaseError(
                f"Could not install the {backend_label} llama.cpp build on this machine. "
                "The installed backend was kept.",
                reload_required = model_was_active,
            ) from exc
        if exc.returncode == _EXIT_FALLBACK:
            message = str(exc)
            lowered = message.lower()
            if (
                "github api returned 403" in lowered
                or "rate limit" in lowered
                or "gh_token" in lowered
            ):
                logger.warning("llama update: GitHub rate limit")
                raise _LlamaPhaseError(
                    "Could not update llama.cpp: GitHub is rate-limiting release downloads. "
                    "Set GH_TOKEN or GITHUB_TOKEN in your environment and try again.",
                    reload_required = model_was_active,
                ) from exc
            logger.warning("llama update: prebuilt fallback", error = message)
            detail = message.split(": ", 1)[-1] if ": " in message else message
            raise _LlamaPhaseError(
                f"Could not update llama.cpp from the prebuilt bundle. {detail}",
                reload_required = model_was_active,
            ) from exc
        logger.warning("llama update: failed", error = str(exc))
        raise _LlamaPhaseError(str(exc), reload_required = model_was_active) from exc
    except Exception as exc:
        logger.warning("llama update: failed", error = str(exc))
        if isinstance(exc, _LlamaPhaseError):
            raise
        if model_was_active:
            raise _LlamaPhaseError(str(exc), reload_required = True) from exc
        raise
    finally:
        # Always clear maintenance state.
        mtmd_guard.close()
        if backend is not None:
            try:
                backend._llama_update_in_progress = False
            except Exception:  # pragma: no cover - defensive
                pass


def _block_mtmd_sidecar(stack: ExitStack) -> bool:
    """Hold the mtmd sidecar's maintenance guard for the install, if it exists.

    Unlike whisper.cpp this is not fail-closed: llama.cpp updates predate this
    sidecar and must keep working where dictation cannot even be imported.
    Returns whether a warm dictation server had to be unloaded.
    """
    try:
        from core.inference.stt_mtmd_sidecar import get_mtmd_stt_sidecar
        return stack.enter_context(get_mtmd_stt_sidecar().update_maintenance())
    except Exception as exc:  # noqa: BLE001 - the update proceeds without it
        logger.debug("llama update: mtmd coordination failed", error = str(exc))
        return False


# Combined-job progress split when both phases run (download sizes: the llama
# bundle dwarfs the whisper one); normalized to 0..1 when a phase is skipped.
_LLAMA_PHASE_WEIGHT = 0.7
_WHISPER_PHASE_WEIGHT = 0.3


def _plan_llama_phase(backend_request: Optional[str] = None) -> dict:
    """Plan the llama phase for an update or backend switch."""
    binary = _find_binary()
    if _studio_custom_path_active():
        return {
            "skip_reason": "custom_path",
            "refusal": {
                "started": False,
                "reason": "custom_path",
                "message": (
                    "llama.cpp is using the custom folder selected in Settings; "
                    "Unsloth won't replace it. Update that build yourself or restore "
                    "the bundled runtime first."
                ),
            },
        }
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
        # A switch replaces the backend at the same release.
        status = (
            {}
            if backend_request is not None
            else _llama_only_status(
                force_refresh = True,
                allow_source_probe_while_running = True,
            )
        )
        if backend_request is None and not status.get("update_available"):
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
        # Updates let the installer preserve the marker's recorded choice.
        llama_backend = marker.get("llama_backend")
        rocm_gfx = marker.get("rocm_gfx")
        # Install exactly the release the banner offered: the installer's own
        # "latest" is commit-date ordered and can lag the published_at pick
        # above, reinstalling the current build in a loop (the #6219 class).
        # Not on macOS, which needs the older-release walk-back a pin disables
        # (skipping too-new prebuilts); elsewhere an unusable latest now fails
        # the job loudly (retryable) instead of walking back.
        #
        # Switch only the backend. Slim whisper bundles require this exact release.
        installed_release_tag = marker.get("release_tag") or marker.get("tag")
        wanted_tag = (
            installed_release_tag if backend_request is not None else status.get("latest_tag")
        )
        pin_release_tag = None if sys.platform == "darwin" else wanted_tag
    elif backend_request is not None:
        # Never replace a user-managed tree with a prebuilt implicitly.
        return {
            "skip_reason": "not_prebuilt",
            "refusal": {
                "started": False,
                "reason": "not_prebuilt",
                "message": (
                    "This llama.cpp install was not made from an Unsloth prebuilt, so "
                    "its backend cannot be switched from here."
                ),
            },
        }
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
        # A source build records no choice, so there is nothing to preserve here.
        llama_backend = None
        rocm_gfx = None
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
            "llama_backend": llama_backend,
            "rocm_gfx": rocm_gfx,
            "backend_request": backend_request,
        }
    }


def start_update() -> dict:
    """Kick off a background update job. The job chains the llama phase (the
    existing flow) with a whisper phase that runs only when whisper is actually
    behind; either phase no-ops cleanly when its component is current or
    unmanaged. Idempotent: a second call while one is running returns the
    in-flight job rather than starting another."""
    return _start_llama_job()


def start_backend_switch(backend: str) -> dict:
    """Install the llama.cpp bundle for ``backend`` and record it as the choice.

    Deliberately the same job, lock and phase machinery as an update: both replace
    the same tree while the same server is unloaded, so sharing them is what makes a
    switch and an update mutually exclusive instead of two writers racing over one
    install. Callers poll get_update_status() exactly as they do for an update.
    """
    normalized = normalize_backend(backend)
    if normalized is None or normalized not in REQUESTABLE_BACKENDS:
        with _job_lock:
            job = dict(_job)
        return {
            "started": False,
            "reason": "unknown_backend",
            "message": f"{backend!r} is not a llama.cpp backend Unsloth can install.",
            "job": job,
        }
    env_backend = _env_backend_override()
    if env_backend is not None:
        with _job_lock:
            job = dict(_job)
        return {
            "started": False,
            "reason": "environment_override",
            "message": (
                f"llama.cpp is controlled by the {env_backend} environment override. "
                "Unset it and restart Unsloth before switching backends here."
            ),
            "job": job,
        }

    return _start_llama_job(backend_request = normalized)


def _whisper_phase_plan(
    backend_request: Optional[str],
    *,
    llama_will_run: bool,
    llama_skip_reason: Optional[str] = None,
) -> dict:
    """Whisper's half of the chained job: catch up on releases for an update, or
    re-pair with the new backend for a switch.

    A switch that installs nothing leaves llama's ggml where it was, so there is
    nothing to re-pair either: a refused switch stays refused instead of turning into
    a whisper-only job.

    The exception is the state a failed re-pair leaves behind. The llama phase runs
    first and records the new backend, so a retryable whisper failure (a download that
    dropped, an install that was busy) ends with llama on the requested backend and
    dictation still hardlinked to the old runtime. Retrying that selection is then
    ``already_selected``, and refusing it strands the user: the only escape is to switch
    away and back. So allow a repair-only job for that one refusal, and only while the
    pairing is genuinely stale, which keeps an ordinary already-selected request a
    refusal rather than a no-op job reporting success."""
    if backend_request is None:
        return (
            _whisper_chain_status(force_refresh = True, paired_llama_will_update = llama_will_run) or {}
        )
    try:
        from utils import whisper_cpp_update
        if not llama_will_run:
            if llama_skip_reason != "already_selected":
                return {}
            if not whisper_cpp_update.slim_pairing_is_stale():
                return {}
        return whisper_cpp_update.repair_pairing_plan()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llama switch: whisper repair probe failed", error = str(exc))
        return {}


def _claim_operation(backend_request: Optional[str]) -> bool:
    """Reserve the operation and publish its planning state atomically."""
    with _job_lock:
        if not _operation_lock.acquire(blocking = False):
            return False
        if _job["state"] == _JOB_RUNNING:
            _operation_lock.release()
            return False
        _job.update(_flow.new_job())
        _job.update(
            state = _JOB_RUNNING,
            operation = "switch" if backend_request is not None else "update",
            requested_backend = backend_request,
            message = (
                f"Checking the {backend_request} llama.cpp build..."
                if backend_request is not None
                else "Checking for llama.cpp updates..."
            ),
            progress = 0.0,
            started_at = _utcnow(),
        )
    return True


def _finish_planning_refusal(reason: str, message: str) -> dict:
    """Publish a terminal result for an operation that stopped during planning."""
    succeeded = reason in {"up_to_date", "already_selected"}
    with _job_lock:
        _job.update(
            state = _JOB_SUCCESS if succeeded else _JOB_ERROR,
            message = message,
            error = None if succeeded else message,
            progress = 1.0 if succeeded else None,
            finished_at = _utcnow(),
        )
        job = dict(_job)
    return {"started": False, "reason": reason, "message": message, "job": job}


def _run_claimed_job(phases: list[dict]) -> None:
    """Run a planned job and always release its full-operation reservation."""
    try:
        _flow.run_chained_update(phases, job = _job, job_lock = _job_lock)
    except Exception as exc:  # pragma: no cover - phase failures are handled inside the runner
        logger.exception("llama update: job runner failed", error = str(exc))
        with _job_lock:
            _job.update(
                state = _JOB_ERROR,
                message = "llama.cpp install failed.",
                error = str(exc),
                finished_at = _utcnow(),
            )
    finally:
        _operation_lock.release()


def _start_llama_job(backend_request: Optional[str] = None) -> dict:
    """Shared body of start_update() and start_backend_switch()."""
    if not _claim_operation(backend_request):
        with _job_lock:
            return {
                "started": False,
                "reason": "already_running",
                "message": _ALREADY_RUNNING_MESSAGE,
                "job": dict(_job),
            }

    handed_to_worker = False
    try:
        # The operation reservation starts before any marker read or resolver
        # call. A second update/switch cannot change the install between this
        # plan and the worker that applies it.
        llama_plan = _plan_llama_phase(backend_request)
        llama_spec = llama_plan.get("spec")
        if backend_request is not None and llama_spec is not None:
            resolved = _resolve_backends_for_host(
                llama_spec["install_dir"],
                force_refresh = True,
                published_repo = llama_spec["repo"],
            )
            if not resolved:
                return _finish_planning_refusal(
                    "unresolved",
                    "Could not verify the available llama.cpp backends. Try again online.",
                )
            options = _backend_options(resolved)
            option = next((o for o in options if o["backend"] == backend_request), None)
            if (
                option is None
                or not option["available"]
                or (backend_request != "auto" and option["resolved_backend"] != backend_request)
            ):
                return _finish_planning_refusal(
                    "backend_unavailable",
                    f"No {backend_request} llama.cpp build is available for this machine.",
                )
            marker = read_install_marker(_find_binary())
            if marker_backend_request(marker) == backend_request and _selection_applied(
                backend_request, marker_backend(marker), options
            ):
                llama_plan = {
                    "skip_reason": "already_selected",
                    "refusal": {
                        "started": False,
                        "reason": "already_selected",
                        "message": f"llama.cpp is already set to {backend_request}.",
                    },
                }
                llama_spec = None

        whisper_plan = _whisper_phase_plan(
            backend_request,
            llama_will_run = llama_spec is not None,
            llama_skip_reason = llama_plan.get("skip_reason"),
        )
        whisper_spec = (whisper_plan or {}).get("phase")
        if llama_spec is None and whisper_spec is None:
            # Nothing to run: answer with the llama refusal so the existing reasons
            # (local_link / up_to_date / already_selected / ...) keep their meaning.
            refusal = llama_plan["refusal"]
            return _finish_planning_refusal(refusal["reason"], refusal["message"])

        whisper_run = None
        if whisper_spec is not None:
            from utils import whisper_cpp_update as _whisper
            whisper_run = (
                (lambda set_progress: _whisper.run_repair_phase(whisper_spec, set_progress))
                if whisper_spec.get("repair")
                else (lambda set_progress: _whisper.run_chained_phase(whisper_spec, set_progress))
            )

        phases = [
            {
                "name": "llama",
                "weight": _LLAMA_PHASE_WEIGHT,
                "failure_message": (
                    f"Could not switch llama.cpp to {backend_request}."
                    if backend_request is not None
                    else "llama.cpp update failed."
                ),
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
                            llama_backend = llama_spec.get("llama_backend"),
                            rocm_gfx = llama_spec.get("rocm_gfx"),
                            backend_request = llama_spec.get("backend_request"),
                        )
                    )
                    if llama_spec
                    else None
                ),
            },
            {
                "name": "whisper",
                "weight": _WHISPER_PHASE_WEIGHT,
                "failure_message": (
                    "whisper.cpp could not be re-paired with the new backend."
                    if backend_request is not None
                    else "whisper.cpp update failed."
                ),
                # The sidecar reload is whisper-internal; it must not trip the
                # job-level reload flag the chat frontend resyncs on.
                "affects_job_reload": False,
                "skip_reason": (whisper_plan or {}).get("skip_reason") or "unavailable",
                "run": whisper_run,
            },
        ]
        running = " + ".join(
            name
            for name, spec in (("llama.cpp", llama_spec), ("whisper.cpp", whisper_spec))
            if spec
        )
        if backend_request is not None and llama_spec is not None:
            starting_message = f"Installing the {backend_request} llama.cpp build..."
        elif backend_request is not None:
            starting_message = "Re-pairing whisper.cpp with llama.cpp..."
        else:
            starting_message = f"Downloading and installing the latest {running} prebuilt..."

        with _job_lock:
            _job.update(
                message = starting_message,
                from_tag = (llama_spec or {}).get("from_tag"),
            )
            job_snapshot = dict(_job)

        thread = threading.Thread(
            target = _run_claimed_job,
            args = (phases,),
            name = "llama-cpp-backend-switch" if backend_request else "llama-cpp-update",
            daemon = True,
        )
        try:
            thread.start()
        except Exception as exc:  # pragma: no cover - interpreter resource failure
            with _job_lock:
                _job.update(
                    state = _JOB_ERROR,
                    message = "Could not start the llama.cpp install worker.",
                    error = str(exc),
                    finished_at = _utcnow(),
                )
                failed_job = dict(_job)
            return {
                "started": False,
                "reason": "worker_start_failed",
                "message": "Could not start the llama.cpp install worker.",
                "job": failed_job,
            }
        handed_to_worker = True
        return {"started": True, "reason": None, "job": job_snapshot}
    except Exception as exc:
        with _job_lock:
            _job.update(
                state = _JOB_ERROR,
                message = "Could not plan the llama.cpp install.",
                error = str(exc),
                finished_at = _utcnow(),
            )
        raise
    finally:
        if not handed_to_worker:
            _operation_lock.release()


def _reset_job_for_tests() -> None:
    """Test-only: return the job tracker to idle."""
    _flow.reset_job(_job, _job_lock)
