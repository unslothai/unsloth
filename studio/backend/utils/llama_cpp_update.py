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
    latest_release_assets,
    parse_base_build,
    read_install_marker,
    reset_caches,
    update_download_size_bytes,
)
from utils.process_lifetime import child_popen_kwargs

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
    "reload_required": None,
    "error": None,
    "progress": None,
    "started_at": None,
    "finished_at": None,
}

# Matches the installer's download progress lines, e.g.
# "Downloading x.zip:  35.0% (12.3 MiB/35.1 MiB) at 8.2 MiB/s".
_PROGRESS_LINE_RE = re.compile(r"(\d+(?:\.\d+)?)%\s*\(")
# The download dominates the update; extract/validate fill the last slice.
_DOWNLOAD_PROGRESS_CEILING = 0.95


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


def _is_under(path: Path, root: Path) -> bool:
    try:
        p, r = path.resolve(), root.resolve()
    except (OSError, ValueError):
        p, r = path, root
    return p == r or r in p.parents


def _llama_install_root(binary: Optional[str]) -> Optional[Path]:
    """The Studio-managed llama.cpp root the active binary lives under, or None
    when the binary is unmanaged. Installing anywhere the active binary is not
    would not replace what _find_llama_server_binary runs (which prefers a pinned
    LLAMA_SERVER_PATH, then UNSLOTH_LLAMA_CPP_PATH, then a llama.cpp tree), so we
    refuse rather than silently install into an inactive or foreign tree."""
    marked = _install_dir_for(binary)
    if marked is not None:
        return marked
    if not binary:
        return None
    # LLAMA_SERVER_PATH is an explicit user pin that always wins in discovery;
    # never auto-replace its tree (even a user's own llama.cpp checkout).
    if os.environ.get("LLAMA_SERVER_PATH"):
        return None
    p = Path(binary)
    env = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    if env and _is_under(p, Path(env)):
        return Path(env)
    for parent in p.parents:
        if parent.name == "llama.cpp":
            return parent
    # PATH / system / custom install: not a managed tree, so do not offer.
    return None


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


def _is_external_link(path: Optional[Path]) -> bool:
    """True when ``path`` is a --with-llama-cpp-dir local link: a POSIX symlink
    or a Windows directory junction / reparse point. Such a link resolves into
    the user's own llama.cpp checkout, so Studio must never auto-update it."""
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


def _active_install_is_local_link(binary: Optional[str]) -> bool:
    """True when the active llama-server resolves through a --with-llama-cpp-dir
    local link at the canonical llama.cpp directory. An update would write
    through that link into the user's own checkout (or fail), so the install is
    treated as externally managed: no update is offered or applied. Checks only
    up to and including the ``llama.cpp`` dir so a symlinked HOME / studio root
    above it can't trip a false positive."""
    if not binary:
        return False
    for parent in Path(binary).parents:
        if _is_external_link(parent):
            return True
        if parent.name == "llama.cpp":
            break
    return False


def _local_link_status() -> dict:
    """Status payload for a local-link install: unmanaged, no update offered."""
    with _job_lock:
        job = dict(_job)
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
        "job": job,
    }


def get_update_status(*, force_refresh: bool = False) -> dict:
    """Report whether a newer prebuilt exists plus the current job state.

    force_refresh bypasses the 24h release cache for an explicit "check now".
    """
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


def _rocm_install_args(asset: Optional[str]) -> list[str]:
    """Forward --rocm-gfx/--has-rocm from the marker asset, mirroring setup.sh.
    The installer probe can miss the gfx arch on amd-smi-only hosts; per-gfx
    ROCm bundles carry the family in the name (rocm-gfx110X), version-tagged
    bundles only rocm/hip."""
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


def _run_update(
    install_dir: Path,
    repo: str,
    asset: Optional[str],
    script: Path,
    pin_release_tag: Optional[str] = None,
    force_cpu: bool = False,
) -> None:
    """Worker: put the backend into a maintenance state, run the installer for
    the latest prebuilt, then refresh caches so the next load uses the new build.

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
        logger.info("llama update: installing", cmd = " ".join(cmd))
        # Stream progress lines into job["progress"].
        env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
        # Preserve a Vulkan install across updates: detect_host on a CUDA/ROCm
        # box would otherwise re-route and silently replace the Vulkan build.
        # Re-assert it via the same env flag setup uses (mirrors
        # _rocm_install_args).
        if asset and "vulkan" in asset.lower():
            env["UNSLOTH_FORCE_VULKAN"] = "1"
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
                m = _PROGRESS_LINE_RE.search(line)
                if m is None:
                    continue
                fraction = min(float(m.group(1)) / 100.0, 1.0) * _DOWNLOAD_PROGRESS_CEILING
                with _job_lock:
                    _job["progress"] = max(_job.get("progress") or 0.0, fraction)
            returncode = proc.wait()
        finally:
            watchdog.cancel()
        if timed_out.is_set():
            raise RuntimeError(f"installer timed out after {_INSTALL_TIMEOUT_SECONDS}s")
        if returncode != 0:
            tail = "".join(tail_lines).strip()[-1500:]
            raise RuntimeError(f"installer exited {returncode}: {tail or 'no output'}")

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

        with _job_lock:
            _job.update(
                state = _JOB_SUCCESS,
                message = (
                    f"Updated llama.cpp to {new_tag}."
                    + (" Reload your model to use it." if model_was_active else "")
                ),
                to_tag = new_tag,
                reload_required = model_was_active,
                error = None,
                progress = 1.0,
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
        # Always clear maintenance state.
        if backend is not None:
            try:
                backend._llama_update_in_progress = False
            except Exception:  # pragma: no cover - defensive
                pass


def start_update() -> dict:
    """Kick off a background update. Idempotent: a second call while one is
    running returns the in-flight job rather than starting another."""
    binary = _find_binary()
    # Refuse to update a --with-llama-cpp-dir local link: installing a prebuilt
    # here would write through the link into the user's own checkout (or fail)
    # and silently drop the link the flag created.
    if _active_install_is_local_link(binary):
        return {
            "started": False,
            "reason": "local_link",
            "message": (
                "llama.cpp is a local directory linked with --with-llama-cpp-dir; "
                "Studio won't replace it. Update your own llama.cpp checkout instead."
            ),
            "job": get_update_status()["job"],
        }
    marker = read_install_marker(binary)
    script = _installer_script()
    if script is None:
        return {
            "started": False,
            "reason": "installer_missing",
            "message": "install_llama_prebuilt.py could not be located.",
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
                "message": "The installed llama.cpp build is already at the latest prebuilt.",
                "job": status["job"],
            }
        install_dir = _install_dir_for(binary)
        repo = marker.get("published_repo") or DEFAULT_PUBLISHED_REPO
        from_tag = marker.get("tag") or marker.get("release_tag")
        asset = marker.get("asset")
        force_cpu = bool(marker.get("force_cpu"))
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
                "started": False,
                "reason": "no_prebuilt_available",
                "message": (
                    "No official llama.cpp prebuilt is available for this host, "
                    "so the source build cannot be swapped automatically."
                ),
                "job": get_update_status()["job"],
            }
        if not src.get("update_available"):
            return {
                "started": False,
                "reason": "up_to_date",
                "message": "The installed llama.cpp build is already at or newer than the latest prebuilt.",
                "job": get_update_status()["job"],
            }
        res = _resolve_prebuilt_for_host()
        install_dir = _llama_install_root(binary)
        repo = (res or {}).get("repo") or DEFAULT_PUBLISHED_REPO
        from_tag = None
        asset = (res or {}).get("asset")
        # Source builds carry no forced-CPU marker, so nothing to preserve here.
        force_cpu = False
        # No pin: source-build detection resolves via --resolve-prebuilt latest,
        # the same resolver the unpinned apply uses, so the two already agree.
        pin_release_tag = None

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
            reload_required = None,
            error = None,
            progress = 0.0,
            started_at = _utcnow(),
            finished_at = None,
        )
        job_snapshot = dict(_job)

    thread = threading.Thread(
        target = _run_update,
        args = (install_dir, repo, asset, script, pin_release_tag, force_cpu),
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
            reload_required = None,
            error = None,
            progress = None,
            started_at = None,
            finished_at = None,
        )
