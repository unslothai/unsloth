# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app audio.cpp runtime update: the third phase of the combined llama.cpp update item (utils.llama_cpp_update chains run_chained_phase after the whisper phase). Like llama.cpp and whisper.cpp, the target tracks the newest unslothai/audio.cpp release (or the release UNSLOTH_AUDIO_CPP_TAG names, which needs no network call), looked up through utils.audio_cpp_freshness. Only the Unsloth Studio-managed tree is offered, the same predicate the Audio page notice reads. The mechanics (the streamed installer run, the chained job) live in utils.prebuilt.update_flow; this module keeps the audio.cpp policy."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

import structlog

from utils import audio_cpp_freshness as _freshness
from utils.prebuilt import update_flow as _flow
from utils.update_status import update_checks_disabled

logger = structlog.get_logger(__name__)

_INSTALL_TIMEOUT_SECONDS = 1800
# setup.sh / setup.ps1 leave the managed tree alone when any of these is set, so an update would not take effect.
_SETUP_SKIP_VARS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH")
_EXIT_BUSY = 3
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_VERSION_COMMIT = re.compile(r"^git: ([0-9a-f]{7,40})\b", re.MULTILINE)
_backfill_started = False
_backfill_lock = threading.Lock()


class _AudioPhaseError(RuntimeError):
    """An audio phase failed after possibly unloading audio.cpp models."""

    def __init__(self, message: str, *, reload_required: bool):
        super().__init__(message)
        self.reload_required = reload_required


def _installer():
    studio_dir = str(Path(__file__).resolve().parents[2])
    if studio_dir not in sys.path:
        sys.path.insert(0, studio_dir)
    import install_audio_cpp_prebuilt

    return install_audio_cpp_prebuilt


def _target() -> tuple[str, Optional[str]]:
    """``(repo, pin)`` the installer would install from: the configured repo, and the release UNSLOTH_AUDIO_CPP_TAG pins (None tracks the newest)."""
    installer = _installer()
    return installer._repo(), installer._pinned_tag()


def _installer_script() -> Optional[Path]:
    return _flow.find_installer_script(
        env_var = "UNSLOTH_AUDIO_CPP_INSTALLER", script_name = "install_audio_cpp_prebuilt.py"
    )


def _is_managed(binary: str) -> bool:
    """Whether ``binary`` is the Studio-owned tree setup would replace."""
    from core.inference import audio_cpp_server

    if any(os.environ.get(name) for name in _SETUP_SKIP_VARS):
        return False
    if os.environ.get("UNSLOTH_SKIP_AUDIO_CPP_INSTALL") == "1":
        return False
    managed_dir = audio_cpp_server.managed_audio_cpp_dir().resolve()
    return (
        Path(binary).resolve().is_relative_to(managed_dir)
        and (managed_dir / ".unsloth-studio-owned").is_file()
    )


def _tag(record: dict) -> Optional[str]:
    tag = record.get("release_tag")
    return tag if isinstance(tag, str) and tag else None


def get_installed_audio_cpp_version() -> Optional[str]:
    """Display string for the active audio.cpp runtime: the release tag its install record names (e.g. 'v0.9.1-unsloth.1'), or None for no runtime or a custom build without a record. Reads only that record, no network and no exec, so it is safe on the About panel's path."""
    from core.inference import audio_cpp_server
    try:
        binary = audio_cpp_server.find_audio_cpp_server_binary()
        return _tag(audio_cpp_server.read_install_record(binary)) if binary else None
    except Exception as exc:  # noqa: BLE001 - the About panel omits the row rather than failing
        logger.debug("audio.cpp version probe failed: %s", exc)
        return None


def release_status(binary: str, record: dict) -> dict:
    """``expected_tag`` and ``outdated`` for an installed runtime: the release an update would install, and whether the record is behind it or from another repo. Reads only the cached newest release, never the network, since the async Audio page status calls this. Raises when the installer cannot be read; callers treat that as current."""
    status = {"expected_tag": None, "outdated": False}
    tag = _tag(record)
    if tag is None or not _is_managed(binary):
        return status
    repo, pin = _target()
    if pin:
        status["expected_tag"] = pin
        status["outdated"] = (record.get("published_repo"), tag) != (repo, pin)
        return status
    if update_checks_disabled():
        return status
    latest = _freshness.cached_latest_release(repo)
    if latest is None:
        return status
    status["expected_tag"] = latest
    status["outdated"] = record.get("published_repo") != repo or _freshness.is_behind(tag, latest)
    return status


def _host_has_prebuilt(
    names: Optional[list],
    accelerator: Optional[str],
    explicit: Optional[str] = None,
) -> bool:
    """Whether the newest release publishes a bundle for this host. Unknown names (offline, or a pinned tag) can only be answered by the installer, so they count as available. An explicit request is never downgraded to CPU by the installer, so only it is checked."""
    if not names:
        return True
    installer = _installer()
    accels = (explicit,) if explicit else tuple(dict.fromkeys((accelerator or "cpu", "cpu")))
    return any(
        installer.resolve_release_asset(
            names,
            system = platform.system(),
            machine = platform.machine(),
            accelerator = accel,
        )
        for accel in accels
    )


def _update_size_bytes(
    assets: Optional[dict], record: dict, installed_tag: str, latest_tag: str
) -> Optional[int]:
    """Download size of the newest release's bundle for this install (plus its CUDA runtime zip when the install has one). The bundle is the recorded asset renamed to the new tag, else the installer's pick for the recorded accelerator. None on any miss."""
    if not assets:
        return None
    installer = _installer()
    names = list(assets)
    bundle = None
    asset = record.get("asset")
    prefix = f"audio-{installed_tag}-"
    if isinstance(asset, str) and asset.startswith(prefix):
        renamed = f"audio-{latest_tag}-{asset[len(prefix) :]}"
        if renamed in assets:
            bundle = renamed
    if bundle is None:
        accelerator = record.get("accelerator")
        accelerator = accelerator if isinstance(accelerator, str) else "cpu"
        # The installer's own constraints, so another repo's CUDA line maps onto the bundle it will pick.
        bundle = installer.resolve_release_asset(
            names,
            system = platform.system(),
            machine = platform.machine(),
            accelerator = accelerator,
            driver_cuda = installer.nvidia_driver_cuda_version() if accelerator == "cuda" else None,
            prefer_cuda_major = installer.torch_cuda_major(),
        )
    if bundle is None or bundle not in assets:
        return None
    size = assets[bundle]
    cudart = installer.cudart_asset_for(names, bundle)
    if cudart is not None:
        # The installer's own call: the runtime zip is skipped only when torch ships that CUDA major now.
        line = installer._CUDA_LINE.search(bundle)
        if not (line and installer.torch_provides_cuda_runtime(int(line.group(1)))):
            if cudart not in assets:
                return None
            size += assets[cudart]
    return size


def _release_assets(
    repo: str,
    latest: str,
    *,
    force_refresh: bool = False,
) -> Optional[dict]:
    assets = _freshness.latest_release_assets(repo, force_refresh = force_refresh)
    # The tag can come from the 24h disk cache while the asset list is fetched fresh, so after a newer release the two can disagree: keep only the offered release's assets.
    prefixes = (f"audio-{latest}-bin-", f"audio-{latest}-cudart-")
    return {
        name: size for name, size in (assets or {}).items() if name.startswith(prefixes)
    } or None


def _install_plan(plan: dict, *, force_refresh: bool = False) -> dict:
    """No runtime at all: offer a first install into the managed tree, but only where setup would have installed one and this host has a bundle for the known release. Otherwise ``plan`` stays not_installed."""
    from core.inference import audio_cpp_server

    skipped = any(os.environ.get(name) for name in _SETUP_SKIP_VARS)
    if (
        skipped
        or os.environ.get("UNSLOTH_SKIP_AUDIO_CPP_INSTALL") == "1"
        or update_checks_disabled()
    ):
        return plan
    repo, pin = _target()
    latest = pin or _freshness.latest_published_release(repo, force_refresh = force_refresh)
    assets = (
        None
        if pin or latest is None
        else _release_assets(repo, latest, force_refresh = force_refresh)
    )
    script = _installer_script()
    if (
        latest is None
        or script is None
        # The installer's own pick: a macOS release has Metal bundles only.
        or not _host_has_prebuilt(
            list(assets) if assets else None, _installer().detect_accelerator()
        )
    ):
        return plan
    plan.update(
        status = {"installed_tag": None, "latest_tag": latest, "update_size_bytes": None},
        update_available = True,
        skip_reason = None,
        phase = dict(
            install_dir = audio_cpp_server.managed_audio_cpp_dir(),
            script = script,
            accelerator = None,
            from_tag = None,
            to_tag = latest,
            pin_release_tag = latest,
            repo = repo,
        ),
    )
    return plan


def _plan(*, force_refresh: bool = False) -> dict:
    from core.inference import audio_cpp_server

    plan: dict = {"status": None, "update_available": False, "skip_reason": None, "phase": None}
    binary = audio_cpp_server.find_audio_cpp_server_binary()
    if binary is None:
        plan["skip_reason"] = "not_installed"
        return _install_plan(plan, force_refresh = force_refresh)
    record = audio_cpp_server.read_install_record(binary)
    status = {"installed_tag": _tag(record), "latest_tag": None, "update_size_bytes": None}
    plan["status"] = status
    if not _is_managed(binary):
        plan["skip_reason"] = "unmanaged"
        return plan
    repo, pin = _target()
    installed = status["installed_tag"]
    names = None
    if pin:
        # A pinned release is compared as is, with no lookup, and its assets stay unknown.
        status["latest_tag"] = pin
        if installed is None or (record.get("published_repo"), installed) == (repo, pin):
            plan["skip_reason"] = "up_to_date"
            return plan
        if update_checks_disabled():
            plan["skip_reason"] = "update_checks_disabled"
            return plan
    else:
        # Checked before the lookup: disabled checks make no GitHub call.
        if update_checks_disabled():
            plan["skip_reason"] = "update_checks_disabled"
            return plan
        latest = _freshness.latest_published_release(repo, force_refresh = force_refresh)
        status["latest_tag"] = latest
        # An unknown latest, or one that is not newer (the downgrade guard), is nothing to offer. A record from another repo is replaced by this repo's newest.
        if (
            installed is None
            or latest is None
            or (
                record.get("published_repo") == repo and not _freshness.is_behind(installed, latest)
            )
        ):
            plan["skip_reason"] = "up_to_date"
            return plan
        assets = _release_assets(repo, latest, force_refresh = force_refresh)
        names = list(assets) if assets else None
        status["update_size_bytes"] = _update_size_bytes(assets, record, installed, latest)
    installer = _installer()
    request = record.get("accelerator_request")
    resolved = record.get("accelerator")
    explicit = request if request in installer.ACCELERATORS and request != "auto" else None
    if not _host_has_prebuilt(names, resolved if isinstance(resolved, str) else None, explicit):
        plan["skip_reason"] = "no_prebuilt"
        return plan
    script = _installer_script()
    if script is None:
        plan["skip_reason"] = "installer_missing"
        return plan
    plan["update_available"] = True
    plan["phase"] = {
        "install_dir": audio_cpp_server.managed_audio_cpp_dir(),
        "script": script,
        # What setup asked for, so an automatic install keeps detecting the host.
        "accelerator": request if request in installer.ACCELERATORS else None,
        "from_tag": installed,
        "to_tag": status["latest_tag"],
        # Installs exactly the offered release, even if a newer one appears before the phase runs.
        "pin_release_tag": status["latest_tag"],
        "repo": repo,
    }
    return plan


def chained_phase_plan(*, force_refresh: bool = False) -> dict:
    """audio.cpp's side of the combined update item. Returns {status, update_available, skip_reason, phase} like whisper_cpp_update.chained_phase_plan; force_refresh bypasses the 24h release cache. Never raises: a probe that fails is a skip, so audio.cpp can never break the llama status or apply."""
    try:
        return _plan(force_refresh = force_refresh)
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.debug("audio.cpp update: plan failed", error = str(exc))
        return {
            "status": None,
            "update_available": False,
            "skip_reason": "unavailable",
            "phase": None,
        }


def _unload_audio_cpp_models() -> bool:
    """Stop every audio.cpp model in the main and kept slots, a load in flight included. Returns whether one was unloaded."""
    from core.inference import model_slots

    unloaded = False
    try:
        from routes.inference import _peek_inference_backend
        main = _peek_inference_backend()
    except Exception as exc:  # noqa: BLE001 - no orchestrator means nothing to unload
        logger.debug("audio.cpp update: inference backend unavailable", error = str(exc))
        main = None
    if main is not None:
        for name in model_slots.audio_cpp_model_names(main):
            unloaded = True
            # A failed unload can leave the worker's server running from the tree being replaced.
            if not main.unload_model(name):
                raise RuntimeError(f"{name} did not unload")
    # A kept server that survived would run from, or lock, the tree being replaced.
    if model_slots.unload_audio_cpp_slots(strict = True):
        unloaded = True
    return unloaded


def _failure_message(exc: _flow.InstallerExit, env: dict) -> str:
    if exc.returncode == _EXIT_BUSY:
        return "The audio runtime is in use by another Unsloth Studio or setup run. Close it and try again."
    text = str(exc)
    if _flow.is_github_rate_limit_text(text):
        advice = _flow.github_rate_limit_advice(_flow.github_token_present(env))
        return f"Could not update audio.cpp: GitHub is rate-limiting release downloads. {advice}"
    # The installer's ReleaseLookupUnavailable: the offered release was unreachable and the tree was never touched.
    if "no audio.cpp release lookup answered" in text:
        return "Could not reach the audio.cpp release on GitHub, so the existing install was kept. Try again later."
    # The installer's own verdict is its last "error:" line; the rest of the tail is download progress.
    errors = [line.strip() for line in text.splitlines() if line.strip().startswith("error: ")]
    return errors[-1][len("error: ") :] if errors else text


def _refresh_release_cache(repo: Optional[str]) -> None:
    """Forget the cached newest release after an install, the tree kept or replaced, so the next status reads it fresh. The disk cache goes too: offline, the old value would otherwise come back as the last good one."""
    _freshness.reset_caches(drop_disk = True)
    if not repo:
        return
    try:
        _freshness.latest_published_release(repo, force_refresh = True)
    except Exception as exc:  # noqa: BLE001 - the install already finished
        logger.debug("audio.cpp update: release refresh failed", error = str(exc))


def _installed_identity() -> tuple[Optional[str], Optional[str]]:
    """``(published_repo, release_tag)`` of the managed runtime's install record."""
    from core.inference import audio_cpp_server

    binary = audio_cpp_server.find_audio_cpp_server_binary()
    record = audio_cpp_server.read_install_record(binary) if binary else {}
    repo = record.get("published_repo")
    return (repo if isinstance(repo, str) else None), _tag(record)


def _install(phase: dict, set_progress, unloaded: bool) -> dict:
    before = _installed_identity()
    reload_hint = " Reload your model to use it." if unloaded else ""
    # The offered tag is passed as a pin, which the installer never second-guesses. A setup run that installed a
    # newer release since the plan was made must not be downgraded by it; a user's UNSLOTH_AUDIO_CPP_TAG still is.
    to_tag = phase.get("to_tag")
    if (
        before[0] == phase.get("repo")
        and before[1]
        and to_tag
        and _target()[1] is None
        and _freshness.is_behind(to_tag, before[1])
    ):
        logger.info("audio.cpp update: already newer", installed = before[1], offered = to_tag)
        return {
            "to_tag": before[1],
            "reload_required": unloaded,
            "message": f"audio.cpp is already at {before[1]}.{reload_hint}",
        }
    cmd = [sys.executable, str(phase["script"]), "--install-dir", str(phase["install_dir"])]
    if phase.get("accelerator"):
        cmd.extend(["--accelerator", phase["accelerator"]])
    if phase.get("pin_release_tag"):
        cmd.extend(["--published-release-tag", phase["pin_release_tag"]])
        if _target()[1] is None:
            # The installer re-checks under its lock, which the guard above cannot hold.
            cmd.append("--keep-newer")
    logger.info("audio.cpp update: installing", cmd = " ".join(cmd))
    env = dict(os.environ, UNSLOTH_PROGRESS_PERCENT_STEP = "5")
    try:
        _flow.stream_installer(
            cmd,
            env,
            set_progress = set_progress,
            timeout_seconds = _INSTALL_TIMEOUT_SECONDS,
        )
    except _flow.InstallerExit as exc:
        logger.warning("audio.cpp update: installer failed", error = str(exc)[-500:])
        raise _AudioPhaseError(_failure_message(exc, env), reload_required = unloaded) from exc
    except Exception as exc:
        raise _AudioPhaseError(str(exc), reload_required = unloaded) from exc
    _refresh_release_cache(phase.get("repo"))

    new_repo, new_tag = _installed_identity()
    # Repo and tag together: switching UNSLOTH_AUDIO_CPP_REPO can replace a runtime with another repo's same tag.
    # A runtime that is already the offered release (a setup run got there first) is not a kept old tree.
    if before[1] is None and new_tag is None:
        # A first install that exited 0 without a runtime (the release lookup could not answer).
        message = "The audio runtime could not be installed right now. Try again later."
        raise _AudioPhaseError(message, reload_required = unloaded)
    elif (new_repo, new_tag) == before and (new_repo, new_tag) != (
        phase.get("repo"),
        phase.get("to_tag"),
    ):
        # Exit 0 also means the release lookup could not answer and the intact tree was kept.
        message = (
            f"audio.cpp could not be updated right now, so the existing {new_tag} "
            f"install was kept. Try again later.{reload_hint}"
        )
    elif before[1] is None:
        message = f"Installed the audio runtime {new_tag}.{reload_hint}"
    else:
        message = f"Updated audio.cpp to {new_tag}.{reload_hint}"
    logger.info("audio.cpp update: done", to_tag = new_tag)
    return {"to_tag": new_tag, "reload_required": unloaded, "message": message}


def run_chained_phase(phase: dict, set_progress) -> dict:
    """Replace the managed audio.cpp tree while no audio.cpp server can run from it."""
    from core.inference import audio_cpp_server

    try:
        from core.inference.stt_audiocpp_sidecar import get_audio_cpp_stt_sidecar
        sidecar = get_audio_cpp_stt_sidecar()
    except Exception as exc:
        # Replacing the tree without the dictation guard would race a transcription starting from it. Fail closed.
        raise RuntimeError("could not coordinate the audio.cpp dictation sidecar update") from exc

    # Set first: a load that starts while the sidecar drains a transcription is refused, not unloaded later.
    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        with sidecar.update_maintenance():
            try:
                unloaded = _unload_audio_cpp_models()
            except Exception as exc:
                # Some model may already be gone, so the UI must resync either way.
                raise _AudioPhaseError(
                    f"Could not stop the audio models before the update: {exc}",
                    reload_required = True,
                ) from exc
            return _install(phase, set_progress, unloaded)
    finally:
        try:
            # Resolutions follow the runtime's model_specs/, so none made against the old tree may outlive it.
            from core.inference import audio_cpp_models
            audio_cpp_models.forget()
        finally:
            audio_cpp_server.UPDATE_IN_PROGRESS.clear()


def _spec_commit(installer, record: dict, repo: str, tag: str) -> Optional[str]:
    """The source commit the install was built from: the record, the release manifest, else the server version string."""
    commit = record.get("source_commit")
    if isinstance(commit, str) and _COMMIT.fullmatch(commit):
        return commit
    try:
        commit = (installer._release_index(repo, tag) or {}).get("source_commit")
    except Exception as exc:  # noqa: BLE001 - offline or rate-limited; try the version string
        logger.debug("audio.cpp spec backfill: release lookup failed", error = str(exc))
        commit = None
    if isinstance(commit, str) and _COMMIT.fullmatch(commit):
        return commit
    version = record.get("server_version")
    match = _VERSION_COMMIT.search(version) if isinstance(version, str) else None
    return match.group(1) if match else None


def _backfill_model_specs() -> bool:
    """Fetch ``model_specs/`` for a managed install made before the installer fetched it. Returns
    whether the folder was added; never raises."""
    from core.inference import audio_cpp_server
    try:
        binary = audio_cpp_server.find_audio_cpp_server_binary()
        if not binary or not _is_managed(binary) or audio_cpp_server.UPDATE_IN_PROGRESS.is_set():
            return False
        bin_dir = Path(binary).parent
        record = audio_cpp_server.read_install_record(binary)
        tag = _tag(record)
        repo = record.get("published_repo")
        if tag is None or not isinstance(repo, str) or (bin_dir / "model_specs").exists():
            return False
        installer = _installer()
        commit = _spec_commit(installer, record, repo, tag)
        if commit is None:
            logger.info("audio.cpp spec backfill: no source commit", tag = tag)
            return False
        core = installer.core
        managed_dir = audio_cpp_server.managed_audio_cpp_dir().resolve()
        with core.install_lock(core.install_lock_path(managed_dir), timeout = 2):
            # An update may have replaced the tree while the lock was taken.
            if audio_cpp_server.UPDATE_IN_PROGRESS.is_set() or (bin_dir / "model_specs").exists():
                return False
            if audio_cpp_server.read_install_record(binary).get("release_tag") != tag:
                return False
            for leftover in bin_dir.glob(".model_specs-*"):
                # Left by a killed fetch; only the exact name model_specs is ever read.
                shutil.rmtree(leftover, ignore_errors = True)
            if not installer._fetch_model_specs(repo, commit, bin_dir):
                logger.info("audio.cpp spec backfill: fetch failed", tag = tag)
                return False
        from core.inference import audio_cpp_models

        audio_cpp_models.forget()
        logger.info("audio.cpp spec backfill: added model_specs", tag = tag, commit = commit)
        return True
    except Exception as exc:  # noqa: BLE001 - families stay unknown, as before
        logger.debug("audio.cpp spec backfill failed", error = str(exc))
        return False


def start_model_specs_backfill() -> None:
    """Run the spec backfill once per process on a daemon thread, unless update checks are disabled."""
    global _backfill_started
    if update_checks_disabled():
        return
    with _backfill_lock:
        if _backfill_started:
            return
        _backfill_started = True
    threading.Thread(
        target = _backfill_model_specs, daemon = True, name = "audio-cpp-spec-backfill"
    ).start()
