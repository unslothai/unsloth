# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app audio.cpp runtime update: the third phase of the combined llama.cpp update item (utils.llama_cpp_update chains run_chained_phase after the whisper phase). Unlike llama.cpp and whisper.cpp, audio.cpp is pinned in source: the target is the release install_audio_cpp_prebuilt.py would install for this Studio (its release ladder and sha256 pins), never GitHub's latest, so the status needs no network call. Only the Studio-managed tree is offered, the same predicate the Audio page notice reads. The mechanics (the streamed installer run, the chained job) live in utils.prebuilt.update_flow; this module keeps the audio.cpp policy."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Optional

import structlog

from utils.prebuilt import update_flow as _flow
from utils.update_status import update_checks_disabled

logger = structlog.get_logger(__name__)

_INSTALL_TIMEOUT_SECONDS = 1800
# setup.sh / setup.ps1 leave the managed tree alone when any of these is set, so an update would not take effect.
_SETUP_SKIP_VARS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH")
_EXIT_BUSY = 3


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


def _release_ladder() -> list:
    return _installer()._release_ladder()


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


def release_status(binary: str, record: dict) -> dict:
    """``expected_tag`` and ``outdated`` for an installed runtime: the release this Studio would install, and whether the record names neither rung of the ladder. Raises when the ladder cannot be read; callers treat that as current."""
    status = {"expected_tag": None, "outdated": False}
    if _tag(record) is None or not _is_managed(binary):
        return status
    ladder = _release_ladder()
    # a None tag tracks the latest release, so an installed release cannot be compared.
    if ladder and all(tag for _, tag in ladder):
        status["expected_tag"] = ladder[0][1]
        status["outdated"] = (record.get("published_repo"), _tag(record)) not in ladder
    return status


def _host_has_prebuilt(ladder: list, accelerator: Optional[str]) -> bool:
    """Whether any pinned rung publishes a bundle for this host. Offline: the pins list every asset name. A rung without pins (a release the user picked) can only be answered by the online lookup, so it counts as available."""
    installer = _installer()
    pins = installer.load_pins()
    for repo, tag in ladder:
        names = list((pins.get(repo) or {}).get(tag) or {})
        if not names:
            return True
        for accel in dict.fromkeys((accelerator or "cpu", "cpu")):
            if installer.resolve_release_asset(
                names,
                system = platform.system(),
                machine = platform.machine(),
                accelerator = accel,
            ):
                return True
    return False


def _plan() -> dict:
    from core.inference import audio_cpp_server

    plan: dict = {"status": None, "update_available": False, "skip_reason": None, "phase": None}
    binary = audio_cpp_server.find_audio_cpp_server_binary()
    if binary is None:
        plan["skip_reason"] = "not_installed"
        return plan
    record = audio_cpp_server.read_install_record(binary)
    status = {"installed_tag": _tag(record), "latest_tag": None, "update_size_bytes": None}
    plan["status"] = status
    if not _is_managed(binary):
        plan["skip_reason"] = "unmanaged"
        return plan
    ladder = _release_ladder()
    if not ladder or not all(tag for _, tag in ladder):
        plan["skip_reason"] = "tracks_latest"
        return plan
    status["latest_tag"] = ladder[0][1]
    if (
        status["installed_tag"] is None
        or (
            record.get("published_repo"),
            status["installed_tag"],
        )
        in ladder
    ):
        plan["skip_reason"] = "up_to_date"
        return plan
    if update_checks_disabled():
        plan["skip_reason"] = "update_checks_disabled"
        return plan
    installer = _installer()
    request = record.get("accelerator_request")
    resolved = record.get("accelerator")
    if not _host_has_prebuilt(ladder, resolved if isinstance(resolved, str) else None):
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
        "from_tag": status["installed_tag"],
        "to_tag": status["latest_tag"],
    }
    return plan


def chained_phase_plan() -> dict:
    """audio.cpp's side of the combined update item. Returns {status, update_available, skip_reason, phase} like whisper_cpp_update.chained_phase_plan. Never raises: a probe that fails is a skip, so audio.cpp can never break the llama status or apply."""
    try:
        return _plan()
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
    # The installer's own verdict is its last "error:" line; the rest of the tail is download progress.
    errors = [line.strip() for line in text.splitlines() if line.strip().startswith("error: ")]
    return errors[-1][len("error: ") :] if errors else text


def _install(phase: dict, set_progress, unloaded: bool) -> dict:
    from core.inference import audio_cpp_server

    cmd = [sys.executable, str(phase["script"]), "--install-dir", str(phase["install_dir"])]
    if phase.get("accelerator"):
        cmd.extend(["--accelerator", phase["accelerator"]])
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

    binary = audio_cpp_server.find_audio_cpp_server_binary()
    new_tag = _tag(audio_cpp_server.read_install_record(binary)) if binary else None
    reload_hint = " Reload your model to use it." if unloaded else ""
    if new_tag == phase["from_tag"]:
        # Exit 0 also means the release lookup could not answer and the intact tree was kept.
        message = (
            f"audio.cpp could not be updated right now, so the existing {new_tag} "
            f"install was kept. Try again later.{reload_hint}"
        )
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
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
