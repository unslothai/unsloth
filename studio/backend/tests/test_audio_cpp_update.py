# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp phase of the combined in-app update: what it offers, and how it swaps the runtime."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_server, model_slots
from core.inference.audio_cpp_models import AUDIO_CPP_MUSIC_AUDIO_TYPE, AUDIO_CPP_TTS_AUDIO_TYPE
from utils import audio_cpp_update as aupd
from utils.prebuilt import update_flow

# A Studio pinned one release past the installed one, so the tests do not depend on the real pin.
_LADDER = [("unslothai/audio.cpp", "v0.9.1-unsloth.1"), ("0xShug0/audio.cpp", "v0.9.1")]
_real_unload = aupd._unload_audio_cpp_models
_SETUP_SKIPS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH", "UNSLOTH_SKIP_AUDIO_CPP_INSTALL")


def _write_record(tree, **fields):
    record = {
        "published_repo": "unslothai/audio.cpp",
        "release_tag": "v0.9.0-unsloth.1",
        "accelerator": "cpu",
        "accelerator_request": "cpu",
        **fields,
    }
    (tree / audio_cpp_server.INSTALL_RECORD).write_text(json.dumps(record))


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """A Studio-managed install of v0.9.0-unsloth.1 under a Studio that pins v0.9.1-unsloth.1."""
    for name in (*_SETUP_SKIPS, "UNSLOTH_DISABLE_UPDATE_CHECK"):
        monkeypatch.delenv(name, raising = False)
    tree = tmp_path / "audio.cpp"
    tree.mkdir()
    (tree / ".unsloth-studio-owned").touch()
    binary = tree / "audiocpp_server"
    binary.write_text("stub")
    _write_record(tree)
    script = tmp_path / "install_audio_cpp_prebuilt.py"
    script.write_text("stub")
    monkeypatch.setattr(audio_cpp_server, "managed_audio_cpp_dir", lambda: tree)
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(binary))
    monkeypatch.setattr(aupd, "_release_ladder", lambda: list(_LADDER))
    monkeypatch.setattr(aupd, "_installer_script", lambda: script)
    return SimpleNamespace(tree = tree, binary = binary, script = script)


# --- plan ---


def test_an_old_managed_runtime_is_offered_the_pinned_release(managed):
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True and plan["skip_reason"] is None
    assert plan["status"] == {
        "installed_tag": "v0.9.0-unsloth.1",
        "latest_tag": "v0.9.1-unsloth.1",
        "update_size_bytes": None,
    }
    assert plan["phase"] == {
        "install_dir": managed.tree,
        "script": managed.script,
        "accelerator": "cpu",
        "from_tag": "v0.9.0-unsloth.1",
        "to_tag": "v0.9.1-unsloth.1",
    }


def test_an_automatic_install_keeps_detecting_the_host(managed):
    _write_record(managed.tree, accelerator = "cuda", accelerator_request = "auto")
    assert aupd.chained_phase_plan()["phase"]["accelerator"] == "auto"


@pytest.mark.parametrize("name", _SETUP_SKIPS)
def test_a_runtime_setup_skips_is_unmanaged(managed, monkeypatch, name):
    monkeypatch.setenv(name, "1" if name == "UNSLOTH_SKIP_AUDIO_CPP_INSTALL" else "/elsewhere")
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "unmanaged"


def test_a_tree_without_the_ownership_marker_is_unmanaged(managed):
    (managed.tree / ".unsloth-studio-owned").unlink()
    assert aupd.chained_phase_plan()["skip_reason"] == "unmanaged"


def test_a_binary_outside_the_managed_tree_is_unmanaged(managed, monkeypatch, tmp_path):
    custom = tmp_path / "custom" / "audiocpp_server"
    custom.parent.mkdir()
    custom.write_text("stub")
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(custom))
    assert aupd.chained_phase_plan()["skip_reason"] == "unmanaged"


def test_no_runtime_is_not_installed(managed, monkeypatch):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: None)
    plan = aupd.chained_phase_plan()
    assert plan["skip_reason"] == "not_installed" and plan["status"] is None


@pytest.mark.parametrize("repo, tag", _LADDER)
def test_either_rung_is_up_to_date(managed, repo, tag):
    _write_record(managed.tree, published_repo = repo, release_tag = tag)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "up_to_date"


def test_a_latest_tracking_ladder_cannot_compare(managed, monkeypatch):
    monkeypatch.setattr(aupd, "_release_ladder", lambda: [("unslothai/audio.cpp", None)])
    assert aupd.chained_phase_plan()["skip_reason"] == "tracks_latest"


def test_disabled_update_checks_skip_the_offer(managed, monkeypatch):
    # Without this skip the union would show a card that start_update refuses.
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    assert aupd.chained_phase_plan()["skip_reason"] == "update_checks_disabled"


def test_a_host_no_pinned_bundle_covers_has_no_prebuilt(managed, monkeypatch):
    installer = aupd._installer()
    pins = {
        repo: {tag: {f"audio-{tag}-bin-windows-x64-cpu.zip": "0" * 64}} for repo, tag in _LADDER
    }
    monkeypatch.setattr(installer, "load_pins", lambda path = None: pins)
    monkeypatch.setattr(aupd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "x86_64")
    assert aupd.chained_phase_plan()["skip_reason"] == "no_prebuilt"
    # The same pins cover a Windows host.
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    assert aupd.chained_phase_plan()["update_available"] is True


@pytest.mark.parametrize("request_, offered", [("vulkan", False), ("auto", True)])
def test_an_explicit_accelerator_is_not_offered_the_cpu_bundle(
    managed, monkeypatch, request_, offered
):
    # The installer never downgrades an explicit request, so a CPU-only release cannot satisfy it.
    installer = aupd._installer()
    pins = {
        repo: {tag: {f"audio-{tag}-bin-ubuntu-arm64-cpu.tar.gz": "0" * 64}} for repo, tag in _LADDER
    }
    monkeypatch.setattr(installer, "load_pins", lambda path = None: pins)
    monkeypatch.setattr(aupd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "aarch64")
    _write_record(managed.tree, accelerator = "vulkan", accelerator_request = request_)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is offered
    assert plan["skip_reason"] == (None if offered else "no_prebuilt")


def test_a_missing_installer_skips_the_offer(managed, monkeypatch):
    monkeypatch.setattr(aupd, "_installer_script", lambda: None)
    assert aupd.chained_phase_plan()["skip_reason"] == "installer_missing"


def test_a_probe_that_raises_is_a_skip(managed, monkeypatch):
    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr(aupd, "_release_ladder", boom)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "unavailable"


def test_the_pinned_ladder_comes_from_the_installer(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_TAG", raising = False)
    installer = aupd._installer()
    assert aupd._release_ladder()[0] == (installer.DEFAULT_REPO, installer.DEFAULT_TAG)


# --- run ---


class _Sidecar:
    def __init__(self, events):
        self.events = events

    @contextmanager
    def update_maintenance(self):
        self.events.append("sidecar:enter")
        try:
            yield False
        finally:
            self.events.append("sidecar:exit")


@pytest.fixture
def run_env(managed, monkeypatch):
    """The run path with the sidecar, slot unload and installer recorded."""
    from core.inference import stt_audiocpp_sidecar

    events: list = []
    calls: dict = {}
    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", lambda: _Sidecar(events))
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: events.append("unload") or False)

    def _installer(new_tag = "v0.9.1-unsloth.1", error = None):
        def _stream(cmd, env, *, set_progress, timeout_seconds):
            events.append("install")
            calls.update(cmd = cmd, env = env, flag = audio_cpp_server.UPDATE_IN_PROGRESS.is_set())
            if error is not None:
                raise error
            set_progress(0.5)
            _write_record(managed.tree, release_tag = new_tag)

        monkeypatch.setattr(update_flow, "stream_installer", _stream)

    _installer()
    phase = aupd.chained_phase_plan()["phase"]
    return SimpleNamespace(events = events, calls = calls, phase = phase, installer = _installer)


def test_the_runtime_is_swapped_while_no_audio_cpp_server_can_start(run_env):
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert run_env.events == ["sidecar:enter", "unload", "install", "sidecar:exit"]
    assert run_env.calls["flag"] is True
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()
    assert result == {
        "to_tag": "v0.9.1-unsloth.1",
        "reload_required": False,
        "message": "Updated audio.cpp to v0.9.1-unsloth.1.",
    }


def test_the_installer_gets_the_managed_dir_and_recorded_accelerator(run_env, managed):
    aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    cmd = run_env.calls["cmd"]
    assert cmd[1:] == [
        str(managed.script),
        "--install-dir",
        str(managed.tree),
        "--accelerator",
        "cpu",
    ]
    assert run_env.calls["env"]["UNSLOTH_PROGRESS_PERCENT_STEP"] == "5"


def test_an_unloaded_model_asks_for_a_reload(run_env, monkeypatch):
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: True)
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["reload_required"] is True
    assert (
        result["message"] == "Updated audio.cpp to v0.9.1-unsloth.1. Reload your model to use it."
    )


def test_the_update_flag_clears_when_the_install_fails(run_env):
    run_env.installer(error = RuntimeError("installer timed out after 1800s"))
    with pytest.raises(RuntimeError, match = "timed out"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()
    assert run_env.events[-1] == "sidecar:exit"


def test_new_loads_are_refused_while_the_runtime_is_replaced():
    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        problem = audio_cpp_server.model_runtime_problem(SimpleNamespace(unsupported = None))
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert problem == "The audio runtime is being updated. Try again in a moment."


def test_a_load_registering_after_the_unload_scan_is_refused():
    # Preflight can pass just before the flag is set; the registration is what the scan races.
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.loading_models = set()
    orchestrator._managed_engine = None
    config = SimpleNamespace(identifier = "kokoro", audio_cpp = object())
    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        with pytest.raises(RuntimeError, match = "being updated"):
            orchestrator.load_model(config)
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert orchestrator.loading_models == set()


def test_a_kept_tree_is_not_reported_as_updated(run_env):
    # Exit 0 also means the release lookup could not answer and the intact tree was kept.
    run_env.installer(new_tag = "v0.9.0-unsloth.1")
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["message"] == (
        "audio.cpp could not be updated right now, so the existing v0.9.0-unsloth.1 "
        "install was kept. Try again later."
    )


def test_a_kept_tree_still_asks_for_a_reload_after_an_unload(run_env, monkeypatch):
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: True)
    run_env.installer(new_tag = "v0.9.0-unsloth.1")
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["reload_required"] is True
    assert result["message"].endswith("Try again later. Reload your model to use it.")


def test_a_busy_install_says_the_runtime_is_in_use(run_env):
    run_env.installer(error = update_flow.InstallerExit(3, "installer exited 3: error: busy"))
    with pytest.raises(RuntimeError, match = "in use by another Unsloth Studio or setup run"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)


def test_a_checksum_failure_reports_the_installer_verdict(run_env):
    tail = (
        "installer exited 1: audio.cpp: downloading audio-x.tar.gz\n"
        " 50.0% (1 MiB/2 MiB)\n"
        "error: audio-x.tar.gz: sha256 mismatch (expected aa, got bb)\n"
    )
    run_env.installer(error = update_flow.InstallerExit(1, tail))
    with pytest.raises(RuntimeError) as info:
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert str(info.value) == "audio-x.tar.gz: sha256 mismatch (expected aa, got bb)"


def test_a_rate_limit_gets_token_advice(run_env, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    tail = "installer exited 1: error: GitHub API is rate limiting release lookups\n"
    run_env.installer(error = update_flow.InstallerExit(1, tail))
    with pytest.raises(RuntimeError, match = "Set GH_TOKEN or GITHUB_TOKEN"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)


def test_no_sidecar_coordination_fails_closed(run_env, monkeypatch):
    from core.inference import stt_audiocpp_sidecar

    def boom():
        raise ImportError("sidecar unavailable")

    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", boom)
    with pytest.raises(RuntimeError, match = "dictation sidecar"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert "install" not in run_env.events
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()


# --- unload ---


class _Orchestrator:
    def __init__(
        self,
        models = None,
        loading = (),
    ):
        self.models = dict(models or {})
        self.loading_models = set(loading)
        self.unloaded: list = []

    def unload_model(self, name):
        self.unloaded.append(name)
        self.models.pop(name, None)
        self.loading_models.discard(name)
        return True

    def cancel_load(self, name):
        self.loading_models.discard(name)
        return True


@pytest.fixture
def slots(monkeypatch):
    from routes import inference

    dropped: list = []
    state = SimpleNamespace(main = None, dropped = dropped)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: state.main)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "drop", lambda slot: dropped.append(slot))
    return state


def test_a_main_slot_audio_cpp_model_is_unloaded(slots):
    slots.main = _Orchestrator(
        {"kokoro": {"audio_type": AUDIO_CPP_TTS_AUDIO_TYPE, "is_audio": True}}
    )
    assert aupd._unload_audio_cpp_models() is True
    assert slots.main.unloaded == ["kokoro"]


def test_a_failed_unload_stops_the_update_before_the_install(run_env, slots, monkeypatch):
    # The run_env fixture stubs the unload; put the real one back to drive the orchestrator.
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", _real_unload)
    slots.main = _Orchestrator({"kokoro": {"audio_type": AUDIO_CPP_TTS_AUDIO_TYPE}})
    slots.main.unload_model = lambda name: False
    with pytest.raises(RuntimeError, match = "kokoro did not unload") as info:
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert info.value.reload_required is True
    assert "install" not in run_env.events


def test_a_main_slot_chat_model_is_left_alone(slots):
    slots.main = _Orchestrator({"qwen": {"audio_type": None}})
    assert aupd._unload_audio_cpp_models() is False
    assert slots.main.unloaded == []


def test_an_audio_cpp_load_in_flight_is_cancelled(slots, monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")
    slots.main = _Orchestrator(loading = {"kokoro"})
    assert aupd._unload_audio_cpp_models() is True
    assert slots.main.unloaded == ["kokoro"]


def test_a_load_that_publishes_during_the_scan_is_still_found(monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")

    class _Publishing:
        loading_models = {"kokoro"}

        @property
        def models(self):
            # The load finishes right after this read: it leaves loading_models for models.
            self.loading_models = set()
            return {}

    assert model_slots.audio_cpp_model_names(_Publishing()) == ["kokoro"]


def test_a_kept_slot_load_is_cancelled_before_its_slot_is_dropped(slots, monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")
    chat = SimpleNamespace(orchestrator = _Orchestrator({"qwen": {"audio_type": None}}))
    loading = SimpleNamespace(orchestrator = _Orchestrator(loading = {"kokoro"}))
    model_slots.slots[:] = [chat, loading]
    assert aupd._unload_audio_cpp_models() is True
    # A marker left behind would let the load spawn its worker after the drop.
    assert loading.orchestrator.loading_models == set()
    assert slots.dropped == [loading]


def test_a_kept_audio_cpp_slot_is_dropped(slots):
    music = SimpleNamespace(
        orchestrator = _Orchestrator({"ace": {"audio_type": AUDIO_CPP_MUSIC_AUDIO_TYPE}})
    )
    chat = SimpleNamespace(orchestrator = _Orchestrator({"qwen": {"audio_type": None}}))
    model_slots.slots[:] = [music, chat]
    assert aupd._unload_audio_cpp_models() is True
    assert slots.dropped == [music]
