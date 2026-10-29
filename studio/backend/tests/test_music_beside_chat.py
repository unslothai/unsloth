# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A music model kept beside the chat model runs the music workflow, as far as its generate call."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import core.inference.model_slots as model_slots
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from core.inference import audio_inputs
from core.inference.load_policy import LoadPolicy
from models.inference import AudioRunRequest
from utils import multi_model_settings
from utils import openai_auto_switch_settings as switch_settings
from utils.account_context import OWNER, bind_account, reset_account

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_run_route import _music_info  # noqa: E402
from test_multi_model_serving import FakeLlama, FakeOrchestrator  # noqa: E402

ACE_STEP = "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF"


class _Generated(Exception):
    pass


class MusicOrchestrator(FakeOrchestrator):
    """Empty until loaded; its generate call records itself and stops the run."""

    generated: list = []

    def load(self):
        self.active_model_name = ACE_STEP
        self.models = {ACE_STEP: _music_info("ace_step", "ACE-Step1.5-GGUF")}

    def generate_audio_response(self, **_kwargs):
        MusicOrchestrator.generated.append(self)
        raise _Generated()


class _Request:
    scope: dict = {}
    url = type("Url", (), {"path": "/v1/audio/run"})()
    state = type("State", (), {})()

    async def is_disconnected(self):
        return False


@pytest.fixture
def chat_model(monkeypatch, tmp_path):
    """A Qwen GGUF serving chat; Always; the owner's chat tools key; switch off."""
    primary = FakeLlama("unsloth/Qwen-GGUF", "Q4_K_M")
    primary._is_audio = False
    monkeypatch.setattr(inf, "_llama_cpp_backend", primary)
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", MusicOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    monkeypatch.setattr(multi_model_settings, "get_multi_model_enabled", lambda: True)
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "always")
    monkeypatch.setattr(switch_settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(inf, "_auto_switch_waiters", {})
    monkeypatch.setattr(inf, "_preflight_speech_codec_for_switch", lambda *_a: None)
    monkeypatch.setattr(
        inf,
        "_audio_cpp_switch_target",
        lambda m: (ACE_STEP, "Q8_0", ACE_STEP, False) if m == ACE_STEP else None,
    )
    monkeypatch.setattr(inf, "_target_speech_audio_type", lambda *_a: "audiocpp_music")
    monkeypatch.setattr(inf, "_target_audio_workflows", lambda *_a: {"music"})
    monkeypatch.setattr(audio_inputs, "inputs_dir", lambda: tmp_path)
    monkeypatch.setattr(MusicOrchestrator, "generated", [])
    loads = []

    async def load(request, *_a, **_kw):
        loads.append(request.alongside)
        inf.get_inference_backend().load()
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    token = bind_account(replace(OWNER, via_api_key = True, load_policy = LoadPolicy.BESIDE_ONLY))
    yield loads
    reset_account(token)


def _keep_music_model():
    music = model_slots.ExtraSlot(FakeLlama(), MusicOrchestrator(), "owner")
    music.orchestrator.load()
    model_slots.slots.append(music)
    return music


def _run_music(model):
    body = AudioRunRequest(workflow = "music", mode = "song", text = "lofi", model = model)

    async def run():
        try:
            await inf._run_music_workflow(body, _Request(), "s")
        except Exception as exc:  # noqa: BLE001 - the stub stops every run that gets this far
            return exc
        return None

    return asyncio.run(run())


def test_a_named_music_model_kept_beside_chat_generates(chat_model):
    music = _keep_music_model()
    _run_music(ACE_STEP)
    assert MusicOrchestrator.generated == [music.orchestrator]


def test_a_music_model_loaded_beside_chat_on_this_call_generates(chat_model):
    _run_music(ACE_STEP)
    assert chat_model == [True] and len(model_slots.slots) == 1
    assert MusicOrchestrator.generated == [model_slots.slots[0].orchestrator]
    # The next call finds it kept.
    _run_music(ACE_STEP)
    assert MusicOrchestrator.generated == [model_slots.slots[0].orchestrator] * 2


def test_an_omitted_model_generates_on_the_one_kept_music_model(chat_model):
    music = _keep_music_model()
    _run_music(None)
    assert MusicOrchestrator.generated == [music.orchestrator]
