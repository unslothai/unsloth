# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An audio model kept beside the chat model runs its workflow, as far as its generate call."""

import asyncio
from dataclasses import replace

import pytest

import core.inference.model_slots as model_slots
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from core.inference.load_policy import LoadPolicy
from models.inference import AudioRunRequest, ChatCompletionRequest
from studio.backend.tests.test_audio_run_route import _music_info, _speak_info
from studio.backend.tests.test_multi_model_serving import FakeLlama, FakeOrchestrator
from utils import multi_model_settings, openai_auto_switch_settings
from utils.account_context import OWNER, run_as

ACE_STEP = "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF"
_INFO = {ACE_STEP: _music_info, "org/Music": _music_info, "org/Speech": _speak_info}
_OWNER_CHAT_TOOLS_KEY = replace(OWNER, via_api_key = True, load_policy = LoadPolicy.BESIDE_ONLY)


class AudioOrchestrator(FakeOrchestrator):
    """Loads ``model`` (music unless told); its generate call records itself and stops the run."""

    generated: list = []

    def load(self, model = ACE_STEP):
        self.active_model_name, self.models = model, {model: _INFO[model]()}

    def generate_audio_response(self, **_kwargs):
        AudioOrchestrator.generated.append(self)
        raise RuntimeError("stop")


async def _load(*_a, **_kw):
    inf.get_inference_backend().load()


def _run(workflow, model = None):
    if workflow == "music":
        body = AudioRunRequest(workflow = "music", mode = "song", text = "lofi", model = model)
        run = inf._run_music_workflow(body, None, "s")
    else:
        payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hello"}])
        run = inf._generate_tts_wav("hello", payload, None, "s")
    with pytest.raises(Exception):  # the stub stops every run that gets this far
        run_as(_OWNER_CHAT_TOOLS_KEY, asyncio.run, run)


def test_audio_loads_beside_chat_then_an_omitted_model_runs_on_the_one_kept_for_it(monkeypatch):
    kept = [model_slots.ExtraSlot(FakeLlama(), AudioOrchestrator(), "owner") for _ in range(2)]
    for slot, model in zip(kept, ["org/Music", "org/Speech"]):
        slot.orchestrator.load(model)
    for mod, name, value in [
        (inf, "_llama_cpp_backend", FakeLlama("unsloth/Qwen-GGUF", "Q4_K_M")),
        (orchestrator, "_inference_backend", FakeOrchestrator()),
        (model_slots, "slots", kept),
        (AudioOrchestrator, "generated", []),
        (inf, "InferenceOrchestrator", AudioOrchestrator),
        (inf, "_run_tracked_load_model_impl", _load),
        (inf, "_target_speech_audio_type", lambda *_a: "audiocpp_music"),
        (inf, "_target_audio_workflows", lambda *_a: {"music"}),
        (inf, "_audio_cpp_switch_target", {ACE_STEP: (ACE_STEP, "Q8_0", ACE_STEP, False)}.get),
        (multi_model_settings, "get_multi_model_enabled", lambda: True),
        (multi_model_settings, "get_multi_model_mode", lambda: "always"),
        (openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False),
    ]:
        monkeypatch.setattr(mod, name, value)
    # Loads beside; its TTS step, which names no model, stays on it ("No model loaded." before).
    _run("music", ACE_STEP)
    _run("music")  # two music models kept: the caller names one
    _run("speak")
    assert AudioOrchestrator.generated == [model_slots.slots[2].orchestrator, kept[1].orchestrator]
