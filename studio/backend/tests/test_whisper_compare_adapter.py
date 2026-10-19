# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

try:
    import core.inference.inference
except ImportError:
    pass

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import routes.inference as inference_route
from core.inference import worker
from core.inference.orchestrator import InferenceOrchestrator
from models.inference import ChatCompletionRequest


class _ChatRequest:
    method = "POST"
    url = SimpleNamespace(path = "/v1/chat/completions")
    state = SimpleNamespace(skip_api_monitor = True)
    scope: dict = {}

    async def is_disconnected(self):
        return False


def test_the_chat_route_passes_each_compare_panes_adapter_state_to_whisper(monkeypatch):
    seen = []

    class _WhisperBackend:
        active_model_name = "org/whisper-lora"
        models = {"org/whisper-lora": {"has_audio_input": True, "audio_type": "whisper"}}

        def generate_whisper_response(
            self,
            use_adapter = None,
            **_kwargs,
        ):
            seen.append(use_adapter)
            yield "hello"

        def reset_generation_state(self, caller_cancel_event = None):
            pass

    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = False, supports_tools = False, is_vision = False, context_length = None
        ),
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _WhisperBackend())
    monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(
        inference_route, "_decode_audio_clips", lambda _clips: [np.zeros(16, np.float32)]
    )

    async def _no_auto_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_auto_switch)

    for use_adapter in (False, True):
        payload = ChatCompletionRequest(
            model = "org/whisper-lora",
            messages = [{"role": "user", "content": ""}],
            audio_base64 = "ZmFrZQ==",
            use_adapter = use_adapter,
        )
        response = asyncio.run(
            inference_route.openai_chat_completions(
                payload, _ChatRequest(), current_subject = "tester"
            )
        )
        assert response.status_code == 200

    assert seen == [False, True]


def test_the_orchestrator_sends_the_adapter_state_with_a_whisper_request(monkeypatch):
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = {}

    def _inner(**kwargs):
        sent.update(kwargs)
        yield from ()

    monkeypatch.setattr(orchestrator, "_generate_audio_input_inner", _inner)

    list(orchestrator.generate_whisper_response(audio_array = [0.0], use_adapter = False))

    assert sent["audio_type"] == "whisper"
    assert sent["use_adapter"] is False


def test_the_worker_hands_the_adapter_state_to_whisper():
    calls = []

    class _Backend:
        def generate_whisper_response(self, **kwargs):
            calls.append(kwargs)
            return iter(())

    worker._handle_generate_audio_input(
        _Backend(),
        {
            "request_id": "r",
            "audio_clips": [np.zeros(16, np.float32).tobytes()],
            "audio_type": "whisper",
            "use_adapter": False,
        },
        SimpleNamespace(put = lambda *_args, **_kwargs: None),
        SimpleNamespace(is_set = lambda: False),
    )

    assert calls[0]["use_adapter"] is False


def test_whisper_applies_the_adapter_state_under_the_lock_before_transcribing(monkeypatch):
    pytest.importorskip("peft")
    from core.inference.inference import InferenceBackend

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "whisper-lora"
    backend._generation_lock = threading.Lock()
    events = []

    def _pipe(_inputs):
        events.append(("transcribe", backend._generation_lock.locked()))
        return {"text": "hello"}

    backend.models = {"whisper-lora": {"model": object(), "whisper_pipeline": _pipe}}
    monkeypatch.setattr(
        backend,
        "_apply_adapter_state",
        lambda state: events.append((state, backend._generation_lock.locked())),
    )
    audio = np.zeros(16, np.float32)

    for use_adapter in (False, True):
        assert list(backend.generate_whisper_response(audio, use_adapter = use_adapter)) == ["hello"]

    assert events == [(False, True), ("transcribe", True), (True, True), ("transcribe", True)]


def test_a_base_pane_transcription_turns_the_lora_back_on_afterwards():
    peft = pytest.importorskip("peft")
    import torch
    from core.inference.inference import InferenceBackend

    class _Lora:
        disabled = False

        def disable_adapter_layers(self):
            self.disabled = True

        def enable_adapter_layers(self):
            self.disabled = False

    class _PeftWhisper(peft.PeftModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.base_model = _Lora()

    model = _PeftWhisper()
    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "whisper-lora"
    backend._generation_lock = threading.Lock()
    seen = []

    def _pipe(_inputs):
        seen.append(model.base_model.disabled)
        return {"text": "hello"}

    backend.models = {"whisper-lora": {"model": model, "whisper_pipeline": _pipe}}
    audio = np.zeros(16, np.float32)

    list(backend.generate_whisper_response(audio, use_adapter = False))
    list(backend.generate_whisper_response(audio))

    assert seen == [True, False]
