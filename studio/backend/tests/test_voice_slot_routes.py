# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice residency authorization and concurrent route teardown, without model weights."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.inference as routes


@pytest.fixture
def voice(monkeypatch):
    events = []
    backend = SimpleNamespace(is_active = False, is_loaded = False, model_identifier = None)

    def unload():
        events.append("unload")
        backend.is_active = backend.is_loaded = False
        backend.model_identifier = None

    backend.unload_model = unload
    monkeypatch.setattr(routes, "get_voice_llama_backend", lambda: backend)
    monkeypatch.setattr(routes, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(routes.account_access, "require_live_account", lambda: None)
    monkeypatch.setattr(routes.account_access, "require_resident_control", lambda *a: None)
    monkeypatch.setattr(routes.account_access, "publish_resident", lambda *a: events.append(a))
    monkeypatch.setattr(routes.account_access, "clear_resident", lambda *a: events.append("clear"))
    return backend, events


def test_voice_unload_waits_for_load_and_clears_its_residency(monkeypatch, voice):
    backend, events = voice

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def load(*args):
            backend.is_active = True
            started.set()
            await finish.wait()
            backend.is_loaded = True
            backend.model_identifier = "org/voice"
            events.append("loaded")
            return {"status": "loaded"}

        monkeypatch.setattr(routes, "_voice_load_model_impl", load)
        loading = asyncio.create_task(
            routes.voice_load_model(routes._VoiceLoadRequest(model_path = "org/voice"), "owner")
        )
        await started.wait()
        unloading = asyncio.create_task(routes.voice_unload_model("owner"))
        await asyncio.sleep(0.04)
        assert not unloading.done()
        finish.set()
        await loading
        await unloading

    asyncio.run(run())
    assert events == ["loaded", ("voice", "org/voice"), "unload", "clear"]
    assert not backend.is_active


def test_cancelled_voice_load_drains_worker_before_freeing_the_slot(monkeypatch, voice):
    backend, events = voice

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def load(*args):
            backend.is_active = True
            started.set()
            await finish.wait()
            backend.is_loaded = True
            events.append("worker_finished")
            return {"status": "loaded"}

        monkeypatch.setattr(routes, "_voice_load_model_impl", load)
        loading = asyncio.create_task(
            routes.voice_load_model(routes._VoiceLoadRequest(model_path = "org/voice"), "owner")
        )
        await started.wait()
        loading.cancel()
        await asyncio.sleep(0.02)
        assert not loading.done()
        assert events == []
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await loading

    asyncio.run(run())
    assert events == ["worker_finished", "unload", "clear"]
    assert not backend.is_active


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_foreign_resident_cannot_be_replaced_or_unloaded(monkeypatch, voice, operation):
    backend, events = voice
    backend.is_active = backend.is_loaded = True
    backend.model_identifier = "private/voice"

    def deny(modality, reference):
        assert (modality, reference) == ("voice", "private/voice")
        raise HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(routes.account_access, "require_resident_control", deny)
    request = routes._VoiceLoadRequest(model_path = "public/other")
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            routes.voice_load_model(request, "other")
            if operation == "load"
            else routes.voice_unload_model("other")
        )
    assert error.value.status_code == 404
    assert events == []
    assert backend.is_loaded


def test_foreign_voice_status_hides_model_identity(monkeypatch, voice):
    backend, _ = voice
    backend.is_active = backend.is_loaded = True
    backend.model_identifier = "private/voice"
    backend._audio_type = "snac"
    monkeypatch.setattr(routes.account_access, "resident_hidden", lambda *a: True)
    result = asyncio.run(routes.voice_slot_status("other"))
    assert result == {"loaded": False, "loading": False, "model": None, "audio_type": None}


def test_stream_refuses_foreign_voice_before_opening_a_monitor_row(monkeypatch, voice):
    backend, events = voice
    backend.is_active = backend.is_loaded = True
    backend.model_identifier = "private/voice"
    backend._audio_type = "snac"

    def deny(*args):
        raise HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(routes.account_access, "require_resident_control", deny)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            routes.openai_audio_speech_stream(
                routes.AudioSpeechRequest(input = "hello"), None, "other"
            )
        )
    assert error.value.status_code == 404
    assert events == []
