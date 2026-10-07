# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/speech.

The TTS core (_generate_tts_wav) is faked, so these cover route wiring, validation,
gallery persistence and the raw-WAV response without torch, weights or a GPU."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import re

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery_module
from core.inference.api_monitor import api_monitor
import routes.inference as routes_module
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers
from core.inference.external_provider import ExternalProviderClient
from models.inference import AudioSpeechRequest
import core.inference.audio_gallery as gallery
import core.inference.external_provider as provider_module
import threading


async def _boom(text):
    raise HTTPException(status_code = 400, detail = "No model loaded.")


_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-payload"


def _make_client(monkeypatch, generate = None):
    calls = []

    async def _fake_generate(text, payload, request, current_subject, **kwargs):
        calls.append({"text": text, "payload": payload, **kwargs})
        if generate is not None:
            return await generate(text)
        return _WAV, 24000, "unsloth/orpheus-3b-0.1-ft", "snac"

    saved = []

    def _save(wav_bytes, meta):
        saved.append({"bytes": wav_bytes, "meta": meta})
        return {**meta, "id": "aud0", "url": "/api/inference/audio/gallery/aud0/file"}

    monkeypatch.setattr(routes_module, "_generate_tts_wav", _fake_generate)
    monkeypatch.setattr(gallery_module, "save", _save)

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), calls, saved


def test_returns_raw_wav_bytes(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hello sloth"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content == _WAV
    assert calls[0]["text"] == "hello sloth"


def test_persists_clip_to_gallery(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "persist me"})
    assert resp.status_code == 200
    assert len(saved) == 1
    meta = saved[0]["meta"]
    assert meta["prompt"] == "persist me"
    assert meta["model"] == "unsloth/orpheus-3b-0.1-ft"
    assert meta["audio_type"] == "snac"
    assert meta["sample_rate"] == 24000
    assert isinstance(meta["duration_s"], float)
    assert meta["created_at"]


def test_gallery_persist_failure_still_serves_audio(monkeypatch):
    # Persistence is best-effort: a full disk must not fail the request that produced the audio.
    cli, calls, saved = _make_client(monkeypatch)

    def _boom(wav_bytes, meta):
        raise OSError("disk full")

    monkeypatch.setattr(gallery_module, "save", _boom)
    resp = cli.post("/v1/audio/speech", json = {"input": "still speaks"})
    assert resp.status_code == 200
    assert resp.content == _WAV


def test_voice_and_speed_accepted_and_ignored(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {"input": "hi", "voice": "alloy", "speed": 1.25, "model": "tts-1"},
    )
    assert resp.status_code == 200


def test_unknown_response_format_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": "wma"})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["param"] == "response_format"
    assert "'wma'" in error["message"] and "mp3" in error["message"]
    assert calls == []  # rejected before any generation


def test_sse_stream_format_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "stream_format": "sse"})
    assert (resp.status_code, resp.json()["error"]["param"]) == (400, "stream_format")
    assert calls == []
    assert (
        cli.post("/v1/audio/speech", json = {"input": "hi", "stream_format": "audio"}).status_code
        == 200
    )


def test_null_response_format_means_wav(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": None})
    assert resp.status_code == 200


def test_empty_input_is_rejected(monkeypatch):
    # install_api_error_handlers maps validation errors to a 400 OpenAI envelope on /v1.
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": ""})
    assert resp.status_code == 400
    assert calls == []


def test_core_error_propagates(monkeypatch):
    # "No model loaded" from the TTS core keeps its status through the route.
    async def _no_model(text):
        raise HTTPException(status_code = 400, detail = "No model loaded.")

    cli, calls, saved = _make_client(monkeypatch, generate = _no_model)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi"})
    assert resp.status_code == 400
    assert saved == []


def test_wav_duration_seconds_reads_header():
    # A real 1-second 24 kHz mono WAV reports ~1.0s.
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x00\x00" * 24000)
    assert routes_module._wav_duration_seconds(buf.getvalue(), 24000) == 1.0
    # Unreadable bytes fall back to the 16-bit mono PCM estimate.
    fallback = routes_module._wav_duration_seconds(b"\x00" * (44 + 48000), 24000)
    assert fallback == 1.0


def test_the_speech_route_asks_for_the_full_audio_token_budget(monkeypatch):
    """CreateSpeech has no field for it, so the chat default of 2048 silently truncated
    any input past roughly half a minute and still returned HTTP 200 with a short WAV."""
    from core.inference.orchestrator import AUDIO_GENERATION_MAX_TOKENS

    cli, calls, _saved = _make_client(monkeypatch)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: None)
    assert cli.post("/v1/audio/speech", json = {"input": "a long script"}).status_code == 200
    payload = calls[0]["payload"]
    assert payload.max_tokens == AUDIO_GENERATION_MAX_TOKENS


def test_the_budget_leaves_room_for_the_prompt(monkeypatch):
    """The cap now lives in _tts_max_new_tokens, which both TTS routes share, rather than
    being computed at the speech route. Exercised directly since the route tests fake the
    shared core that applies it."""
    from core.inference.orchestrator import AUDIO_GENERATION_MAX_TOKENS
    from models.inference import ChatCompletionRequest

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 2048)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "x"}],
        max_tokens = AUDIO_GENERATION_MAX_TOKENS,
    )
    text = "x" * 300

    budget = routes_module._tts_max_new_tokens(payload, text)

    assert budget < 2048
    # Minus the codec wrapper too: the backends generate from a formatted prompt, not the
    # raw text, so budgeting the whole remainder left the few delimiter tokens to overflow.
    assert budget == (
        2048 - routes_module._prompt_token_estimate(text) - routes_module._TTS_PROMPT_FORMAT_RESERVE
    )


def test_an_over_context_prompt_is_a_client_error(monkeypatch):
    """Flooring at one token forwarded the whole over-context prompt anyway and failed deep
    in generation. Both routes share this guard through _generate_tts_wav."""
    from fastapi import HTTPException

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 2048)

    with pytest.raises(HTTPException) as excinfo:
        routes_module._raise_if_prompt_leaves_no_speech_budget("x" * 8000)

    assert excinfo.value.status_code == 400
    assert "too long" in str(excinfo.value.detail).lower()
    # A normal line is untouched.
    routes_module._raise_if_prompt_leaves_no_speech_budget("A short line.")


@pytest.mark.parametrize("model", [None, "", "org/B-GGUF"])
def test_speech_model_selection(monkeypatch, model):
    cli, calls, _saved = _make_client(monkeypatch)
    body = {"input": "hi", **({"model": model} if model is not None else {})}
    assert cli.post("/v1/audio/speech", json = body).status_code == 200
    assert calls[0]["requested_model"] == (model or routes_module._RELOAD_ONLY_MODEL)


@pytest.mark.parametrize("named", [False, True])
def test_only_resident_requests_use_the_pre_switch_budget(monkeypatch, named):
    async def _switch(_model, *_a, **kw):
        assert kw["require_speech"] is True
        raise RuntimeError("reached the switch")

    monkeypatch.setattr(routes_module, "_maybe_auto_switch_model", _switch)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 2048)
    monkeypatch.setattr(routes_module, "_prompt_token_estimate", lambda _t: 2048)
    payload = SimpleNamespace(audio_instructions = None, audio_language = None)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))
    model = "org/B-GGUF" if named else routes_module._RELOAD_ONLY_MODEL
    with pytest.raises(RuntimeError if named else HTTPException) as error:
        asyncio.run(
            routes_module._generate_tts_wav(
                "a long line",
                payload,
                request,
                "tester",
                requested_model = model,
            )
        )
    assert str(error.value) == "reached the switch" if named else error.value.status_code == 400


@pytest.mark.parametrize("named", [False, True])
def test_a_loaded_voice_slot_serves_only_the_resident_model_form(monkeypatch, named):
    """The voice slot owns speech when the caller names no model, which is what the
    conversation loop sends. A caller that names one keeps main's switch path exactly,
    voice slot or not: the switch is that request, so it must be reached."""

    async def _switch(_model, *_a, **kw):
        assert kw["require_speech"] is True
        raise RuntimeError("reached the switch")

    def _picked_voice(_backend):
        raise RuntimeError("reached the voice slot")

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _is_audio = True,
        _audio_type = "snac",
        _process = SimpleNamespace(poll = lambda: None),
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_maybe_auto_switch_model", _switch)
    monkeypatch.setattr(routes_module, "_llama_public_model_id", _picked_voice)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 2048)
    monkeypatch.setattr(routes_module, "_prompt_token_estimate", lambda _t: 8)
    payload = SimpleNamespace(audio_instructions = None, audio_language = None)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))
    model = "org/B-GGUF" if named else routes_module._RELOAD_ONLY_MODEL
    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            routes_module._generate_tts_wav(
                "hi",
                payload,
                request,
                "tester",
                requested_model = model,
            )
        )
    assert str(error.value) == ("reached the switch" if named else "reached the voice slot")


@pytest.mark.parametrize(
    "run_inputs, workflow",
    [
        ({"workflow": "clone", "audio_inputs": {"reference": "r.wav"}}, "clone"),
        ({"workflow": "speak", "audio_inputs": {"reference": "r.wav"}}, "clone"),
        ({"workflow": "convert", "audio_inputs": {"source": "s.wav"}}, "convert"),
    ],
)
def test_the_voice_slot_leaves_cloning_and_conversion_to_the_workflow_check(
    monkeypatch, run_inputs, workflow
):
    """The voice slot's GGUF path only speaks, so a clone or conversion must reach the
    switch, whose workflow check refuses a model that can't do it."""
    seen = {}

    async def _switch(_model, *_a, **kw):
        seen.update(kw)
        raise RuntimeError("reached the switch")

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _is_audio = True,
        _audio_type = "snac",
        _process = SimpleNamespace(poll = lambda: None),
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_maybe_auto_switch_model", _switch)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 2048)
    monkeypatch.setattr(routes_module, "_prompt_token_estimate", lambda _t: 8)
    payload = SimpleNamespace(audio_instructions = None, audio_language = None)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))
    with pytest.raises(RuntimeError, match = "reached the switch"):
        asyncio.run(
            routes_module._generate_tts_wav("hi", payload, request, "tester", run_inputs = run_inputs)
        )
    assert seen["require_audio_workflow"] == workflow


def test_the_voice_slot_budgets_speech_against_its_own_context(monkeypatch):
    """The chat slot can hold a far larger context than the voice server; budgeting against
    it admits text the voice server would then truncate or reject."""

    def _picked_voice(_backend):
        raise RuntimeError("reached the voice slot")

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _is_audio = True,
        _audio_type = "snac",
        context_length = 512,
        _process = SimpleNamespace(poll = lambda: None),
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_llama_public_model_id", _picked_voice)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 32768)
    monkeypatch.setattr(routes_module, "_prompt_token_estimate", lambda _t: 600)
    payload = SimpleNamespace(audio_instructions = None, audio_language = None)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes_module._generate_tts_wav("long text", payload, request, "tester"))
    assert error.value.status_code == 400 and "512-token context" in error.value.detail
    budget = routes_module._tts_max_new_tokens(
        SimpleNamespace(max_completion_tokens = 8192, max_tokens = None), "x", context_length = 512
    )
    assert budget < 512


@pytest.mark.parametrize("prompt_tokens, refused", [(100, False), (600, True)])
def test_streaming_speech_fits_the_voice_servers_context(monkeypatch, prompt_tokens, refused):
    """No max_new_tokens used to send the 8192 ceiling into a 4096 voice server, which ended
    the stream early after a 200 had already gone out."""
    seen = {}

    def _stream(**kwargs):
        seen.update(kwargs)
        yield b"\x00\x00"

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _process = SimpleNamespace(poll = lambda: None),
        _audio_type = "snac",
        context_length = 512,
        _orpheus_voice_prefix_ok = lambda: True,
        generate_audio_response_stream = _stream,
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_prompt_token_estimate", lambda _t: prompt_tokens)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))

    async def _run():
        response = await routes_module.openai_audio_speech_stream(
            AudioSpeechRequest(input = "hello"), request, "tester"
        )
        return [chunk async for chunk in response.body_iterator]

    if refused:
        with pytest.raises(HTTPException) as error:
            asyncio.run(_run())
        assert error.value.status_code == 400 and "512-token context" in error.value.detail
        assert seen == {}
    else:
        asyncio.run(_run())
        reserve = routes_module._TTS_PROMPT_FORMAT_RESERVE
        assert seen["max_new_tokens"] == 512 - prompt_tokens - reserve


def test_streaming_speech_honours_the_requested_model(monkeypatch):
    """`model` used to be a monitor label only: the first loaded SNAC backend spoke, whichever
    voice the caller named."""
    spoken = []

    def _backend(public_id):
        def _stream(**_kwargs):
            spoken.append(public_id)
            yield b"\x00\x00"

        return SimpleNamespace(
            is_loaded = True,
            _process = SimpleNamespace(poll = lambda: None),
            _audio_type = "snac",
            context_length = None,
            model_identifier = f"/models/{public_id.split('/')[-1]}/model.gguf",
            _openai_advertised_id = public_id,
            _orpheus_voice_prefix_ok = lambda: True,
            generate_audio_response_stream = _stream,
        )

    voice = _backend("unsloth/orpheus-3b-0.1-ft-GGUF")
    chat = _backend("me/my-orpheus-finetune-GGUF")
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module, "get_llama_cpp_backend", lambda: chat)
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))

    async def _run(model):
        response = await routes_module.openai_audio_speech_stream(
            AudioSpeechRequest(input = "hello", model = model), request, "tester"
        )
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(_run(None))
    asyncio.run(_run("me/my-orpheus-finetune-GGUF"))
    asyncio.run(_run("unsloth/orpheus-3b-0.1-ft-GGUF"))
    assert spoken == [
        "unsloth/orpheus-3b-0.1-ft-GGUF",
        "me/my-orpheus-finetune-GGUF",
        "unsloth/orpheus-3b-0.1-ft-GGUF",
    ]
    with pytest.raises(HTTPException) as error:
        asyncio.run(_run("unsloth/Spark-TTS-0.5B-GGUF"))
    assert error.value.status_code == 400
    assert "Spark-TTS-0.5B-GGUF" in error.value.detail and "Load it first" in error.value.detail
    assert len(spoken) == 3

    # Two local GGUFs sharing a basename: the exact path picks the slot, not the public id.
    voice.model_identifier = "/voices/a/model.gguf"
    voice._openai_advertised_id = None
    chat.model_identifier = "/voices/b/model.gguf"
    chat._openai_advertised_id = None
    spoken.clear()
    asyncio.run(_run("/voices/b/model.gguf"))
    asyncio.run(_run("/voices/a/model.gguf"))
    assert spoken == ["me/my-orpheus-finetune-GGUF", "unsloth/orpheus-3b-0.1-ft-GGUF"]


def test_streaming_speech_talks_to_llama_server_over_local_transport():
    """The streaming client went through an ambient HTTP(S)_PROXY while the blocking one did not."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    source = inspect.getsource(LlamaCppBackend.generate_audio_response_stream)
    client = source[source.index("httpx.Client(") :]
    client = client[: client.index(") as client")]
    assert "trust_env = False" in client and "verify = _local_ssl_context()" in client


def test_the_shared_core_guards_before_generating():
    """Wired in _generate_tts_wav so /audio/generate inherits it, not only /audio/speech."""
    import inspect

    source = inspect.getsource(routes_module._generate_tts_wav)
    assert "_raise_if_prompt_leaves_no_speech_budget(text," in source


def test_the_budget_is_rechecked_after_an_idle_model_is_restored():
    """With nothing loaded there is no context to measure, so the guard passes everything.
    Idle auto-unload leaves exactly that state, and the restore below it brings the context
    back, so the first request after an eviction reached generation over-context and came
    back as a one-token clip."""
    import inspect

    source = inspect.getsource(routes_module._generate_tts_wav)
    guards = [
        i
        for i, line in enumerate(source.splitlines())
        if "_raise_if_prompt_leaves_no_speech_budget(text," in line
    ]
    restore = next(
        i for i, line in enumerate(source.splitlines()) if "await _maybe_auto_switch_model(" in line
    )
    assert len(guards) == 2, "one check before the restore, one after"
    assert guards[0] < restore < guards[1]


def test_the_gallery_is_bounded_so_an_api_client_cannot_fill_the_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "gallery_dir", lambda: tmp_path)
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "3")
    meta = {
        "prompt": "p",
        "model": "m",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 0.1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    ids = [gallery.save(b"RIFFfake", meta)["id"] for _ in range(6)]

    remaining = {clip["id"] for clip in gallery.list_audio()}
    assert len(remaining) == 3
    # Newest kept, oldest dropped.
    assert set(ids[-3:]) == remaining


def test_the_gallery_is_bounded_by_bytes_not_only_by_count(monkeypatch, tmp_path):
    """A count alone does not bound the disk: 2000 clips of maximum-length speech is tens
    of gigabytes, and stopping /v1/audio/speech filling the disk is what the cap is for."""
    monkeypatch.setattr(gallery, "gallery_dir", lambda: tmp_path)
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "1000")
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_BYTES", str(4 * 1024))
    meta = {
        "prompt": "p",
        "model": "m",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 0.1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    ids = [gallery.save(b"R" * 1024, meta)["id"] for _ in range(10)]

    remaining = [clip["id"] for clip in gallery.list_audio()]
    assert len(remaining) == 4, remaining
    assert set(ids[-4:]) == set(remaining)


def test_one_oversized_clip_is_still_returned_rather_than_pruned_immediately(monkeypatch, tmp_path):
    """The newest clip is the one the caller just generated. Pruning it because it alone
    exceeds the quota would read as a silent failure."""
    monkeypatch.setattr(gallery, "gallery_dir", lambda: tmp_path)
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_BYTES", "64")
    meta = {
        "prompt": "p",
        "model": "m",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 0.1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    saved = gallery.save(b"R" * 4096, meta)

    assert [clip["id"] for clip in gallery.list_audio()] == [saved["id"]]


def test_unreachable_subprocess_tokenizers_use_a_conservative_byte_budget():
    estimate = routes_module._prompt_token_estimate

    for text in (
        "a " * 100,
        "مرحبا بالعالم " * 40,
        "Привет мир " * 40,
        "שלום עולם " * 40,
        "नमस्ते दुनिया " * 40,
        "你好世界" * 50,
    ):
        assert estimate(text) == len(text.encode("utf-8"))


# ── External connection proxying (provider_id) ───────────────────


def _install_external(
    monkeypatch,
    *,
    enabled = True,
    media_type = "audio/wav",
):
    created = []
    speech_calls = []

    monkeypatch.setattr(
        routes_module.providers_db,
        "get_provider",
        lambda pid: (
            {
                "provider_type": "custom",
                "display_name": "Kokoro Box",
                "base_url": "http://tts.local:8880/v1",
                "is_enabled": enabled,
            }
            if pid == "conn-1"
            else None
        ),
    )
    monkeypatch.setattr(routes_module, "validate_provider_base_url", lambda url: url)
    monkeypatch.setattr(routes_module, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test")

    class _FakeClient:
        def __init__(self, provider_type, base_url, api_key):
            created.append(
                {"provider_type": provider_type, "base_url": base_url, "api_key": api_key}
            )

        async def create_speech(self, **kwargs):
            speech_calls.append(kwargs)
            return b"external-audio", media_type

    monkeypatch.setattr(routes_module, "ExternalProviderClient", _FakeClient)
    return created, speech_calls


def test_provider_id_routes_to_external_endpoint(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    created, speech_calls = _install_external(monkeypatch)
    api_monitor.clear()
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "af_heart",
            "instructions": "Speak warmly.",
        },
    )
    assert resp.status_code == 200
    assert resp.content == b"external-audio"
    assert resp.headers["content-type"].startswith("audio/wav")
    assert calls == []  # the local TTS core never runs
    assert saved == []  # external clips skip the gallery
    assert created[0]["base_url"] == "http://tts.local:8880/v1"
    assert created[0]["provider_type"] == "custom"
    assert speech_calls[0]["model"] == "kokoro"
    assert speech_calls[0]["voice"] == "af_heart"
    assert speech_calls[0]["instructions"] == "Speak warmly."
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/speech"
    assert rows[0]["status"] == "completed"
    assert rows[0]["model"] == "kokoro"
    assert rows[0]["prompt_preview"] == "hi"


def test_external_rejects_non_wav_response_format(monkeypatch):
    cli, _calls, _saved = _make_client(monkeypatch)
    created, speech_calls = _install_external(monkeypatch, media_type = "audio/mpeg")
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "alloy",
            "response_format": "mp3",
        },
    )
    assert resp.status_code == 400
    assert "Only 'wav' is supported" in resp.json()["error"]["message"]
    assert created == []
    assert speech_calls == []


def test_external_rejects_sse_before_the_upstream_call(monkeypatch):
    cli, _calls, _saved = _make_client(monkeypatch)
    created, speech_calls = _install_external(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "alloy",
            "stream_format": "sse",
        },
    )
    assert (resp.status_code, resp.json()["error"]["param"]) == (400, "stream_format")
    assert created == [] and speech_calls == []


def test_external_missing_model_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    _install_external(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "provider_id": "conn-1"})
    assert resp.status_code == 400


def test_external_missing_voice_is_400(monkeypatch):
    cli, _calls, _saved = _make_client(monkeypatch)
    _install_external(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {"input": "hi", "provider_id": "conn-1", "model": "kokoro"},
    )
    assert resp.status_code == 400
    assert "voice" in resp.json()["error"]["message"].lower()


def test_external_unknown_provider_is_404(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    _install_external(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "missing",
            "model": "kokoro",
            "voice": "alloy",
        },
    )
    assert resp.status_code == 404


def test_external_disabled_provider_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    _install_external(monkeypatch, enabled = False)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "alloy",
        },
    )
    assert resp.status_code == 400


def test_external_upstream_error_is_502(monkeypatch):
    import httpx

    cli, calls, saved = _make_client(monkeypatch)
    created, speech_calls = _install_external(monkeypatch)
    api_monitor.clear()

    async def _boom(self, **kwargs):
        request = httpx.Request("POST", "http://tts.local:8880/v1/audio/speech")
        raise httpx.HTTPStatusError(
            "boom",
            request = request,
            response = httpx.Response(500, text = "upstream broke", request = request),
        )

    monkeypatch.setattr(routes_module.ExternalProviderClient, "create_speech", _boom)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "alloy",
        },
    )
    assert resp.status_code == 502
    assert "upstream broke" in resp.json()["error"]["message"]
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "TTS endpoint returned HTTP 500" in rows[0]["error"]


def test_external_disconnect_cancels_the_upstream_request(monkeypatch):
    _install_external(monkeypatch)
    upstream_cancelled = asyncio.Event()

    class _DisconnectingRequest:
        headers = {}

        async def is_disconnected(self):
            return True

    class _BlockingClient:
        def __init__(self, **_kwargs):
            pass

        async def create_speech(self, **_kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise

    monkeypatch.setattr(routes_module, "ExternalProviderClient", _BlockingClient)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await routes_module._external_tts_speech(
                AudioSpeechRequest(input = "hi", provider_id = "conn-1", model = "kokoro", voice = "alloy"),
                _DisconnectingRequest(),
            )

    asyncio.run(_run())
    assert upstream_cancelled.is_set()


def test_external_forwards_a_legacy_browser_key(monkeypatch):
    cli, _calls, _saved = _make_client(monkeypatch)
    created, _speech_calls = _install_external(monkeypatch)
    seen = {}

    def _resolve(provider_id, encrypted_api_key, **_kwargs):
        seen["provider_id"] = provider_id
        seen["encrypted_api_key"] = encrypted_api_key
        return "sk-from-legacy"

    monkeypatch.setattr(routes_module, "resolve_provider_api_key_or_400", _resolve)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "model": "kokoro",
            "voice": "alloy",
            "provider_base_url": "http://tts.local:8880/v1",
            "encrypted_api_key": "enc-legacy",
        },
    )
    assert resp.status_code == 200
    assert seen["encrypted_api_key"] == "enc-legacy"
    assert created[0]["api_key"] == "sk-from-legacy"


def test_external_rejects_a_legacy_key_snapshotted_for_another_base_url(monkeypatch):
    cli, _calls, _saved = _make_client(monkeypatch)
    _install_external(monkeypatch)

    def _must_not_resolve(*_args, **_kwargs):
        pytest.fail("the stale legacy key was decrypted")

    monkeypatch.setattr(routes_module, "resolve_provider_api_key_or_400", _must_not_resolve)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "hi",
            "provider_id": "conn-1",
            "provider_base_url": "http://old-tts.local:8880/v1",
            "model": "kokoro",
            "voice": "alloy",
            "encrypted_api_key": "enc-old-key",
        },
    )
    assert resp.status_code == 409
    assert "changed" in resp.json()["error"]["message"].lower()


def test_external_tts_drops_the_local_keepwarm_count_before_proxy(monkeypatch):
    from core.inference import llama_keepwarm

    monkeypatch.setattr(llama_keepwarm, "_inflight", 1)
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    observed_counts = []

    @asynccontextmanager
    async def _monitor(*_args, **_kwargs):
        yield "monitor-1"

    async def _proxy(_body, _request):
        observed_counts.append(
            llama_keepwarm.other_inference_request_count(current_request_counted = False)
        )
        return routes_module.Response(content = b"external-audio", media_type = "audio/wav")

    monkeypatch.setattr(routes_module, "_monitored_media_request", _monitor)
    monkeypatch.setattr(routes_module, "_external_tts_speech", _proxy)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/speech",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }
    )

    asyncio.run(
        routes_module.openai_audio_speech(
            AudioSpeechRequest(
                input = "hi",
                provider_id = "conn-1",
                model = "kokoro",
                voice = "alloy",
            ),
            request,
            "test-user",
        )
    )
    assert observed_counts == [0]


def test_provider_client_appends_speech_path_before_the_base_query(monkeypatch):
    sent = {}

    class _Response:
        content = b"audio"
        headers = {"content-type": "audio/wav"}

        def raise_for_status(self):
            return None

    async def _post(url, **kwargs):
        sent["url"] = url
        sent["json"] = kwargs["json"]
        return _Response()

    monkeypatch.setattr(provider_module._http_client, "post", _post)
    client = ExternalProviderClient(
        "custom",
        "http://127.0.0.1:8880/v1?api-version=2026-08-24",
        "sk-test",
    )
    asyncio.run(client.create_speech(text = "hi", model = "kokoro", instructions = "Speak warmly."))
    assert sent["url"] == ("http://127.0.0.1:8880/v1/audio/speech?api-version=2026-08-24")
    assert sent["json"]["instructions"] == "Speak warmly."
    assert "stream" not in sent["json"]


def test_provider_client_merges_concatenated_wav_segments(monkeypatch):
    import io
    import struct
    import wave

    def _wav(frames, rate = 24_000):
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(frames)
        return output.getvalue()

    first_frames = b"\x01\x00" * 2
    second_frames = b"\x02\x00" * 3

    class _Response:
        content = _wav(first_frames) + _wav(second_frames)
        headers = {"content-type": "audio/wav"}

        def raise_for_status(self):
            return None

    async def _post(_url, **_kwargs):
        return _Response()

    monkeypatch.setattr(provider_module._http_client, "post", _post)
    client = ExternalProviderClient("custom", "http://127.0.0.1:8880/v1", "")
    audio, media_type = asyncio.run(client.create_speech(text = "one. two.", model = "kokoro"))

    with wave.open(io.BytesIO(audio), "rb") as reader:
        assert reader.getnframes() == 5
        assert reader.readframes(5) == first_frames + second_frames
    assert media_type == "audio/wav"
    assert audio.count(b"RIFF") == 1
    single = _wav(first_frames)
    incompatible = single + _wav(second_frames, rate = 16_000)
    assert provider_module._merge_concatenated_wav_segments(single) == single
    assert provider_module._merge_concatenated_wav_segments(incompatible) == incompatible
    assert provider_module._merge_concatenated_wav_segments(b"not-a-wave") == b"not-a-wave"
    malformed = b"RIFF" + (12).to_bytes(4, "little") + b"WAVEJUNK" + (100).to_bytes(4, "little")
    assert provider_module._merge_concatenated_wav_segments(malformed * 2) == malformed * 2
    fmt = struct.pack("<HHIIHH", 1, 1, 8_000, 16_000, 2, 16)
    body = (
        b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + fmt
        + b"data"
        + (100).to_bytes(4, "little")
        + b"\x01\x00"
    )
    truncated = b"RIFF" + len(body).to_bytes(4, "little") + body
    assert provider_module._merge_concatenated_wav_segments(truncated * 2) == truncated * 2
    tiny_pseudo_segment = b"RIFF" + (5).to_bytes(4, "little") + b"WAVE\x00"
    many_pseudo_segments = tiny_pseudo_segment * 10_000
    assert (
        provider_module._merge_concatenated_wav_segments(many_pseudo_segments)
        == many_pseudo_segments
    )
    too_many_valid_segments = _wav(b"") * (provider_module._MAX_CONCATENATED_WAV_SEGMENTS + 1)
    assert (
        provider_module._merge_concatenated_wav_segments(too_many_valid_segments)
        == too_many_valid_segments
    )


def test_provider_client_merges_wav_off_the_event_loop(monkeypatch):
    merge_started = threading.Event()
    release_merge = threading.Event()

    class _Response:
        content = b"audio"
        headers = {"content-type": "audio/wav"}

        def raise_for_status(self):
            return None

    async def _post(_url, **_kwargs):
        return _Response()

    def _blocking_merge(audio, _cancelled):
        merge_started.set()
        release_merge.wait()
        return audio

    monkeypatch.setattr(provider_module._http_client, "post", _post)
    monkeypatch.setattr(provider_module, "_merge_concatenated_wav_segments", _blocking_merge)

    async def _run():
        client = ExternalProviderClient("custom", "http://127.0.0.1:8880/v1", "")
        speech_task = asyncio.create_task(client.create_speech(text = "hi", model = "kokoro"))
        while not merge_started.is_set():
            await asyncio.sleep(0)
        heartbeat_seen = False
        await asyncio.sleep(0)
        heartbeat_seen = True
        release_merge.set()
        await speech_task
        return heartbeat_seen

    assert asyncio.run(_run()) is True


def test_cancelling_provider_speech_stops_the_wav_worker(monkeypatch):
    merge_started = threading.Event()
    merge_cancel_seen = threading.Event()
    merge_stopped = threading.Event()
    release_merge = threading.Event()

    class _Response:
        content = b"audio"
        headers = {"content-type": "audio/wav"}

        def raise_for_status(self):
            return None

    async def _post(_url, **_kwargs):
        return _Response()

    def _cancellable_merge(audio, cancelled):
        merge_started.set()
        cancelled.wait()
        merge_cancel_seen.set()
        release_merge.wait()
        merge_stopped.set()
        return audio

    monkeypatch.setattr(provider_module._http_client, "post", _post)
    monkeypatch.setattr(provider_module, "_merge_concatenated_wav_segments", _cancellable_merge)

    async def _run():
        client = ExternalProviderClient("custom", "http://127.0.0.1:8880/v1", "")
        speech_task = asyncio.create_task(client.create_speech(text = "hi", model = "kokoro"))
        while not merge_started.is_set():
            await asyncio.sleep(0)
        speech_task.cancel()
        while not merge_cancel_seen.is_set():
            await asyncio.sleep(0)
        speech_task.cancel()
        await asyncio.sleep(0)
        assert not speech_task.done()
        release_merge.set()
        with pytest.raises(asyncio.CancelledError):
            await speech_task
        assert merge_stopped.is_set()

    asyncio.run(_run())


def test_external_provider_reads_do_not_block_the_event_loop(monkeypatch):
    _install_external(monkeypatch)
    original_get_provider = routes_module.providers_db.get_provider
    read_started = threading.Event()
    release_read = threading.Event()
    heartbeat_seen = threading.Event()
    event_loop_blocked = []

    def _slow_get_provider(provider_id):
        read_started.set()
        release_read.wait()
        return original_get_provider(provider_id)

    monkeypatch.setattr(routes_module.providers_db, "get_provider", _slow_get_provider)

    class _ConnectedRequest:
        headers = {}

        async def is_disconnected(self):
            return False

    def _watchdog():
        if not read_started.wait(timeout = 1):
            event_loop_blocked.append(True)
            release_read.set()
            return
        if not heartbeat_seen.wait(timeout = 1):
            event_loop_blocked.append(True)
        release_read.set()

    async def _run():
        watchdog = threading.Thread(target = _watchdog)
        watchdog.start()
        speech = asyncio.create_task(
            routes_module._external_tts_speech(
                AudioSpeechRequest(input = "hi", provider_id = "conn-1", model = "kokoro", voice = "alloy"),
                _ConnectedRequest(),
            )
        )
        await asyncio.sleep(0)
        heartbeat_seen.set()
        release_read.set()
        await speech
        watchdog.join()

    asyncio.run(_run())
    assert event_loop_blocked == []


def test_external_rejects_a_cross_process_provider_edit_after_resolving_its_key(monkeypatch):
    old_config = {
        "provider_type": "custom",
        "display_name": "Old TTS",
        "base_url": "http://old-tts.local:8880/v1",
        "is_enabled": True,
    }
    new_config = {
        **old_config,
        "display_name": "New TTS",
        "base_url": "http://new-tts.local:8880/v1",
    }
    # A second process is not covered by provider_config_guard. It can update
    # the row and then the secret while this process is resolving that secret.
    snapshots = iter((old_config, old_config, new_config))
    monkeypatch.setattr(routes_module.providers_db, "get_provider", lambda _pid: next(snapshots))
    monkeypatch.setattr(routes_module, "validate_provider_base_url", lambda url: url)
    key_resolved = False

    def _resolve(*_args, **_kwargs):
        nonlocal key_resolved
        key_resolved = True
        return "new-key"

    monkeypatch.setattr(routes_module, "resolve_provider_api_key_or_400", _resolve)

    class _ConnectedRequest:
        headers = {}

        async def is_disconnected(self):
            return False

    async def _run():
        with pytest.raises(HTTPException) as excinfo:
            await routes_module._external_tts_speech(
                AudioSpeechRequest(input = "hi", provider_id = "conn-1", model = "kokoro", voice = "alloy"),
                _ConnectedRequest(),
            )
        assert excinfo.value.status_code == 409

    asyncio.run(_run())
    assert key_resolved


def test_speech_opens_a_monitor_row(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    api_monitor.clear()
    assert cli.post("/v1/audio/speech", json = {"input": "hello sloth"}).status_code == 200
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/speech"
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_preview"] == "hello sloth"
    # Relabelled to the loaded TTS model, not the informational body.model.
    assert rows[0]["model"] == "unsloth/orpheus-3b-0.1-ft"


def test_v1_audio_generate_opens_a_monitor_row_and_the_chat_mount_does_not(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    cli.app.include_router(router, prefix = "/api/inference")
    api_monitor.clear()
    body = {"messages": [{"role": "user", "content": "read me"}]}
    assert cli.post("/api/inference/audio/generate", json = body).status_code == 200
    assert api_monitor.snapshot(include_details = False) == []
    assert cli.post("/v1/audio/generate", json = body).status_code == 200
    (row,) = api_monitor.snapshot(include_details = False)
    assert (row["endpoint"], row["status"], row["model"]) == (
        "/v1/audio/generate",
        "completed",
        "unsloth/orpheus-3b-0.1-ft",
    )


def test_tts_failure_records_an_error_row(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch, generate = _boom)
    api_monitor.clear()
    assert cli.post("/v1/audio/speech", json = {"input": "hi"}).status_code == 400
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "No model loaded."


def test_rejected_response_format_records_nothing(monkeypatch):
    # Refused before any work, so it is not traffic the monitor should show.
    cli, calls, saved = _make_client(monkeypatch)
    api_monitor.clear()
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": "wma"})
    assert resp.status_code == 400
    assert api_monitor.snapshot(include_details = False) == []


def test_client_abort_records_a_cancelled_row(monkeypatch):
    # The disconnect watcher turns a client abort into a 499, not a CancelledError.
    async def _cancelled(text):
        raise HTTPException(status_code = 499, detail = "Audio generation cancelled")

    cli, calls, saved = _make_client(monkeypatch, generate = _cancelled)
    api_monitor.clear()
    assert cli.post("/v1/audio/speech", json = {"input": "hi"}).status_code == 499
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "cancelled"
    assert not rows[0]["error"]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("/home/ana/models/orpheus-3b-0.1-ft", "orpheus-3b-0.1-ft"),
        ("/srv/voices/Kokoro-82M-Q4_K_M.gguf", "Kokoro-82M-Q4_K_M"),
        (r"C:\Users\ana\models\kokoro-82m.gguf", "kokoro-82m"),
        (r"\\fileserver\share\models\orpheus-3b", "orpheus-3b"),
    ],
)
def test_a_failure_before_the_relabel_does_not_leak_the_requested_path(
    monkeypatch, requested, expected
):
    """body.model is informational and is echoed straight into the row, so the relabel on
    the success path is the only thing that ever cleaned it. A failure before generation
    (no audio model loaded) left the raw client string on a terminal row that the monitor
    overlay polls and serves. Windows and UNC forms are covered because redacting a host
    path is the whole point."""

    cli, calls, saved = _make_client(monkeypatch, generate = _boom)
    api_monitor.clear()
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "model": requested})
    assert resp.status_code == 400
    row = api_monitor.snapshot(include_details = False)[0]
    assert row["status"] == "error"
    assert row["model"] == expected
    assert "/" not in row["model"] and "\\" not in row["model"]


def test_an_ordinary_model_id_is_still_recorded_verbatim(monkeypatch):
    # The redaction must not rewrite the ids clients actually send.
    cli, calls, saved = _make_client(monkeypatch, generate = _boom)
    for requested in ("tts-1", "gpt-4o-mini-tts", "unsloth/orpheus-3b-0.1-ft"):
        api_monitor.clear()
        assert (
            cli.post("/v1/audio/speech", json = {"input": "hi", "model": requested}).status_code
            == 400
        )
        assert api_monitor.snapshot(include_details = False)[0]["model"] == requested


def test_voice_load_applies_the_managed_account_gate_before_resolving(monkeypatch):
    """A managed account may only load a model within its grants, and an absent token is
    the account's own, never the installation's ambient Hub credential. /load and
    /validate apply both before resolving; /voice/load resolved the caller's identifier
    with the caller's token first, which made the voice slot a second door past both."""
    from hub.services.models import account_access
    from utils import models as models_module

    monkeypatch.setattr(account_access, "managed_account", lambda: True)

    def _resolve_before_gate(*_a, **_k):
        raise AssertionError("resolved the model before the access check")

    monkeypatch.setattr(models_module.ModelConfig, "from_identifier", _resolve_before_gate)

    def _deny(reference, repo_type = "model"):
        raise HTTPException(status_code = 403, detail = f"no grant for {reference}")

    monkeypatch.setattr(account_access, "require_model_access", _deny)
    monkeypatch.setattr(account_access, "account_hf_token", lambda token: "account-token")
    request = routes_module._VoiceLoadRequest(model_path = "org/private-voice-GGUF")
    with pytest.raises(HTTPException) as denied:
        asyncio.run(routes_module.voice_load_model(request, "tester"))
    assert denied.value.status_code == 403


def test_voice_load_resolves_with_the_account_token_not_the_callers(monkeypatch):
    from hub.services.models import account_access
    from utils import models as models_module

    seen = {}

    def _resolve(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after resolve")

    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "require_model_access", lambda *a, **k: None)
    monkeypatch.setattr(account_access, "account_hf_token", lambda token: "account-token")
    monkeypatch.setattr(models_module.ModelConfig, "from_identifier", staticmethod(_resolve))
    request = routes_module._VoiceLoadRequest(
        model_path = "org/voice-GGUF", hf_token = "callers-own-token"
    )
    with pytest.raises(HTTPException) as failed:
        asyncio.run(routes_module.voice_load_model(request, "tester"))
    assert failed.value.status_code == 400  # the route wraps the resolve failure
    assert seen["hf_token"] == "account-token"


def test_voice_load_claims_the_gpu_for_chat_before_spawning():
    """A voice load went around the GPU arbiter, so a resident Images/Video pipeline stayed put
    beside the new llama-server."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    claim = source.index("acquire_for_request, _CHAT, None, alongside = True")
    spawn = source.index("voice_backend.load_model, intent, load_cancel_event = load_cancel")
    assert claim < spawn
    # The in-flight marker covers the whole request, resolution and warm-up included, so an
    # unload or an Images/Video acquire anywhere in it finds a load to cancel.
    assert source.index("in_flight.__enter__()") < source.index("_resolve_config")
    assert "_leave_in_flight()" in source[source.index('"status": "loaded"') :]
    unload = inspect.getsource(routes_module.voice_unload_model)
    assert "require_no_foreign_generations(scope)" in unload


def test_release_chat_gpu_claim_keeps_chat_while_the_voice_slot_is_live(monkeypatch):
    """Unloading the chat model released CHAT with the voice llama-server still holding VRAM, so
    the next Images/Video load saw no owner and allocated beside it."""
    import core.inference.gpu_arbiter as arb

    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    live = {"active": True}
    voice = type("Voice", (), {"is_active": property(lambda self: live["active"])})()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    arb.acquire_for(arb.CHAT)
    assert routes_module.release_chat_gpu_claim() is False
    assert arb.current_owner() == arb.CHAT
    live["active"] = False
    assert routes_module.release_chat_gpu_claim() is True
    assert arb.current_owner() is None


def test_a_zero_vram_primary_keeps_the_chat_claim_while_the_voice_slot_is_live(monkeypatch):
    """Replacing the primary with a CPU-only chat model released CHAT straight away with no extra
    slot kept, so a live voice llama-server was left beside the next Images/Video load."""
    import core.inference.gpu_arbiter as arb
    from core.inference import model_slots

    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    live = {"active": True}
    voice = type("Voice", (), {"is_active": property(lambda self: live["active"])})()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    arb.acquire_for(arb.CHAT)
    routes_module._release_chat_for_zero_vram_primary()
    assert arb.current_owner() == arb.CHAT
    live["active"] = False
    routes_module._release_chat_for_zero_vram_primary()
    assert arb.current_owner() is None


def test_voice_load_undoes_itself_when_the_gpu_changed_hands_during_the_spawn():
    """load_model clears the cancel event an eviction set between the claim and the spawn, so
    the loader rechecks the owner after the load, like /load, instead of trusting the event."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    spawn = source.index("voice_backend.load_model, intent, load_cancel_event = load_cancel")
    recheck = source.index("if current_owner() != _CHAT:")
    assert recheck > spawn
    assert "await asyncio.to_thread(voice_backend.unload_model)" in source[recheck:]
    assert "status_code = 409" in source[recheck:]


def test_voice_unload_cancels_a_load_that_has_not_spawned_yet(monkeypatch):
    """/voice/unload during the GGUF download answered not_loaded (no process yet) and left the
    in-flight load to finish onto the GPU after an explicit unload."""
    from core.inference.llama_cpp import voice_load_in_flight

    calls = []
    voice = type(
        "Voice",
        (),
        {
            "is_active": False,
            "model_identifier": "voice.gguf",
            "unload_model": lambda self: calls.append("unload") or True,
        },
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "account_scope", lambda: None)
    assert asyncio.run(routes_module.voice_unload_model("s")) == {"status": "not_loaded"}
    assert calls == []
    with voice_load_in_flight():
        assert asyncio.run(routes_module.voice_unload_model("s"))["status"] == "unloaded"
    assert calls == ["unload"]


def test_voice_unload_drops_an_empty_chat_claim(monkeypatch):
    """The voice slot as the last CHAT resident left the claim with the previous account after
    its unload, so the next account saw a hidden foreign resident with no model behind it."""
    import core.inference.gpu_arbiter as arb

    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    live = {"active": True}
    voice = type(
        "Voice",
        (),
        {
            "is_active": property(lambda self: live["active"]),
            "model_identifier": "voice.gguf",
            "unload_model": lambda self: live.update(active = False) or True,
        },
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "account_scope", lambda: None)
    arb.acquire_for(arb.CHAT)
    assert asyncio.run(routes_module.voice_unload_model("s"))["status"] == "unloaded"
    assert arb.current_owner() is None


def test_voice_load_undoes_itself_when_an_unload_landed_before_the_spawn():
    """An unload between the in-flight mark and load_model set a cancel event load_model then
    cleared, with CHAT still the owner, so the server came up after an explicit unload. The
    unload epoch is read before the claim and compared after the load."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    read = source.index('unload_epoch = getattr(voice_backend, "_unload_epoch", None)')
    resolve = (
        source.index("resolve_audio_model_config(")
        if "resolve_audio_model_config(" in source
        else source.index("GgufLoadIntent(")
    )
    claim = source.index("acquire_for_request, _CHAT, None, alongside = True")
    spawn = source.index("voice_backend.load_model, intent, load_cancel_event = load_cancel")
    check = source.index('getattr(voice_backend, "_unload_epoch", None) != unload_epoch')
    warm = source.index('generate_audio_response, "Hi there."')
    after_warm = source.index('getattr(voice_backend, "_unload_epoch", None) != unload_epoch', warm)
    # Read before the model is resolved, so an unload during resolution or preflight counts too,
    # and checked again after the warm-up, whose errors are swallowed.
    assert read < resolve < claim < spawn < check < warm < after_warm
    assert "status_code = 409" in source[check:]


def test_voice_status_hides_another_accounts_resident(monkeypatch):
    """Every authenticated account read the loaded voice's identifier (a local GGUF's absolute
    path) off the singleton slot; the chat status hides a foreign resident and this now does too."""
    import json

    voice = type(
        "Voice",
        (),
        {
            "is_active": True,
            "is_loaded": True,
            "model_identifier": "C:/voices/private.gguf",
            "_process": SimpleNamespace(poll = lambda: None),
        },
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: True)
    hidden = asyncio.run(routes_module.voice_slot_status("s"))
    assert json.loads(hidden.body) == {"loaded": True, "yours": False}
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: False)
    shown = asyncio.run(routes_module.voice_slot_status("s"))
    assert shown["model"] == "C:/voices/private.gguf"


def test_voice_unload_refuses_another_accounts_resident(monkeypatch):
    """Any account could stop the singleton voice slot while the status hid it from them; /unload
    answers 404 for a hidden chat resident, and this now does too."""
    from fastapi import HTTPException

    stopped = []
    voice = type(
        "Voice",
        (),
        {
            "is_active": True,
            "is_loaded": True,
            "model_identifier": "C:/voices/private.gguf",
            "unload_model": lambda self: stopped.append(True),
        },
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "account_scope", lambda: None)
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: True)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(routes_module.voice_unload_model("s"))
    assert refused.value.status_code == 404
    assert stopped == []


@pytest.mark.parametrize("voice_active", [True, False])
def test_voice_load_refuses_to_replace_another_accounts_resident(monkeypatch, voice_active):
    """A load replaced the singleton voice server another account had loaded, and with the slot
    empty it joined another account's chat claim, hidden from the account that loaded it."""
    from fastapi import HTTPException

    loads = []
    voice = type(
        "Voice",
        (),
        {
            "is_active": voice_active,
            "is_loaded": voice_active,
            "load_model": lambda self, *a, **k: loads.append(a),
        },
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(
        routes_module.account_access, "require_idle_other_accounts", lambda *a, **k: None
    )
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: True)
    request = routes_module._VoiceLoadRequest(model_path = "x.gguf")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(routes_module.voice_load_model(request, "s"))
    assert refused.value.status_code == 404
    assert loads == []


def test_voice_unload_leaves_another_accounts_load_in_flight(monkeypatch):
    """Before a load spawns or claims CHAT, nothing marked it as another account's, so any
    account could cancel it through /voice/unload."""
    from fastapi import HTTPException

    import core.inference.llama_cpp as llama_cpp

    stopped = []
    voice = type(
        "Voice",
        (),
        {"is_active": False, "is_loaded": False, "unload_model": lambda self: stopped.append(True)},
    )()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(llama_cpp, "voice_load_active", lambda: True)
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: False)
    monkeypatch.setattr(routes_module, "_voice_loading_account", ["account-a"])
    monkeypatch.setattr(routes_module.account_access, "account_scope", lambda: "account-b")
    monkeypatch.setattr(
        "core.inference.gpu_arbiter.require_no_foreign_generations", lambda *a, **k: None
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(routes_module.voice_unload_model("s"))
    assert refused.value.status_code == 404
    assert stopped == []


def test_a_voice_server_that_exited_is_not_reported_or_reused_as_loaded(monkeypatch):
    """is_loaded stayed true after the voice llama-server died, so /voice/status said loaded and
    /voice/load answered already_loaded instead of relaunching it."""
    import inspect

    dead = SimpleNamespace(poll = lambda: 1)
    voice = SimpleNamespace(
        is_active = True,
        is_loaded = True,
        _process = dead,
        model_identifier = "unsloth/orpheus-3b-0.1-ft-GGUF",
        _audio_type = "snac",
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    monkeypatch.setattr(routes_module.account_access, "resident_hidden", lambda *a, **k: False)
    status = asyncio.run(routes_module.voice_slot_status("s"))
    assert status["loaded"] is False and status["loading"] is False and status["model"] is None

    source = inspect.getsource(routes_module.voice_load_model)
    fast_path = source[
        source.index("voice_backend.is_loaded") : source.index('"status": "already_loaded"')
    ]
    assert "_voice_server_alive(voice_backend)" in fast_path
    # Neither speech route serves from it, so the client's 400-then-reload path runs.
    tts = inspect.getsource(routes_module._generate_tts_wav)
    serves = tts[tts.index("_voice_slot_serves = bool(") :][:400]
    assert "_voice_server_alive(_voice_backend)" in serves
    stream = inspect.getsource(routes_module.openai_audio_speech_stream)
    assert "_voice_server_alive(candidate)" in stream[stream.index("loaded = [") :][:500]


def test_voice_loads_run_one_at_a_time():
    """Two loads for different voices both passed the already-loaded check, and the second then
    replaced the server the first had just reported as loaded."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    lock = source.index("async with _voice_load_lock():")
    fast_path = source.index('"status": "already_loaded"')
    spawn = source.index("voice_backend.load_model, intent, load_cancel_event = load_cancel")
    assert lock < fast_path < spawn


def test_voice_load_rejects_a_context_above_the_requestable_ceiling():
    """/voice/load models its load as a chat LoadRequest for the training-coexistence
    guard, and that model caps max_seq_length at MAX_REQUESTABLE_CONTEXT. Without the
    same bound on n_ctx, an oversized value passed validation and then blew up inside
    the handler as a 500, after the HF resolve, instead of a 422 up front."""
    from pydantic import ValidationError

    from core.inference.runtime_context import MAX_REQUESTABLE_CONTEXT

    with pytest.raises(ValidationError):
        routes_module._VoiceLoadRequest(
            model_path = "unsloth/orpheus-3b-0.1-ft-GGUF", n_ctx = MAX_REQUESTABLE_CONTEXT + 1
        )
    assert (
        routes_module._VoiceLoadRequest(
            model_path = "unsloth/orpheus-3b-0.1-ft-GGUF", n_ctx = MAX_REQUESTABLE_CONTEXT
        ).n_ctx
        == MAX_REQUESTABLE_CONTEXT
    )
    # 0 keeps meaning "model default".
    assert routes_module._VoiceLoadRequest(model_path = "x.gguf", n_ctx = 0).n_ctx == 0


def test_audio_generate_answers_with_the_text_the_clip_speaks(monkeypatch):
    """Content is the whole spoken text, not a status label cut at 100 characters."""
    cli, _calls, _saved = _make_client(monkeypatch)
    text = (
        "This sentence is deliberately longer than one hundred characters so that a "
        "truncated label would show it. "
    ) * 2
    resp = cli.post(
        "/v1/audio/generate",
        json = {"model": "default", "messages": [{"role": "user", "content": text}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == text


def test_a_foreign_resident_voice_is_not_served(monkeypatch):
    """Account B could take speech from account A's loaded voice through the omitted-model form
    of /audio/speech and through the stream route, while /voice/status hid it."""
    import inspect

    tts = inspect.getsource(routes_module._generate_tts_wav)
    serves = tts.index("_voice_slot_serves = bool(")
    assert 'not account_access.resident_hidden("chat")' in tts[serves : serves + 400]
    stream = inspect.getsource(routes_module.openai_audio_speech_stream)
    assert stream.index('account_access.resident_hidden("chat")') < stream.index("loaded = [")


def test_a_streaming_clip_counts_as_a_generation_while_it_plays(monkeypatch):
    """The stream route registered no generation, so another account's /voice/unload passed the
    foreign-generation gate and stopped the voice server mid-clip."""
    from state import active_generations

    counts = []

    def _stream(**kwargs):
        counts.append(active_generations.count())
        yield b"\x00\x00"
        counts.append(active_generations.count())

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _process = SimpleNamespace(poll = lambda: None),
        _audio_type = "snac",
        context_length = None,
        _orpheus_voice_prefix_ok = lambda: True,
        generate_audio_response_stream = _stream,
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_llama_public_model_id", lambda _b: "voice")
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))
    before = active_generations.count()

    async def _run():
        response = await routes_module.openai_audio_speech_stream(
            AudioSpeechRequest(input = "hello"), request, "tester"
        )
        return [chunk async for chunk in response.body_iterator]

    assert asyncio.run(_run()) == [b"\x00\x00"]
    assert counts == [before + 1, before + 1]
    assert active_generations.count() == before


def test_a_rejected_voice_load_gives_the_chat_claim_back():
    """A GGUF that started but was not a supported TTS type (or failed to start) was torn down
    with the CHAT claim left under the caller's account and nothing resident."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    undo = source.index("async def _undo_load():")
    body = source[undo : undo + 500]
    # The marker counts as a live voice slot to the release predicate, so it ends first.
    assert body.index("_leave_in_flight()") < body.index(
        "await asyncio.to_thread(release_chat_gpu_claim)"
    )
    for marker in (
        'detail = f"Failed to load voice model: {e}"',
        'detail = "Voice model failed to start."',
        "Not a supported TTS type",
    ):
        at = source.index(marker)
        assert "await _undo_load()" in source[at - 400 : at + 200], marker
    # Both unload-epoch rejections too: the unload that moved the epoch could not release the
    # claim itself, since this load was still marked in flight when it ran.
    epoch = 'detail = "The voice model was unloaded while it was loading. Load it again."'
    hits = [m.start() for m in re.finditer(re.escape(epoch), source)]
    assert len(hits) == 3
    for at in hits:
        assert "await _undo_load()" in source[at - 300 : at]


def test_the_in_flight_marker_keeps_the_chat_claim_until_it_ends(monkeypatch):
    """release_chat_gpu_claim treats a load still marked in flight as a live voice slot, so a
    cleanup that releases inside the marker keeps the stale claim."""
    import core.inference.gpu_arbiter as arb
    from core.inference.llama_cpp import voice_load_in_flight

    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    voice = type("Voice", (), {"is_active": False, "model_identifier": None})()
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice)
    arb.acquire_for(arb.CHAT)
    with voice_load_in_flight():
        routes_module.release_chat_gpu_claim()
        assert arb.current_owner() == arb.CHAT
    routes_module.release_chat_gpu_claim()
    assert arb.current_owner() is None


def test_voice_load_with_a_new_context_size_is_not_already_loaded():
    """The fast path compared model, variant and --parallel but not n_ctx, so a reload for a
    bigger context answered already_loaded and the old server kept serving."""
    import inspect

    source = inspect.getsource(routes_module.voice_load_model)
    fast_path = source.index('"status": "already_loaded"')
    condition = source[source.rindex("if (", 0, fast_path) : fast_path]
    assert 'getattr(voice_backend, "requested_n_ctx", 0)' in condition
    assert "int(request.n_ctx or 0)" in condition


def test_an_eviction_before_the_spawn_cancels_the_voice_load_durably(monkeypatch):
    """load_model clears the backend's cancel event at startup, so an Images/Video eviction (or an
    unload) landing between the claim and the spawn was lost and the server spawned beside the new
    owner. The load now carries its own event, set by the eviction and by /voice/unload."""
    import inspect

    from core.inference import gpu_arbiter
    from core.inference.llama_cpp import cancel_voice_loads, voice_load_in_flight

    own = threading.Event()
    with voice_load_in_flight(own):
        assert cancel_voice_loads() == 1
        assert own.is_set()
    assert cancel_voice_loads() == 0

    source = inspect.getsource(routes_module.voice_load_model)
    marker = source.index("voice_load_in_flight(load_cancel)")
    check = source.index("if load_cancel.is_set():")
    spawn = source.index("voice_backend.load_model, intent, load_cancel_event = load_cancel")
    assert marker < check < spawn
    assert "cancel_voice_loads()" in inspect.getsource(gpu_arbiter._evict_chat)
    assert "cancel_voice_loads()" in inspect.getsource(routes_module.voice_unload_model)


def test_a_cancel_landing_at_the_spawn_never_leaves_a_child_running():
    """load_model does not hold the backend lock across the spawn, so an unload or an eviction
    can set the cancel after the last pre-spawn check. The spawn lock now re-reads the cancel
    before Popen (no child) and right after it (the child is killed here, not by the route)."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    source = inspect.getsource(LlamaCppBackend.load_model)
    helper = source[source.index("def _spawn_and_wait(") :]
    popen = helper.index("_spawned = subprocess.Popen(")
    lock = helper.rindex("with self._spawn_lock:", 0, popen)
    assert "if _load_cancelled():" in helper[lock:popen]
    after = helper[popen : popen + 2500]
    recheck = after.index("if self._spawn_is_stale() or _load_cancelled():")
    assert "self._kill_process()" in after[recheck : recheck + 400]


def test_a_forced_swap_cancels_a_streaming_clip(monkeypatch):
    """The stream route registered its generation with an event nothing read, so a forced
    model swap's cancel_all left the clip reading from llama-server until it finished."""
    from state import active_generations

    seen = []

    def _stream(**kwargs):
        cancel = kwargs["cancel_event"]
        active_generations.cancel_all()
        seen.append(cancel.is_set())
        yield b"\x00\x00"

    voice_backend = SimpleNamespace(
        is_loaded = True,
        _process = SimpleNamespace(poll = lambda: None),
        _audio_type = "snac",
        context_length = None,
        _orpheus_voice_prefix_ok = lambda: True,
        generate_audio_response_stream = _stream,
    )
    monkeypatch.setattr(routes_module, "get_voice_llama_backend", lambda: voice_backend)
    monkeypatch.setattr(routes_module, "_llama_public_model_id", lambda _b: "voice")
    request = SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))

    async def _run():
        response = await routes_module.openai_audio_speech_stream(
            AudioSpeechRequest(input = "hello"), request, "tester"
        )
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(_run())
    assert seen == [True]


def test_the_streaming_backend_stops_reading_once_cancelled(monkeypatch):
    import contextlib
    import threading

    import core.inference.llama_cpp as llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend

    cancel = threading.Event()
    read = []

    class _Response:
        status_code = 200

        def iter_lines(self):
            for i in range(5):
                read.append(i)
                if i == 1:
                    cancel.set()
                yield 'data: {"content": ""}'

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @contextlib.contextmanager
        def stream(self, *a, **k):
            yield _Response()

    monkeypatch.setattr(llama_cpp.httpx, "Client", _Client)
    codec = SimpleNamespace(has_codec = lambda _t: True)
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", codec)
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "base_url", "http://127.0.0.1:1", raising = False)

    out = list(backend.generate_audio_response_stream("hi", "snac", cancel_event = cancel))
    assert out == []
    assert read == [0, 1]


def test_a_cancel_wakes_a_stream_read_blocked_in_prefill(monkeypatch):
    """The cancel check only ran between SSE lines, so a read blocked in prefill sat out its
    300 s timeout past the forced swap's drain window."""
    import contextlib
    import threading
    import time

    import httpx

    import core.inference.llama_cpp as llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend

    cancel, shut = threading.Event(), threading.Event()

    class _Response:
        status_code = 200

        def iter_lines(self):
            shut.wait(10)  # recv() blocked until the socket is shut down
            raise httpx.ReadError("socket shut down")

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

        @contextlib.contextmanager
        def stream(self, *a, **k):
            yield _Response()

    monkeypatch.setattr(llama_cpp.httpx, "Client", _Client)
    monkeypatch.setattr(
        LlamaCppBackend, "_shutdown_active_httpx_sockets", staticmethod(lambda client: shut.set())
    )
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", SimpleNamespace(has_codec = lambda _t: True))
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "base_url", "http://127.0.0.1:1", raising = False)
    backend = LlamaCppBackend.__new__(LlamaCppBackend)

    threading.Timer(0.2, cancel.set).start()
    start = time.monotonic()
    out = list(backend.generate_audio_response_stream("hi", "snac", cancel_event = cancel))
    assert out == []
    assert time.monotonic() - start < 2
