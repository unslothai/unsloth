# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/speech.

The TTS core (_generate_tts_wav) is faked, so these cover route wiring, validation,
gallery persistence and the raw-WAV response without torch, weights or a GPU."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery_module
from core.inference.api_monitor import api_monitor
import routes.inference as routes_module
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers

_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-payload"


def _make_client(monkeypatch, generate = None):
    calls = []

    async def _fake_generate(text, payload, request, current_subject):
        calls.append({"text": text, "payload": payload})
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


def test_non_wav_response_format_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": "mp3"})
    assert resp.status_code == 400
    assert "mp3" in resp.json()["error"]["message"]
    assert calls == []  # rejected before any generation


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


def test_the_shared_core_guards_before_generating():
    """Wired in _generate_tts_wav so /audio/generate inherits it, not only /audio/speech."""
    import inspect

    source = inspect.getsource(routes_module._generate_tts_wav)
    assert "_raise_if_prompt_leaves_no_speech_budget(text)" in source


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
        if "_raise_if_prompt_leaves_no_speech_budget(text)" in line
    ]
    restore = next(
        i
        for i, line in enumerate(source.splitlines())
        if "_maybe_auto_switch_model(_RELOAD_ONLY_MODEL" in line
    )
    assert len(guards) == 2, "one check before the restore, one after"
    assert guards[0] < restore < guards[1]


def test_the_gallery_is_bounded_so_an_api_client_cannot_fill_the_disk(monkeypatch, tmp_path):
    import core.inference.audio_gallery as gallery

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
    import core.inference.audio_gallery as gallery

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
    import core.inference.audio_gallery as gallery

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


def test_non_latin_prompts_are_not_billed_at_the_latin_rate():
    """Without a Python tokenizer (GGUF TTS), the estimate is by character class. Cutting
    at U+2E7F caught CJK and emoji but billed Arabic, Cyrillic, Hebrew and the Indic
    scripts at a third of a token each, so a long prompt in any of them passed the guard
    and then overflowed the loaded context during generation."""
    estimate = routes_module._prompt_token_estimate

    for label, text in (
        ("arabic", "مرحبا بالعالم " * 40),
        ("cyrillic", "Привет мир " * 40),
        ("hebrew", "שלום עולם " * 40),
        ("devanagari", "नमस्ते दुनिया " * 40),
    ):
        assert estimate(text) >= len(text.replace(" ", "")), label

    # Latin is still counted at the cheaper rate, so ordinary English is not rejected.
    english = "the quick brown fox jumps over the lazy dog " * 40
    assert estimate(english) < len(english) // 2

    # And CJK, which already worked, is unchanged.
    assert estimate("你好世界" * 50) >= 200


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
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/speech"
    assert rows[0]["status"] == "completed"
    assert rows[0]["model"] == "kokoro"
    assert rows[0]["prompt_preview"] == "hi"


def test_external_passes_response_format_through(monkeypatch):
    # The wav-only rule is a local-codec limit; an external server may emit mp3.
    cli, calls, saved = _make_client(monkeypatch)
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
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert speech_calls[0]["response_format"] == "mp3"


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
    import asyncio

    from models.inference import AudioSpeechRequest

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
    from models.inference import AudioSpeechRequest

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

    import asyncio

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
    import asyncio

    from core.inference.external_provider import ExternalProviderClient
    import core.inference.external_provider as provider_module

    sent = {}

    class _Response:
        content = b"audio"
        headers = {"content-type": "audio/wav"}

        def raise_for_status(self):
            return None

    async def _post(url, **_kwargs):
        sent["url"] = url
        return _Response()

    monkeypatch.setattr(provider_module._http_client, "post", _post)
    client = ExternalProviderClient(
        "custom",
        "http://127.0.0.1:8880/v1?api-version=2026-08-24",
        "sk-test",
    )
    asyncio.run(client.create_speech(text = "hi", model = "kokoro"))
    assert sent["url"] == ("http://127.0.0.1:8880/v1/audio/speech?api-version=2026-08-24")


def test_external_provider_reads_do_not_block_the_event_loop(monkeypatch):
    import asyncio
    import threading

    from models.inference import AudioSpeechRequest

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
    import asyncio

    from models.inference import AudioSpeechRequest

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


def test_tts_failure_records_an_error_row(monkeypatch):
    async def _boom(text):
        raise HTTPException(status_code = 400, detail = "No model loaded.")

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
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": "mp3"})
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

    async def _boom(text):
        raise HTTPException(status_code = 400, detail = "No model loaded.")

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
    async def _boom(text):
        raise HTTPException(status_code = 400, detail = "No model loaded.")

    cli, calls, saved = _make_client(monkeypatch, generate = _boom)
    for requested in ("tts-1", "gpt-4o-mini-tts", "unsloth/orpheus-3b-0.1-ft"):
        api_monitor.clear()
        assert (
            cli.post("/v1/audio/speech", json = {"input": "hi", "model": requested}).status_code
            == 400
        )
        assert api_monitor.snapshot(include_details = False)[0]["model"] == requested
