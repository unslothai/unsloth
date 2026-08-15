# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/speech.

The TTS core (_generate_tts_wav) is faked, so these cover route wiring, validation,
gallery persistence and the raw-WAV response without torch, weights or a GPU."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery_module
import routes.inference as routes_module
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers

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


def test_native_audio_instructions_and_seed_reach_the_shared_core(monkeypatch):
    cli, calls, _saved = _make_client(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {
            "input": "[verse] Morning light",
            "instructions": "Acoustic pop, 96 BPM.",
            "language": "English",
            "seed": 7,
        },
    )

    assert resp.status_code == 200
    payload = calls[0]["payload"]
    assert payload.audio_instructions == "Acoustic pop, 96 BPM."
    assert payload.audio_language == "English"
    assert payload.seed == 7


@pytest.mark.parametrize("seed", (-(2**63) - 1, 2**64))
def test_seed_outside_torch_generator_range_is_rejected(monkeypatch, seed):
    from models.inference import ChatCompletionRequest
    from pydantic import ValidationError

    cli, calls, _saved = _make_client(monkeypatch)

    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "seed": seed})

    assert resp.status_code == 400
    assert calls == []
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages = [{"role": "user", "content": "hi"}],
            seed = seed,
        )


@pytest.mark.parametrize("seed", (-(2**63), 2**64 - 1))
def test_torch_generator_seed_range_endpoints_are_accepted(seed):
    from models.inference import AudioSpeechRequest, ChatCompletionRequest
    assert AudioSpeechRequest(input = "hi", seed = seed).seed == seed
    assert (
        ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}], seed = seed).seed == seed
    )


def test_native_music_frame_limit_reaches_the_shared_core(monkeypatch):
    cli, calls, _saved = _make_client(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {"input": "[verse] Morning light", "max_new_tokens": 500},
    )

    assert resp.status_code == 200
    assert calls[0]["payload"].max_tokens == 500
    assert calls[0]["speech_api_default_max_tokens"] is False


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
    """Speech models retain the full safety ceiling when no frame limit is supplied."""
    from core.inference.orchestrator import AUDIO_GENERATION_MAX_TOKENS

    cli, calls, _saved = _make_client(monkeypatch)
    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: None)
    assert cli.post("/v1/audio/speech", json = {"input": "a long script"}).status_code == 200
    payload = calls[0]["payload"]
    assert payload.max_tokens == AUDIO_GENERATION_MAX_TOKENS
    assert calls[0]["speech_api_default_max_tokens"] is True


def test_minimax_speech_api_default_is_thirty_seconds(monkeypatch):
    from core.inference.orchestrator import AUDIO_GENERATION_MAX_TOKENS
    from models.inference import ChatCompletionRequest

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: None)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "[verse] Morning light"}],
        max_tokens = AUDIO_GENERATION_MAX_TOKENS,
    )

    assert (
        routes_module._tts_max_new_tokens(
            payload,
            audio_type = "minimax_music3",
            speech_api_default_max_tokens = True,
        )
        == 750
    )
    assert (
        routes_module._tts_max_new_tokens(
            payload,
            audio_type = "higgs_tts2",
            speech_api_default_max_tokens = True,
        )
        == AUDIO_GENERATION_MAX_TOKENS
    )


def test_moss_generation_uses_the_model_context_ceiling(monkeypatch):
    from core.inference.orchestrator import MOSS_TTS_MAX_FRAMES
    from models.inference import ChatCompletionRequest

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: None)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "A short sentence."}],
        max_tokens = 10**310,
    )

    for audio_type in ("moss_tts_local", "moss_tts_nano"):
        assert (
            routes_module._tts_max_new_tokens(payload, audio_type = audio_type)
            == MOSS_TTS_MAX_FRAMES
            == 32768
        )

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: MOSS_TTS_MAX_FRAMES)
    prompt = "A short sentence."
    assert routes_module._tts_max_new_tokens(
        payload,
        prompt,
        audio_type = "moss_tts_local",
    ) == (
        MOSS_TTS_MAX_FRAMES
        - routes_module._prompt_token_estimate(prompt)
        - routes_module._TTS_PROMPT_FORMAT_RESERVE
    )


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


def test_moss_prompt_budget_includes_language_and_instructions(monkeypatch):
    prompt = routes_module._native_tts_prompt_for_budget(
        "A short line.",
        "moss_tts_local",
        "Warm and calm.",
        "English",
    )
    assert prompt == "Warm and calm.\nEnglish\nA short line."

    monkeypatch.setattr(routes_module, "_monitor_context_length", lambda: 128)
    with pytest.raises(HTTPException, match = "too long"):
        routes_module._raise_if_prompt_leaves_no_speech_budget(
            routes_module._native_tts_prompt_for_budget(
                "A short line.",
                "moss_tts_local",
                None,
                "language-tag-" * 100,
            )
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
