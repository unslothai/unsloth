# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Round-trip tests for POST /v1/audio/transcriptions with _transcribe_audio_result faked; no GPU."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.inference as routes_module
from core.inference.api_monitor import api_monitor
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers


def _make_client(monkeypatch, transcribe = None):
    calls = []

    async def _fake_transcribe(
        raw,
        model,
        language,
        fast,
        engine = None,
        request = None,
        device = None,
        **kwargs,
    ):
        calls.append(
            {
                "raw": raw,
                "model": model,
                "language": language,
                "fast": fast,
                "engine": engine,
                "request": request,
                **kwargs,
            }
        )
        if transcribe is not None:
            return await transcribe(raw)
        return {"text": "hello sloth", "language": "en", "duration": 1.2, "model": "small"}

    monkeypatch.setattr(routes_module, "_transcribe_audio_result", _fake_transcribe)

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), calls


def _post(
    cli,
    data = None,
    filename = "clip.wav",
    content = b"RIFFfake",
    content_type = "audio/wav",
):
    return cli.post(
        "/v1/audio/transcriptions",
        files = {"file": (filename, content, content_type)},
        data = data or {},
    )


def test_json_response_is_text_only(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli)
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello sloth"}
    assert calls[0]["raw"] == b"RIFFfake"
    assert calls[0]["fast"] is False
    assert calls[0]["request"] is not None


def test_text_response_is_plain_body(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "text"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello sloth"


def test_whisper1_and_missing_model_map_to_sidecar_default(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    assert _post(cli, data = {"model": "whisper-1"}).status_code == 200
    assert _post(cli).status_code == 200
    assert [c["model"] for c in calls] == [None, None]


def test_explicit_model_passes_through(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"model": "large-v3-turbo", "language": "de"})
    assert resp.status_code == 200
    assert calls[0]["model"] == "large-v3-turbo"
    assert calls[0]["language"] == "de"


def test_unknown_response_format_is_400(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "srt"})
    assert resp.status_code == 400
    assert "srt" in resp.json()["error"]["message"]
    assert calls == []


def test_missing_file_is_rejected(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/transcriptions", data = {"model": "whisper-1"})
    assert resp.status_code == 400
    assert calls == []


def test_sidecar_errors_keep_their_status(monkeypatch):
    # Error mapping lives in _transcribe_audio_result; the route must not swallow or rewrap it.
    async def _bad_model(raw):
        raise HTTPException(status_code = 422, detail = "Unknown STT model id.")

    cli, calls = _make_client(monkeypatch, transcribe = _bad_model)
    resp = _post(cli, data = {"model": "not-a-model"})
    assert resp.status_code == 422
    assert "Unknown STT model id." in resp.json()["error"]["message"]


def test_an_mtmd_only_model_forces_its_engine():
    """Qwen3-ASR runs only on the mtmd sidecar, so its engine is forced, not defaulted to Transformers."""
    from routes.inference import _stt_engine_for_model

    assert _stt_engine_for_model("qwen3-asr-0.6b") == "mtmd"
    assert _stt_engine_for_model("qwen3-asr-1.7b") == "mtmd"


def test_whisper_ids_keep_the_default_engine():
    """Whisper ids are shared with the Transformers sidecar, so nothing is forced."""
    from routes.inference import _stt_engine_for_model
    for model in (None, "", "whisper-1", "small", "large-v3-turbo", "openai/whisper-tiny"):
        assert _stt_engine_for_model(model) is None, model


def test_the_studio_json_route_also_forwards_the_request(monkeypatch):
    """The raw and OpenAI routes always passed the request; the base64 JSON route did not,
    so a client that goes away left the sidecar transcribing under its lock."""
    import base64

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from routes.inference import studio_router

    cli, calls = _make_client(monkeypatch)
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(studio_router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    cli = TestClient(app)
    resp = cli.post(
        "/audio/transcribe",
        json = {"audio": base64.b64encode(b"RIFFfake").decode()},
    )
    assert resp.status_code == 200
    assert calls[0]["raw"] == b"RIFFfake"
    assert calls[0]["request"] is not None


def test_verbose_json_carries_language_and_duration(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "verbose_json", "language": "en"})
    assert resp.status_code == 200
    assert resp.json() == {
        "task": "transcribe",
        "language": "en",
        "duration": 1.2,
        "text": "hello sloth",
    }


def test_verbose_json_without_a_language_is_refused_before_any_work(monkeypatch):
    """Refuse verbose_json without a language: naming one nobody detected would mislabel the clip."""
    cli, calls = _make_client(monkeypatch)
    api_monitor.clear()
    resp = _post(cli, data = {"response_format": "verbose_json"})
    assert resp.status_code == 501
    assert "language" in resp.json()["error"]["message"]
    assert calls == []
    assert api_monitor.snapshot(include_details = False) == []


def test_verbose_json_works_when_the_caller_supplies_a_language(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "verbose_json", "language": "en"})
    assert resp.status_code == 200
    assert resp.json()["language"] == "en"


def test_verbose_json_never_emits_a_null_duration(monkeypatch):
    """A clip that decodes to no samples has no duration; OpenAI requires a number.

    Unlike the language this is not a guess: such a clip really is zero seconds."""

    async def _empty(raw):
        return {"text": "", "language": "en", "duration": None, "model": "small"}

    cli, calls = _make_client(monkeypatch, transcribe = _empty)
    resp = _post(cli, data = {"response_format": "verbose_json", "language": "en"})
    assert resp.json()["duration"] == 0.0


def test_timestamp_granularities_are_refused_not_dropped(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(
        cli,
        data = {
            "response_format": "verbose_json",
            "language": "en",
            "timestamp_granularities[]": "word",
        },
    )
    assert resp.status_code == 400
    assert "timestamp_granularities" in resp.json()["error"]["message"]
    assert calls == []


@pytest.mark.parametrize(
    "result",
    [
        {"text": "hi", "language": "en", "duration": 1.2, "model": "small"},
        {"text": "", "language": "en", "duration": None, "model": "small"},
        {"text": "hi", "language": "fr", "duration": 3, "model": "small"},
    ],
)
def test_verbose_json_validates_against_the_openai_client_model(monkeypatch, result):
    """The response has to survive the schema the official client parses it with."""
    openai_types = pytest.importorskip("openai.types.audio.transcription_verbose")

    async def _result(raw):
        return dict(result)

    cli, calls = _make_client(monkeypatch, transcribe = _result)
    resp = _post(cli, data = {"response_format": "verbose_json", "language": "en"})
    assert resp.status_code == 200
    openai_types.TranscriptionVerbose.model_validate(resp.json())


def test_subtitle_formats_are_still_400(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    for fmt in ("srt", "vtt"):
        assert _post(cli, data = {"response_format": fmt}).status_code == 400


def test_transcription_opens_a_monitor_row(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    api_monitor.clear()
    assert _post(cli, filename = "meeting.wav").status_code == 200
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/transcriptions"
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_preview"] == "meeting.wav"
    assert rows[0]["reply_preview"] == "hello sloth"
    assert rows[0]["model"] == "small"


def test_sidecar_failure_records_an_error_row(monkeypatch):
    async def _boom(raw):
        raise HTTPException(status_code = 409, detail = "Model is busy.")

    cli, calls = _make_client(monkeypatch, transcribe = _boom)
    api_monitor.clear()
    assert _post(cli).status_code == 409
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "Model is busy."


def test_client_abort_records_a_cancelled_row(monkeypatch):
    # SttTranscriptionCancelledError surfaces as a 499, so the row is a cancellation.
    async def _cancelled(raw):
        raise HTTPException(status_code = 499, detail = "Transcription cancelled")

    cli, calls = _make_client(monkeypatch, transcribe = _cancelled)
    api_monitor.clear()
    assert _post(cli).status_code == 499
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "cancelled"
    assert not rows[0]["error"]


@pytest.mark.parametrize(
    "detail",
    [
        {"error": "bad"},
        {"error": ["bad"]},
        {"error": None},
        {"message": "bad"},
        {"error": {"message": "nested"}},
    ],
)
def test_a_dict_detail_never_strands_the_row(monkeypatch, detail):
    """Only openai_error_body's shape nests the message. For any other dict the handler
    called .get() on a non-dict and raised AttributeError out of the context manager,
    which skipped finish() and left the row at "running" forever."""

    async def _boom(raw):
        raise HTTPException(status_code = 400, detail = detail)

    cli, calls = _make_client(monkeypatch, transcribe = _boom)
    api_monitor.clear()
    assert _post(cli).status_code == 400
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_a_baseexception_still_closes_the_row(monkeypatch, exc):
    """KeyboardInterrupt and SystemExit are not Exception, so they used to fall past
    every handler and leave the row stuck at "running" for the life of the process."""

    async def _boom(raw):
        raise exc("bang")

    cli, calls = _make_client(monkeypatch, transcribe = _boom)
    api_monitor.clear()
    with pytest.raises(BaseException):
        _post(cli)
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"]


def test_a_real_cancellederror_records_a_cancelled_row(monkeypatch):
    async def _cancelled(raw):
        raise asyncio.CancelledError()

    cli, calls = _make_client(monkeypatch, transcribe = _cancelled)
    api_monitor.clear()
    with pytest.raises(BaseException):
        _post(cli)
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "cancelled"


def test_a_non_http_failure_records_a_friendly_error_row(monkeypatch):
    async def _boom(raw):
        raise RuntimeError("sidecar exploded")

    cli, calls = _make_client(monkeypatch, transcribe = _boom)
    api_monitor.clear()
    with pytest.raises(RuntimeError):
        _post(cli)
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "sidecar exploded" not in rows[0]["error"]


def test_the_monitor_label_never_carries_a_local_path(monkeypatch):
    async def _pathy(raw):
        return {
            "text": "t",
            "language": "en",
            "duration": 1.0,
            "model": "/home/me/models/whisper-large-v3",
        }

    cli, calls = _make_client(monkeypatch, transcribe = _pathy)
    api_monitor.clear()
    assert _post(cli).status_code == 200
    row = api_monitor.snapshot(include_details = False)[0]
    assert "/" not in row["model"]
    assert row["model"] == "whisper-large-v3"


def test_skip_api_monitor_suppresses_the_row(monkeypatch):
    """Internal workflows set the flag; the media routes must honour it like the
    text routes do, or an internal step shows up as user API traffic."""
    cli, calls = _make_client(monkeypatch)

    @cli.app.middleware("http")
    async def _skip(request, call_next):
        request.state.skip_api_monitor = True
        return await call_next(request)

    api_monitor.clear()
    assert _post(TestClient(cli.app)).status_code == 200
    assert api_monitor.snapshot(include_details = False) == []


def _install_external(
    monkeypatch,
    *,
    enabled = True,
    media_type = "application/json",
):
    client_args = []
    transcription_calls = []
    credential_calls = []
    config = {
        "provider_type": "custom",
        "display_name": "Whisper Box",
        "base_url": "http://stt.local:8000/v1",
        "is_enabled": enabled,
    }

    monkeypatch.setattr(
        routes_module.providers_db,
        "get_provider",
        lambda provider_id: dict(config) if provider_id == "conn-1" else None,
    )
    monkeypatch.setattr(routes_module, "validate_provider_base_url", lambda url: url)

    def _resolve_api_key(
        provider_id,
        encrypted_api_key,
        *,
        allow_saved_key = True,
    ):
        credential_calls.append(
            {
                "provider_id": provider_id,
                "encrypted_api_key": encrypted_api_key,
                "allow_saved_key": allow_saved_key,
            }
        )
        return "sk-test" if allow_saved_key else ""

    monkeypatch.setattr(routes_module, "resolve_provider_api_key_or_400", _resolve_api_key)

    class _FakeClient:
        def __init__(self, provider_type, base_url, api_key):
            client_args.append(
                {
                    "provider_type": provider_type,
                    "base_url": base_url,
                    "api_key": api_key,
                }
            )

        async def create_transcription(self, **kwargs):
            transcription_calls.append(kwargs)
            body = b"remote words" if media_type == "text/plain" else b'{"text":"remote words"}'
            return body, media_type

    monkeypatch.setattr(routes_module, "ExternalProviderClient", _FakeClient)
    return client_args, transcription_calls, credential_calls


def test_provider_id_routes_to_external_endpoint_without_loading_the_sidecar(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    client_args, transcription_calls, credential_calls = _install_external(monkeypatch)
    resp = _post(
        cli,
        data = {
            "provider_id": "conn-1",
            "model": "Systran/faster-distil-whisper-large-v3",
            "language": "en",
        },
        filename = "dictation.webm",
        content = b"webm-audio",
        content_type = "audio/webm",
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "remote words"}
    assert sidecar_calls == []
    assert client_args == [
        {
            "provider_type": "custom",
            "base_url": "http://stt.local:8000/v1",
            "api_key": "sk-test",
        }
    ]
    assert transcription_calls == [
        {
            "audio": b"webm-audio",
            "filename": "dictation.webm",
            "content_type": "audio/webm",
            "model": "Systran/faster-distil-whisper-large-v3",
            "language": "en",
            "response_format": "json",
            "timestamp_granularities": None,
        }
    ]
    assert credential_calls[0]["allow_saved_key"] is True


def test_external_text_response_keeps_plain_text_shape(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch, media_type = "text/plain")
    resp = _post(
        cli,
        data = {
            "provider_id": "conn-1",
            "model": "whisper-1",
            "response_format": "text",
        },
    )

    assert resp.status_code == 200
    assert resp.text == "remote words"
    assert resp.headers["content-type"].startswith("text/plain")
    assert sidecar_calls == []


def test_external_connection_requires_a_model(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    client_args, _, _ = _install_external(monkeypatch)
    resp = _post(cli, data = {"provider_id": "conn-1"})

    assert resp.status_code == 400
    assert "model is required" in resp.json()["error"]["message"]
    assert client_args == []
    assert sidecar_calls == []


@pytest.mark.parametrize(
    ("provider_id", "enabled", "status"),
    [("missing", True, 404), ("conn-1", False, 400)],
)
def test_external_connection_must_exist_and_be_enabled(monkeypatch, provider_id, enabled, status):
    cli, sidecar_calls = _make_client(monkeypatch)
    client_args, _, _ = _install_external(monkeypatch, enabled = enabled)
    resp = _post(
        cli,
        data = {"provider_id": provider_id, "model": "whisper-1"},
    )

    assert resp.status_code == status
    assert client_args == []
    assert sidecar_calls == []


def test_external_connection_validates_the_url_before_reading_its_key(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _, _, credential_calls = _install_external(monkeypatch)

    def _reject_url(_url):
        raise ValueError("refused target")

    monkeypatch.setattr(routes_module, "validate_provider_base_url", _reject_url)
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "whisper-1"},
    )

    assert resp.status_code == 400
    assert credential_calls == []
    assert sidecar_calls == []


def test_api_key_callers_cannot_spend_a_saved_external_stt_key(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    client_args, _, credential_calls = _install_external(monkeypatch)
    resp = cli.post(
        "/v1/audio/transcriptions",
        files = {"file": ("clip.wav", b"RIFFfake", "audio/wav")},
        data = {"provider_id": "conn-1", "model": "whisper-1"},
        headers = {"Authorization": "Bearer sk-unsloth-test"},
    )

    assert resp.status_code == 200
    assert credential_calls[0]["allow_saved_key"] is False
    assert client_args[0]["api_key"] == ""
    assert sidecar_calls == []


def test_external_connection_accepts_a_legacy_encrypted_key(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _, _, credential_calls = _install_external(monkeypatch)
    resp = _post(
        cli,
        data = {
            "provider_id": "conn-1",
            "model": "whisper-1",
            "encrypted_api_key": "sealed-key",
        },
    )

    assert resp.status_code == 200
    assert credential_calls[0]["encrypted_api_key"] == "sealed-key"
    assert sidecar_calls == []


def test_external_upstream_errors_are_502(monkeypatch):
    import httpx

    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch)

    async def _reject(self, **kwargs):
        request = httpx.Request("POST", "http://stt.local:8000/v1/audio/transcriptions")
        response = httpx.Response(503, text = "not ready", request = request)
        raise httpx.HTTPStatusError("rejected", request = request, response = response)

    monkeypatch.setattr(routes_module.ExternalProviderClient, "create_transcription", _reject)
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "whisper-1"},
    )

    assert resp.status_code == 502
    assert "HTTP 503" in resp.json()["error"]["message"]
    assert sidecar_calls == []


def test_external_disconnect_cancels_the_upstream_request(monkeypatch):
    import asyncio

    _install_external(monkeypatch)
    upstream_cancelled = asyncio.Event()

    class _DisconnectingRequest:
        headers = {}

        async def is_disconnected(self):
            return True

    class _BlockingClient:
        def __init__(self, **_kwargs):
            pass

        async def create_transcription(self, **_kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise

    monkeypatch.setattr(routes_module, "ExternalProviderClient", _BlockingClient)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await routes_module._external_stt_transcription(
                provider_id = "conn-1",
                raw = b"RIFFfake",
                filename = "clip.wav",
                content_type = "audio/wav",
                model = "whisper-1",
                language = None,
                response_format = "json",
                encrypted_api_key = None,
                request = _DisconnectingRequest(),
            )

    asyncio.run(_run())
    assert upstream_cancelled.is_set()


def test_external_client_sends_openai_compatible_multipart(monkeypatch):
    import asyncio
    import httpx

    import core.inference.external_provider as provider_module

    captured = {}

    class _HttpClient:
        async def post(self, url, **kwargs):
            captured.update(url = url, **kwargs)
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                content = b'{"text":"hello"}',
                headers = {"content-type": "application/json; charset=utf-8"},
                request = request,
            )

    monkeypatch.setattr(provider_module, "_http_client", _HttpClient())
    client = provider_module.ExternalProviderClient(
        provider_type = "custom",
        base_url = "https://stt.example.com/v1",
        api_key = "sk-test",
    )
    body, media_type = asyncio.run(
        client.create_transcription(
            audio = b"webm-audio",
            filename = "dictation.webm",
            content_type = "audio/webm",
            model = "whisper-1",
            language = "en",
        )
    )

    assert body == b'{"text":"hello"}'
    assert media_type == "application/json"
    assert captured["url"] == "https://stt.example.com/v1/audio/transcriptions"
    assert "Content-Type" not in captured["headers"]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["files"] == {"file": ("dictation.webm", b"webm-audio", "audio/webm")}
    assert captured["data"] == {
        "model": "whisper-1",
        "response_format": "json",
        "language": "en",
    }


def test_verbose_json_is_forwarded_to_the_provider_verbatim(monkeypatch):
    """The proxied arm returns the provider's own verbose_json, segments and all, so
    the format has to reach it and the body must come back untouched."""
    cli, sidecar_calls = _make_client(monkeypatch)
    provider_body = (
        b'{"task":"transcribe","language":"en","duration":1.5,'
        b'"text":"remote words","segments":[{"id":0,"text":"remote words"}]}'
    )

    class _FakeClient:
        def __init__(self, provider_type, base_url, api_key):
            pass

        async def create_transcription(self, **kwargs):
            sidecar_calls.append(kwargs)
            return provider_body, "application/json"

    _install_external(monkeypatch)
    monkeypatch.setattr(routes_module, "ExternalProviderClient", _FakeClient)
    api_monitor.clear()
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "whisper-1", "response_format": "verbose_json"},
    )
    assert resp.status_code == 200
    assert sidecar_calls[-1]["response_format"] == "verbose_json"
    assert resp.json()["segments"] == [{"id": 0, "text": "remote words"}]
    assert api_monitor.snapshot(include_details = False)[0]["reply_preview"] == "remote words"


def test_external_transcription_opens_a_monitor_row(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch)
    api_monitor.clear()
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "Systran/faster-distil-whisper-large-v3"},
        filename = "dictation.webm",
    )
    assert resp.status_code == 200
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/transcriptions"
    assert rows[0]["status"] == "completed"
    assert rows[0]["model"] == "Systran/faster-distil-whisper-large-v3"
    assert rows[0]["prompt_preview"] == "dictation.webm"
    assert rows[0]["reply_preview"] == "remote words"


def test_external_transcription_reply_preview_for_plain_text(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch, media_type = "text/plain")
    api_monitor.clear()
    resp = _post(
        cli,
        data = {
            "provider_id": "conn-1",
            "model": "Systran/faster-distil-whisper-large-v3",
            "response_format": "text",
        },
    )
    assert resp.status_code == 200
    assert api_monitor.snapshot(include_details = False)[0]["reply_preview"] == "remote words"


def test_external_transcription_failure_records_an_error_row(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch, enabled = False)
    api_monitor.clear()
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "Systran/faster-distil-whisper-large-v3"},
    )
    assert resp.status_code >= 400
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"]


def test_external_reply_preview_handles_an_uppercase_json_media_type(monkeypatch):
    cli, sidecar_calls = _make_client(monkeypatch)
    _install_external(monkeypatch, media_type = "Application/JSON")
    api_monitor.clear()
    resp = _post(
        cli,
        data = {"provider_id": "conn-1", "model": "Systran/faster-distil-whisper-large-v3"},
    )
    assert resp.status_code == 200
    assert api_monitor.snapshot(include_details = False)[0]["reply_preview"] == "remote words"


def test_timestamp_granularities_reach_a_capable_provider(monkeypatch):
    # A saved connection may produce timings, so the proxied arm forwards the parameter.
    cli, sidecar_calls = _make_client(monkeypatch)
    _client_args, transcription_calls, _creds = _install_external(monkeypatch)
    resp = _post(
        cli,
        data = {
            "provider_id": "conn-1",
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
    )
    assert resp.status_code == 200
    assert transcription_calls[-1]["timestamp_granularities"] == ["word", "segment"]


def test_the_provider_client_sends_granularities_as_a_repeated_field(monkeypatch):
    from core.inference.external_provider import ExternalProviderClient

    sent = {}

    class _Resp:
        status_code = 200
        content = b'{"text":"x"}'
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

    async def _post_capture(url, **kwargs):
        sent.update(kwargs)
        return _Resp()

    import core.inference.external_provider as ep

    monkeypatch.setattr(ep._http_client, "post", _post_capture)
    client = ExternalProviderClient("custom", "http://stt.local/v1", "sk-test")
    import asyncio as _asyncio

    _asyncio.run(
        client.create_transcription(
            audio = b"x",
            filename = "a.wav",
            content_type = "audio/wav",
            model = "whisper-1",
            timestamp_granularities = ["word"],
        )
    )
    assert sent["data"]["timestamp_granularities[]"] == ["word"]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("/home/ana/models/whisper-large-v3", "whisper-large-v3"),
        (r"C:\Users\ana\models\whisper-large-v3", "whisper-large-v3"),
        (r"\\fileserver\share\models\whisper-large-v3", "whisper-large-v3"),
    ],
)
def test_a_sidecar_failure_does_not_leak_the_requested_path(monkeypatch, requested, expected):
    """The relabel only lands on success, so a sidecar failure kept the raw client string
    on the terminal row. Windows and UNC forms are covered because os.path.basename alone
    would not split either one on a Linux host."""

    async def _boom(raw):
        raise HTTPException(status_code = 409, detail = "Model is busy.")

    cli, calls = _make_client(monkeypatch, transcribe = _boom)
    api_monitor.clear()
    assert _post(cli, data = {"model": requested}).status_code == 409
    row = api_monitor.snapshot(include_details = False)[0]
    assert row["status"] == "error"
    assert row["model"] == expected
    assert "/" not in row["model"] and "\\" not in row["model"]
    assert calls[0]["model"] == requested


def test_the_proxied_row_never_carries_a_local_path(monkeypatch):
    """The proxied arm never relabels at all, so whatever it opens with is what the row
    keeps for its whole life, success included."""
    cli, _sidecar_calls = _make_client(monkeypatch)
    _client_args, transcription_calls, _creds = _install_external(monkeypatch)
    api_monitor.clear()
    resp = _post(cli, data = {"provider_id": "conn-1", "model": "/home/ana/models/whisper-v3"})
    assert resp.status_code == 200
    row = api_monitor.snapshot(include_details = False)[0]
    assert row["model"] == "whisper-v3"
    assert transcription_calls[-1]["model"] == "/home/ana/models/whisper-v3"


_TIMED = {
    "text": "Hi there. Bye.",
    "language": "English",
    "duration": 2.5,
    "model": "audio-cpp/audio.cpp-gguf/VibeVoice-ASR-GGUF",
    "segments": [
        {"start": 0.0, "end": 1.0, "text": "Hi there.", "speaker": "0"},
        {"start": 1.2, "end": 2.5, "text": "Bye.", "speaker": "1"},
    ],
    "words": [
        {"start": 0.0, "end": 0.4, "word": "Hi"},
        {"start": 0.4, "end": 1.0, "word": "there."},
    ],
    "speakers": [{"id": "0", "label": "Speaker 1"}, {"id": "1", "label": "Speaker 2"}],
}


def _audio_cpp(
    monkeypatch,
    tmp_path,
    result = _TIMED,
    **caps,
):
    """An audio.cpp ASR model; the upload's prepared copy is a file the route must remove."""
    from core.inference import stt_capabilities

    async def _result(raw):
        return dict(result)

    cli, calls = _make_client(monkeypatch, transcribe = _result)
    monkeypatch.setattr(routes_module, "_stt_engine_for_model", lambda model: "audiocpp")
    monkeypatch.setattr(routes_module, "_resolve_serving_stt_engine", lambda engine: engine)
    caps = {"family": "vibevoice_asr", "timestamps": "always", "speakers": True, **caps}
    monkeypatch.setattr(stt_capabilities, "capabilities_for", lambda model, engine: caps)
    prepared = []

    def _prepare(raw, rate):
        path = tmp_path / f"upload.{rate}.wav"
        path.write_bytes(raw)
        prepared.append(path)
        return path

    monkeypatch.setattr(routes_module, "_prepared_upload", _prepare)
    return cli, calls, prepared


def test_audio_cpp_verbose_json_has_openai_segments_and_words(monkeypatch, tmp_path):
    openai_types = pytest.importorskip("openai.types.audio.transcription_verbose")
    cli, calls, prepared = _audio_cpp(monkeypatch, tmp_path)
    data = {"response_format": "verbose_json", "timestamp_granularities[]": ["segment", "word"]}
    resp = _post(cli, data = data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    openai_types.TranscriptionVerbose.model_validate(body)
    assert body["language"] == "English"
    assert [(s["id"], s["start"], s["end"], s["text"]) for s in body["segments"]] == [
        (0, 0.0, 1.0, "Hi there."),
        (1, 1.2, 2.5, "Bye."),
    ]
    assert body["words"] == [
        {"word": "Hi", "start": 0.0, "end": 0.4},
        {"word": "there.", "start": 0.4, "end": 1.0},
    ]
    # VibeVoice times its spans at 24 kHz, so the upload is prepared at that rate and removed.
    (path,) = prepared
    assert path.name == "upload.24000.wav" and not path.exists()
    assert calls[0]["source_path"] == path and calls[0]["timestamps"] is True


def test_audio_cpp_diarized_json_names_speakers(monkeypatch, tmp_path):
    cli, calls, prepared = _audio_cpp(monkeypatch, tmp_path)
    resp = _post(cli, data = {"response_format": "diarized_json"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["task"], body["duration"], body["text"]) == ("transcribe", 2.5, "Hi there. Bye.")
    assert body["segments"] == [
        {
            "type": "transcript.text.segment",
            "id": "seg_0",
            "start": 0.0,
            "end": 1.0,
            "text": "Hi there.",
            "speaker": "Speaker 1",
        },
        {
            "type": "transcript.text.segment",
            "id": "seg_1",
            "start": 1.2,
            "end": 2.5,
            "text": "Bye.",
            "speaker": "Speaker 2",
        },
    ]
    assert calls[0]["timestamps"] is False


@pytest.mark.parametrize(
    "caps, data, detail",
    [
        ({"speakers": False}, {"response_format": "diarized_json"}, "cannot tell speakers apart"),
        (
            {"timestamps": "unsupported"},
            {"response_format": "verbose_json", "timestamp_granularities[]": "word"},
            "cannot add timestamps",
        ),
    ],
)
def test_audio_cpp_refuses_what_the_model_cannot_add(monkeypatch, tmp_path, caps, data, detail):
    cli, calls, prepared = _audio_cpp(monkeypatch, tmp_path, **caps)
    resp = _post(cli, data = data)
    assert resp.status_code == 422
    assert detail in resp.json()["error"]["message"]
    assert calls == [] and prepared == []


@pytest.mark.parametrize("fmt", ["json", "text", "diarized_json"])
def test_audio_cpp_granularities_need_verbose_json(monkeypatch, tmp_path, fmt):
    cli, calls, prepared = _audio_cpp(monkeypatch, tmp_path)
    resp = _post(cli, data = {"response_format": fmt, "timestamp_granularities[]": "word"})
    assert (resp.status_code, resp.json()["error"]["param"]) == (400, "timestamp_granularities")
    assert calls == [] and prepared == []


def test_diarized_json_on_whisper_is_refused_before_any_work(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "diarized_json"})
    assert resp.status_code == 422
    assert calls == []


@pytest.mark.parametrize("reported, status", [("English", 200), (None, 501)])
def test_audio_cpp_verbose_json_needs_no_language_when_the_model_reports_one(
    monkeypatch, tmp_path, reported, status
):
    cli, calls, prepared = _audio_cpp(
        monkeypatch, tmp_path, result = {**_TIMED, "language": reported}
    )
    resp = _post(cli, data = {"response_format": "verbose_json"})
    assert resp.status_code == status, resp.text
    if status == 200:
        assert resp.json()["language"] == "English"
