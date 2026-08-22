# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/transcriptions.

The sidecar call (_transcribe_audio_result) is faked, so these cover multipart wiring,
model-id mapping, response formats and error propagation without whisper or a GPU."""

from __future__ import annotations

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
    ):
        calls.append(
            {
                "raw": raw,
                "model": model,
                "language": language,
                "fast": fast,
                "engine": engine,
                "request": request,
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
    # OpenAI's json shape carries only the text; the sidecar's extra fields stay internal.
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
    # install_api_error_handlers maps validation errors to a 400 OpenAI envelope on /v1.
    cli, calls = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/transcriptions", data = {"model": "whisper-1"})
    assert resp.status_code == 400
    assert calls == []


def test_sidecar_errors_keep_their_status(monkeypatch):
    # The shared error mapping (SttModelIdError -> 422, empty audio -> 400, ...) sits inside
    # _transcribe_audio_result; the route must not swallow or rewrap what it raises.
    async def _bad_model(raw):
        raise HTTPException(status_code = 422, detail = "Unknown STT model id.")

    cli, calls = _make_client(monkeypatch, transcribe = _bad_model)
    resp = _post(cli, data = {"model": "not-a-model"})
    assert resp.status_code == 422
    assert "Unknown STT model id." in resp.json()["error"]["message"]


def test_an_mtmd_only_model_forces_its_engine():
    """Qwen3-ASR only runs on the mtmd sidecar.

    The route passed no engine, so _resolve_stt_engine defaulted to Transformers and the
    Whisper sidecar rejected the model.
    """
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
    resp = _post(cli, data = {"response_format": "verbose_json"})
    assert resp.status_code == 200
    assert resp.json() == {
        "task": "transcribe",
        "language": "en",
        "duration": 1.2,
        "text": "hello sloth",
    }


def test_verbose_json_language_is_null_when_none_was_requested(monkeypatch):
    async def _no_language(raw):
        return {"text": "hola", "language": None, "duration": 2.0, "model": "small"}

    cli, calls = _make_client(monkeypatch, transcribe = _no_language)
    resp = _post(cli, data = {"response_format": "verbose_json"})
    assert resp.json()["language"] is None


def test_subtitle_formats_are_still_400(monkeypatch):
    # srt/vtt need per-segment timing the sidecar does not report yet.
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
    # A disabled connection is rejected before the proxy call; the row still closes.
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
