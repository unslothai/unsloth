# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http.server
import io
import json
import threading
import wave

import numpy as np
import pytest

import core.inference.stt_ggml_sidecar as ggml_module
from core.inference.stt_ggml_sidecar import (
    DEFAULT_GGML_STT_MODEL,
    GGML_STT_MODELS,
    GgmlSttSidecar,
    SttEngineUnavailableError,
    find_whisper_server_binary,
    resolve_ggml_model_id,
)
from core.inference.stt_sidecar import (
    SttLanguageError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
)


@pytest.fixture(autouse = True)
def stub_audio_decoder(monkeypatch):
    """Unit tests exercise orchestration, not PyAV container parsing."""
    monkeypatch.setattr(
        ggml_module,
        "_decode_audio_bounded",
        lambda audio: np.zeros(16000, dtype = np.float32),
    )


# ---------------------------------------------------------------------------
# Model id resolution
# ---------------------------------------------------------------------------


def test_curated_ids_resolve():
    for model_id in GGML_STT_MODELS:
        assert resolve_ggml_model_id(model_id) == model_id


def test_default_model_resolves_from_none_and_blank():
    assert resolve_ggml_model_id(None) == DEFAULT_GGML_STT_MODEL
    assert resolve_ggml_model_id("  ") == DEFAULT_GGML_STT_MODEL


def test_custom_repo_ids_are_rejected():
    with pytest.raises(SttModelIdError):
        resolve_ggml_model_id("owner/model")
    with pytest.raises(SttModelIdError):
        resolve_ggml_model_id("large-v2")


def test_curated_ids_mirror_transformers_sidecar():
    from core.inference.stt_sidecar import STT_MODELS
    assert list(GGML_STT_MODELS.keys()) == list(STT_MODELS.keys())


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def test_env_binary_override_wins(monkeypatch, tmp_path):
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("WHISPER_SERVER_PATH", str(binary))
    assert find_whisper_server_binary() == str(binary)


def test_env_dir_override_scans_layouts(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_SERVER_PATH", raising = False)
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents = True)
    binary = build_bin / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(tmp_path))
    assert find_whisper_server_binary() == str(binary)


def test_missing_binary_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_SERVER_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr(ggml_module, "_managed_whisper_cpp_dir", lambda: tmp_path / "gone")
    monkeypatch.setattr(ggml_module.shutil, "which", lambda name: None)
    assert find_whisper_server_binary() is None
    assert not ggml_module.is_available()
    with pytest.raises(SttEngineUnavailableError):
        ggml_module.ensure_engine_available()


def test_engine_unavailable_is_stt_unavailable():
    # Routes map SttUnavailableError to HTTP 501; the engine error must share it.
    assert issubclass(SttEngineUnavailableError, SttUnavailableError)


# ---------------------------------------------------------------------------
# WAV packaging
# ---------------------------------------------------------------------------


def test_pcm_to_wav_bytes_shape_and_rate():
    pcm = np.zeros(3200, dtype = np.float32)
    data = ggml_module._pcm_to_wav_bytes(pcm)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 3200


def test_pcm_to_wav_bytes_clips_out_of_range():
    pcm = np.array([2.0, -2.0], dtype = np.float32)
    data = ggml_module._pcm_to_wav_bytes(pcm)
    with wave.open(io.BytesIO(data)) as w:
        frames = np.frombuffer(w.readframes(2), dtype = "<i2")
    assert frames[0] == 32767
    assert frames[1] == -32767


# ---------------------------------------------------------------------------
# Sidecar orchestration
# ---------------------------------------------------------------------------


def _available(monkeypatch):
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: "/bin/echo")


def test_transcribe_requires_engine(monkeypatch):
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: None)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttEngineUnavailableError):
        sidecar.transcribe(b"RIFF")


def test_transcribe_rejects_unknown_language(monkeypatch):
    _available(monkeypatch)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttLanguageError):
        sidecar.transcribe(b"RIFF", model = "small", language = "xx-QQ")


def test_load_requires_downloaded_model(monkeypatch):
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: None)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttModelNotDownloadedError):
        sidecar.load("small")


def test_unloaded_sidecar_reports_nothing_resident():
    sidecar = GgmlSttSidecar()
    assert sidecar.loaded_model is None
    assert sidecar.device is None
    assert sidecar.is_loading() is False
    sidecar.unload()  # no-op, must not raise


class _FakeWhisperHandler(http.server.BaseHTTPRequestHandler):
    """Stands in for whisper-server's /inference endpoint."""

    response_text = "Hello world.\n Second line."

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"text": self.response_text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_whisper_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_transcribe_joins_segments_one_line(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    sidecar = GgmlSttSidecar()

    def fake_load(model = None):
        sidecar._port = fake_whisper_server
        sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

    monkeypatch.setattr(sidecar, "load", fake_load)
    result = sidecar.transcribe(b"RIFF", model = "small", language = "en", fast = True)
    assert result["text"] == "Hello world. Second line."
    assert result["language"] == "en"
    assert result["model"] == "small"
    assert result["duration"] == pytest.approx(1.0)


def test_transcribe_maps_bad_payload_to_decode_error(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    monkeypatch.setattr(_FakeWhisperHandler, "response_text", None)
    sidecar = GgmlSttSidecar()

    def fake_load(model = None):
        sidecar._port = fake_whisper_server
        sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

    monkeypatch.setattr(sidecar, "load", fake_load)
    from core.inference.stt_sidecar import SttAudioDecodeError

    with pytest.raises(SttAudioDecodeError):
        sidecar.transcribe(b"RIFF", model = "small")


def test_beam_size_matches_fast_flag(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    seen: list[bytes] = []

    orig_post = _FakeWhisperHandler.do_POST

    def capture_post(handler):
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        seen.append(body)
        payload = json.dumps({"text": "ok"}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    monkeypatch.setattr(_FakeWhisperHandler, "do_POST", capture_post)
    try:
        sidecar = GgmlSttSidecar()

        def fake_load(model = None):
            sidecar._port = fake_whisper_server
            sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

        monkeypatch.setattr(sidecar, "load", fake_load)
        sidecar.transcribe(b"RIFF", model = "small", fast = True)
        sidecar.transcribe(b"RIFF", model = "small", fast = False)
    finally:
        _FakeWhisperHandler.do_POST = orig_post
    assert b'name="beam_size"\r\n\r\n1' in seen[0]
    assert b'name="beam_size"\r\n\r\n5' in seen[1]
    # Dictation defaults to deterministic decoding.
    assert b'name="temperature"\r\n\r\n0.0' in seen[0]


def test_download_rejects_custom_ids():
    with pytest.raises(SttModelIdError):
        ggml_module.start_model_download("owner/model")


def test_download_status_idle_shape():
    status = ggml_module.download_status()
    assert set(status) >= {"downloading", "model", "error"}
