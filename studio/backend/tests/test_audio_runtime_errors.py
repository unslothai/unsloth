# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request the audio runtime refuses reaches the client with the runtime's reason, sanitized.

CosyVoice3 without a reference clip answered "CosyVoice3 requires reference audio", and the
Audio page showed "An internal error occurred": the reason crossed the worker as a bare
RuntimeError and ``safe_error_detail`` flattened it. Paths and credentials in that text must
still never reach the client.
"""

from __future__ import annotations

import asyncio
import queue
import threading

import pytest

from core.inference.audio_errors import (
    AUDIO_RUNTIME_ERROR_CODE,
    AudioRuntimeError,
    audio_runtime_http_error,
    sanitize_runtime_detail,
)

_LEAKY = (
    "The audio runtime could not generate audio: CosyVoice3 requires reference audio\n"
    "  while loading /home/alice/.unsloth/studio/cache/models--x/blobs/abc123 "
    "(token hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456, C:\\Users\\alice\\voice.wav)"
)


def test_runtime_text_is_one_bounded_line_without_paths_or_tokens():
    detail = sanitize_runtime_detail(_LEAKY)
    assert detail.startswith(
        "The audio runtime could not generate audio: CosyVoice3 requires reference audio"
    )
    assert "\n" not in detail
    assert "/home/alice" not in detail and "alice" not in detail.replace("abc123", "")
    assert "Users" not in detail and "voice.wav" in detail
    assert "hf_AbCdEf" not in detail and "<redacted>" in detail
    long = sanitize_runtime_detail("x " * 1000)
    assert len(long) <= 300 and long.endswith("...")
    assert (
        sanitize_runtime_detail("VibeVoice prompt has no valid Speaker N: lines")
        == "VibeVoice prompt has no valid Speaker N: lines"
    )
    assert sanitize_runtime_detail("see http://127.0.0.1:8080/v1/audio/speech") == (
        "see http://127.0.0.1:8080/v1/audio/speech"
    )


@pytest.mark.parametrize(
    "status, expected",
    [(500, 500), (None, 500), (502, 500), (422, 400), (404, 400), (401, 400), (503, 503)],
)
def test_runtime_status_maps_to_a_status_the_client_reads_as_this_request(status, expected):
    code, detail = audio_runtime_http_error(
        AudioRuntimeError("Vevo2 requires target_voice", status = status)
    )
    assert (code, detail) == (expected, "Vevo2 requires target_voice")


def test_an_empty_runtime_reason_falls_back_to_the_generic_message():
    assert audio_runtime_http_error(AudioRuntimeError("  \n ", status = 500)) == (
        500,
        "An internal error occurred",
    )


def _route_with_backend(monkeypatch, error):
    import routes.inference as inference_route
    from models.inference import ChatCompletionRequest

    class _Llama:
        is_loaded = False
        _is_audio = False

    class _Backend:
        active_model_name = "audio-cpp/audio.cpp-gguf/CosyVoice3-GGUF"
        models = {active_model_name: {"is_audio": True, "audio_type": "audiocpp_tts"}}

        def generate_audio_response(self, **_kwargs):
            raise error

    async def _noop_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = _Backend.active_model_name, messages = [{"role": "user", "content": "hello"}]
    )
    with pytest.raises(inference_route.HTTPException) as excinfo:
        asyncio.run(
            inference_route._generate_tts_wav("hello", payload, request = None, current_subject = "t")
        )
    return excinfo.value


def test_the_generate_and_speech_core_shows_the_runtime_reason(monkeypatch):
    error = _route_with_backend(monkeypatch, AudioRuntimeError(_LEAKY, status = 500))
    assert error.status_code == 500
    assert "CosyVoice3 requires reference audio" in error.detail
    assert "/home/alice" not in error.detail and "hf_AbCdEf" not in error.detail
    refused = _route_with_backend(
        monkeypatch, AudioRuntimeError("unknown MiDashengLM-Gen request option: x", status = 400)
    )
    assert (refused.status_code, refused.detail) == (
        400,
        "unknown MiDashengLM-Gen request option: x",
    )


def test_any_other_failure_still_gets_the_generic_message(monkeypatch):
    error = _route_with_backend(monkeypatch, RuntimeError("boom in /home/alice/secret.py"))
    assert (error.status_code, error.detail) == (500, "An internal error occurred")


def test_the_backend_raises_the_runtime_reason_typed():
    from core.inference import audio_cpp_backend
    from core.inference.audio_cpp_server import AudioCppRequestError

    class _Server:
        model_id = "studio-test"

        def alive(self):
            return True

        def post_json(self, path, payload, **kwargs):
            raise AudioCppRequestError(500, "CosyVoice3 requires reference audio")

    from core.inference.audio_cpp_models import FAMILIES, AudioCppModel, AudioCppVariant, RepoFile

    family = FAMILIES["cosyvoice3"]
    variant = AudioCppVariant("Q8_0", (RepoFile("m.gguf", 1),), "m.gguf")
    model = AudioCppModel(
        id = "x/CosyVoice3-GGUF",
        repo_id = "x/CosyVoice3-GGUF",
        folder = "",
        display_name = "CosyVoice3-GGUF",
        family = family.family,
        task = family.task,
        server_task = family.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
    )
    backend = audio_cpp_backend.AudioCppBackend()
    backend._server = _Server()
    backend._server.model = model
    backend._model = model
    backend.models = {model.id: {"is_audio": True}}
    backend.active_model_name = model.id
    with pytest.raises(AudioRuntimeError) as excinfo:
        backend.generate_audio_response("hello")
    assert excinfo.value.status == 500
    assert str(excinfo.value) == (
        "The audio runtime could not generate audio: CosyVoice3 requires reference audio"
    )


def test_the_worker_tags_a_runtime_error_and_the_parent_rebuilds_it():
    from core.inference.worker import _handle_generate_audio

    class _Backend:
        def generate_audio_response(self, **_kwargs):
            raise AudioRuntimeError("Vevo2 requires target_voice", status = 500)

    responses: queue.Queue = queue.Queue()
    _handle_generate_audio(
        _Backend(), {"request_id": "r1", "text": "hi"}, responses, threading.Event()
    )
    sent = responses.get_nowait()
    assert sent["type"] == "audio_error"
    assert (sent["code"], sent["status"], sent["error"]) == (
        AUDIO_RUNTIME_ERROR_CODE,
        500,
        "Vevo2 requires target_voice",
    )

    class _Plain:
        def generate_audio_response(self, **_kwargs):
            raise RuntimeError("boom")

    _handle_generate_audio(
        _Plain(), {"request_id": "r2", "text": "hi"}, responses, threading.Event()
    )
    assert "code" not in responses.get_nowait()


def test_the_orchestrator_raises_the_tagged_payload_typed(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []
    payload = {
        "type": "audio_error",
        "error": "Vevo2 requires target_voice",
        "code": AUDIO_RUNTIME_ERROR_CODE,
        "status": 500,
    }

    def read_one(*, timeout):
        return {**payload, "request_id": sent[0]["request_id"]}

    for name, value in {
        "_ensure_subprocess_alive": lambda: True,
        "_send_cmd": lambda cmd: sent.append(cmd),
        "_direct_reader": lambda _rid, _cancel = None: (read_one, lambda **_k: None, lambda: None),
        "_reserve_worker": lambda _why: _Nullcontext(),
        "_wait_worker_idle": lambda **_k: True,
        "_claim_worker": lambda _cancel: None,
        "_release_worker": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(orchestrator, name, value, raising = False)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    with pytest.raises(AudioRuntimeError) as excinfo:
        orchestrator.generate_audio_response("hello")
    assert (excinfo.value.status, str(excinfo.value)) == (500, "Vevo2 requires target_voice")


class _Nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_a_transcription_error_is_sanitized_before_it_reaches_the_client():
    from core.inference import stt_audiocpp_sidecar as stt
    from core.inference.audio_cpp_server import AudioCppRequestError
    from core.inference.stt_sidecar import SttAudioDecodeError

    class _Server:
        model_id = "studio-test"
        model = type("M", (), {"family": "qwen3_asr"})()

        def __init__(self, status):
            self.status = status

        def post_json(self, *_args, **_kwargs):
            raise AudioCppRequestError(self.status, _LEAKY)

    sidecar = stt.AudioCppSttSidecar.__new__(stt.AudioCppSttSidecar)
    sidecar._server = _Server(500)
    with pytest.raises(stt.SttEngineUnavailableError) as failed:
        sidecar._post_details("a.wav", None, None, {})
    message = str(failed.value)
    assert message.startswith("The audio runtime failed: The audio runtime could not generate")
    assert "/home/alice" not in message and "hf_AbCdEf" not in message and "\n" not in message
    sidecar._server = _Server(400)
    with pytest.raises(SttAudioDecodeError) as refused:
        sidecar._post_details("a.wav", None, None, {})
    assert "/home/alice" not in str(refused.value)
    stt.clear_runtime_inference_failure()


def test_a_runtime_that_dies_at_load_reports_its_last_line_sanitized(tmp_path):
    from core.inference.audio_cpp_server import AudioCppServer, AudioCppUnavailableError

    (tmp_path / "server.log").write_text(
        "ggml_cuda_init: found 1 CUDA devices\n"
        "loading /home/alice/.cache/huggingface/hub/models--x/blobs/0f3a token hf_AbCdEfGhIjKlMnOpQrStUv12\n"
        "audiocpp_server failed: unsupported model family hint: crisperwhisper\n"
    )
    exited = type("P", (), {"poll": lambda self: 1, "pid": 0})()
    server = AudioCppServer(exited, 0, None, "studio-test", "cuda", tmp_path)
    with pytest.raises(AudioCppUnavailableError) as excinfo:
        server._wait_until_ready(None)
    message = str(excinfo.value)
    assert message.endswith("audiocpp_server failed: unsupported model family hint: crisperwhisper")
    assert "/home/alice" not in message and "hf_AbCdEf" not in message and "\n" not in message


def test_runtime_tail_redacts_before_the_cut():
    from core.inference.audio_errors import sanitize_runtime_tail

    secret = "/home/someone/private/models/voice.gguf"
    text = "x" * 400 + " failed to open " + secret + " " + "y" * 250
    out = sanitize_runtime_tail(text)
    assert len(out) <= 280
    assert "someone" not in out and "private" not in out
    raw = text[-280:]
    assert "private" in raw


def test_log_tail_drops_the_partial_first_line(tmp_path):
    from core.inference.audio_cpp_server import AudioCppServer

    server = AudioCppServer.__new__(AudioCppServer)
    server._config_dir = tmp_path
    (tmp_path / "server.log").write_bytes(b"/home/someone/secret/path.gguf\n" * 100 + b"last line")
    tail = server.log_tail(limit = 50)
    assert tail.endswith("last line")
    assert tail.split("\n")[0] in ("/home/someone/secret/path.gguf", "last line")


def test_file_urls_and_colon_prefixed_paths_are_redacted_but_network_urls_kept():
    from core.inference.audio_errors import sanitize_runtime_detail

    assert sanitize_runtime_detail("failed: file:///home/alice/private/m.gguf") == "failed: m.gguf"
    assert sanitize_runtime_detail("path:/home/alice/x.gguf bad") == "path:x.gguf bad"
    url = "see https://huggingface.co/a/b/resolve/main/x.gguf"
    assert sanitize_runtime_detail(url) == url
    assert sanitize_runtime_detail("family:cosyvoice3 needs ref") == "family:cosyvoice3 needs ref"


def test_log_tail_without_a_newline_drops_the_partial_first_word(tmp_path):
    from core.inference.audio_cpp_server import AudioCppServer

    server = AudioCppServer.__new__(AudioCppServer)
    server._config_dir = tmp_path
    (tmp_path / "server.log").write_bytes(b"token=hf_secretvalue " * 20 + b"end")
    tail = server.log_tail(limit = 50)
    assert tail.split(" ")[0] in ("token=hf_secretvalue", "end")


def test_log_tail_with_no_delimiter_is_dropped(tmp_path):
    from core.inference.audio_cpp_server import AudioCppServer

    server = AudioCppServer.__new__(AudioCppServer)
    server._config_dir = tmp_path
    (tmp_path / "server.log").write_bytes(b"x" * 200)
    assert server.log_tail(limit = 50) == ""
