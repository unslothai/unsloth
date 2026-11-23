# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cold Auto llama.cpp ASR server starts once, at the window the clip's token count picks."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import stt_mtmd_sidecar as mtmd
from core.inference import stt_registry
from core.inference.stt_sidecar import SttAudioDecodeError
from routes import inference as ri


MODEL = "qwen3-asr-0.6b"
RATE = 16000


def _runtime_tokens(seconds):
    # What the bundled runtime counts for Qwen3-ASR: 13 audio tokens per whole second plus the prompt.
    return 13 * int(seconds) + 28


def _counted_context(seconds):
    return mtmd._auto_context_length(
        _runtime_tokens(seconds), mtmd._transcript_token_budget(seconds)
    )


class _Process:
    _next_pid = 91000

    def __init__(self, command):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.command = list(command)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout = None):
        return self.returncode


class _Response:
    def __init__(self, status, body):
        self.status, self.body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.body).encode()

    def __iter__(self):
        delta = {"content": "language English<asr_text>Hello there."}
        yield (
            b"data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": None}]}).encode()
        )
        yield b"\n"
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        yield b"data: [DONE]\n"


class _Wire:
    """llama-server's /props, /input_tokens and /v1/chat/completions, per port."""

    def __init__(self):
        self.input_tokens = 200
        self.context_by_port = {}
        self.max_tokens = []

    def connection(self, _host, port, **_kwargs):
        wire = self

        class Connection:
            sock = None

            def request(
                self,
                method,
                path,
                body = None,
                headers = None,
            ):
                self.path = path
                self.payload = json.loads(body) if body is not None else None

            def getresponse(self):
                if self.path == "/props":
                    return _Response(
                        200, {"default_generation_settings": {"n_ctx": wire.context_by_port[port]}}
                    )
                if self.path.endswith("/input_tokens"):
                    return _Response(200, {"input_tokens": wire.input_tokens})
                wire.max_tokens.append(self.payload["max_tokens"])
                message = {"content": "language English<asr_text>Hello there."}
                return _Response(200, {"choices": [{"message": message, "finish_reason": "stop"}]})

            def close(self):
                pass

        return Connection()


def _context(process):
    return int(process.command[process.command.index("--ctx-size") + 1])


@pytest.fixture
def studio(monkeypatch):
    """The real route, registry and sidecar; only the process, the wire and the decoder are doubles."""
    from core.inference import llama_cpp

    wire, processes, decodes, downloaded = _Wire(), [], [], {"value": True}

    def spawn(command, **_kwargs):
        process = _Process(command)
        processes.append(process)
        wire.context_by_port[int(command[command.index("--port") + 1])] = _context(process)
        return process

    def decode(audio, _cancel_event = None):
        decodes.append(audio)
        if audio == b"not audio":
            raise SttAudioDecodeError("Could not decode the audio.")
        # The clip length in seconds is the payload, so a test names the duration it wants.
        return b"\0" * int(float(audio.decode()) * RATE)

    def ensure_downloaded(_self, model_id):
        if not downloaded["value"]:
            raise mtmd.SttModelNotDownloadedError(f"STT model '{model_id}' is not downloaded yet.")
        return "m.gguf", "p.gguf"

    monkeypatch.setattr(mtmd.subprocess, "Popen", spawn)
    monkeypatch.setattr(mtmd, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "forget_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "ensure_engine_available", lambda: "/fake/llama-server")
    monkeypatch.setattr(mtmd, "_llama_server_child_env", lambda _binary: {})
    monkeypatch.setattr(mtmd, "_training_active", lambda: False)
    monkeypatch.setattr(mtmd, "is_model_downloaded", lambda _model: downloaded["value"])
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(mtmd.MtmdSttSidecar, "_ensure_model_downloaded", ensure_downloaded)
    monkeypatch.setattr(
        mtmd.MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *_args, **_kwargs: True)
    )
    monkeypatch.setattr(mtmd, "_decode_audio_bounded", decode)
    monkeypatch.setattr(mtmd, "_pcm_to_wav_bytes", lambda _audio: b"RIFFwav")
    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", wire.connection)

    sidecar = mtmd.MtmdSttSidecar(keep_alive_seconds = 0)
    loads = []
    real_load = sidecar.load

    def recording_load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(sidecar, "load", recording_load)
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda _engine: sidecar)
    monkeypatch.setattr(stt_registry, "_model_is_downloaded", lambda *_args: downloaded["value"])
    monkeypatch.setattr(stt_registry, "unload", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (stt_registry.load, stt_registry.unload))
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda _engine: sidecar)
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: engine or "mtmd")
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *_args: None)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)

    def transcribe(
        seconds,
        *,
        context_length = 0,
        streaming = False,
        raw = None,
    ):
        wire.input_tokens = _runtime_tokens(seconds)
        return asyncio.run(
            ri._transcribe_audio_result(
                raw if raw is not None else str(seconds).encode(),
                MODEL,
                None,
                False,
                "mtmd",
                on_progress = (lambda _event: None) if streaming else None,
                context_length = context_length,
            )
        )

    yield SimpleNamespace(
        sidecar = sidecar,
        wire = wire,
        processes = processes,
        decodes = decodes,
        downloaded = downloaded,
        loads = loads,
        transcribe = transcribe,
    )
    sidecar.unload(wait = False)


@pytest.mark.parametrize("seconds", [100, 150, 180, 300, 600, 1800])
@pytest.mark.parametrize("streaming", [False, True])
def test_long_cold_clip_starts_llama_server_once(studio, seconds, streaming):
    context = _counted_context(seconds)
    assert context > 4096
    result = studio.transcribe(seconds, streaming = streaming)
    assert result["text"] == "Hello there."
    # Before: a 4096 start, then a restart once the clip was counted.
    assert [_context(process) for process in studio.processes] == [context]
    assert studio.sidecar.context_length == context
    assert studio.decodes == [str(seconds).encode()], "the clip is decoded once"


@pytest.mark.parametrize("seconds", [1, 5, 28, 60, 90])
def test_short_cold_clip_and_dictation_are_unchanged(studio, seconds):
    studio.transcribe(seconds)
    assert [_context(process) for process in studio.processes] == [4096]
    assert all("auto_start_context" not in kwargs for kwargs in studio.loads)
    assert len(studio.decodes) == 1


def test_warm_auto_server_keeps_its_window_and_resizes_as_before(studio):
    studio.transcribe(28)
    studio.transcribe(180)
    assert [_context(process) for process in studio.processes] == [4096, 8192]
    studio.transcribe(28)
    # A warm server is never sized from the clip: the count resizes it, as before.
    assert [_context(process) for process in studio.processes] == [4096, 8192, 4096]
    assert len(studio.decodes) == 3


@pytest.mark.parametrize("context_length", [4096, 8192, 32768])
def test_explicit_context_is_never_resized_by_the_clip(studio, context_length):
    studio.transcribe(180, context_length = context_length)
    assert [_context(process) for process in studio.processes] == [context_length]
    assert all("auto_start_context" not in kwargs for kwargs in studio.loads)
    assert len(studio.decodes) == 1


def test_undownloaded_model_still_reports_409_before_decoding(studio):
    studio.downloaded["value"] = False
    with pytest.raises(HTTPException) as exc:
        studio.transcribe(180, raw = b"not audio")
    assert exc.value.status_code == 409
    assert "not downloaded" in exc.value.detail
    assert studio.decodes == []
    assert studio.processes == []


def test_undecodable_upload_is_the_same_400_after_the_same_start(studio):
    with pytest.raises(HTTPException) as exc:
        studio.transcribe(180, raw = b"not audio")
    assert exc.value.status_code == 400
    assert exc.value.detail == "Could not decode the audio."
    # As before the clip could not size anything, so the usual Auto server is what started.
    assert [_context(process) for process in studio.processes] == [4096]


def test_direct_sidecar_transcription_also_starts_once(studio):
    studio.wire.input_tokens = _runtime_tokens(600)
    studio.sidecar.transcribe(b"600", model = MODEL, context_length = 0)
    assert [_context(process) for process in studio.processes] == [_counted_context(600)]


def test_start_window_never_exceeds_the_counted_window():
    long_clips = [seconds for seconds in range(1, 1801) if _counted_context(seconds) > 4096]
    still_twice = []
    for seconds in long_clips:
        for model_id in mtmd.MTMD_STT_MODELS:
            start = mtmd._cold_start_context(model_id, seconds)
            # Larger would leave the final --ctx-size to a resize and allocate more than needed.
            assert start <= _counted_context(seconds)
            if start < _counted_context(seconds):
                still_twice.append(seconds)
    # Only a clip whose prompt tokens tip it past a 4096 step still restarts.
    assert len(still_twice) <= len(long_clips) * len(mtmd.MTMD_STT_MODELS) // 100
    for seconds in range(1, 94):
        assert mtmd._cold_start_context(MODEL, seconds) == 4096
