# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for audio context allocation and incomplete transcriptions."""

import json
import threading
from types import SimpleNamespace

import pytest

from core.inference import stt_mtmd_sidecar as mtmd
from core.inference.stt_sidecar import SttContextSizeError, SttIncompleteTranscriptionError


MODEL = "qwen3-asr-0.6b"


class _Process:
    _next_pid = 81000

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


class _Wire:
    """A local server protocol double, including the separate audio token-count endpoint."""

    def __init__(self):
        self.requests = []
        self.input_tokens = 200
        self.count_status = 200
        self.finish_reason = "stop"
        self.before_count = None
        self.before_generate = None
        self.context_by_port = {}

    def connection(self, *args, **kwargs):
        wire = self
        port = args[1]

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
                wire.requests.append((path, self.payload, headers))

            def getresponse(self):
                if self.path == "/props":
                    return Response(
                        200, {"default_generation_settings": {"n_ctx": wire.context_by_port[port]}}
                    )
                is_count = self.path.endswith("/input_tokens")
                callback = wire.before_count if is_count else wire.before_generate
                if callback is not None:
                    callback()
                if is_count:
                    body = (
                        {"input_tokens": wire.input_tokens, "object": "response.input_tokens"}
                        if wire.count_status == 200
                        else {"error": {"message": "not found"}}
                    )
                    return Response(wire.count_status, body)
                return Response(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "language English<asr_text>Complete sentence."
                                },
                                "finish_reason": wire.finish_reason,
                            }
                        ]
                    },
                )

            def close(self):
                pass

        class Response:
            def __init__(self, status, body):
                self.status, self.body = status, body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.body).encode()

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"language English<asr_text>Complete sentence."},"finish_reason":null}]}\n'
                event = {"choices": [{"delta": {}, "finish_reason": wire.finish_reason}]}
                yield b"data: " + json.dumps(event).encode() + b"\n"
                yield b"data: [DONE]\n"

        return Connection()

    def count_requests(self):
        return [request for request in self.requests if request[0].endswith("/input_tokens")]

    def generation_requests(self):
        return [request for request in self.requests if request[0] == "/v1/chat/completions"]


@pytest.fixture
def runtime(monkeypatch):
    from core.inference import llama_cpp

    wire, processes = _Wire(), []

    def spawn(command, **kwargs):
        process = _Process(command)
        processes.append(process)
        wire.context_by_port[int(command[command.index("--port") + 1])] = _context(process)
        return process

    monkeypatch.setattr(mtmd.subprocess, "Popen", spawn)
    monkeypatch.setattr(mtmd, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "forget_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "ensure_engine_available", lambda: "/fake/llama-server")
    monkeypatch.setattr(mtmd, "_llama_server_child_env", lambda _binary: {})
    monkeypatch.setattr(mtmd, "_training_active", lambda: False)
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(
        mtmd.MtmdSttSidecar, "_ensure_model_downloaded", lambda *_args: ("m.gguf", "p.gguf")
    )
    monkeypatch.setattr(
        mtmd.MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *_args, **_kwargs: True)
    )
    monkeypatch.setattr(mtmd, "_decode_audio_bounded", lambda *_args: b"x" * 16000)
    monkeypatch.setattr(mtmd, "_pcm_to_wav_bytes", lambda _audio: b"RIFFwav")
    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", wire.connection)
    sidecar = mtmd.MtmdSttSidecar(keep_alive_seconds = 0)
    yield SimpleNamespace(sidecar = sidecar, wire = wire, processes = processes)
    sidecar.unload(wait = False)


def _context(process):
    return int(process.command[process.command.index("--ctx-size") + 1])


@pytest.mark.parametrize("stream", [False, True])
def test_length_finish_never_reports_a_successful_transcript(monkeypatch, stream):
    """A server can return HTTP 200 and useful partial text despite exhausting the token budget."""
    wire = _Wire()
    wire.finish_reason = "length"
    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", wire.connection)
    sidecar = mtmd.MtmdSttSidecar(keep_alive_seconds = 0)
    progress = []
    with pytest.raises(
        SttIncompleteTranscriptionError, match = "(?i)context|limit|incomplete|truncat"
    ):
        sidecar._post_transcribe(
            12345,
            MODEL,
            b"RIFFwav",
            1.0,
            on_progress = progress.append if stream else None,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_normal_stop_still_returns_complete_transcript(monkeypatch, stream):
    wire = _Wire()
    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", wire.connection)
    progress = []
    assert (
        mtmd.MtmdSttSidecar(keep_alive_seconds = 0)._post_transcribe(
            12345,
            MODEL,
            b"RIFFwav",
            1.0,
            on_progress = progress.append if stream else None,
        )
        == "Complete sentence."
    )


@pytest.mark.parametrize(
    "input_tokens, expected_context", [(200, 4096), (3520, 4096), (3521, 8192)]
)
def test_auto_reserves_audio_output_and_margin(runtime, input_tokens, expected_context):
    runtime.wire.input_tokens = input_tokens
    result = runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert result["text"] == "Complete sentence."
    assert runtime.wire.count_requests(), "Auto must count the actual prepared audio prompt"
    assert _context(runtime.processes[-1]) == expected_context
    assert runtime.wire.generation_requests()[-1][1]["max_tokens"] == 512


def test_device_less_warm_load_preserves_selected_context(runtime):
    runtime.sidecar.load(MODEL, context_length = 8192)
    process = runtime.processes[-1]
    runtime.sidecar.load(MODEL, context_length = None)
    assert runtime.processes == [process]
    assert _context(process) == 8192


def test_manual_context_bounds_output_without_reloading_larger(runtime):
    runtime.wire.input_tokens = 900
    result = runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 1024)
    assert result["text"] == "Complete sentence."
    assert all(_context(process) == 1024 for process in runtime.processes)
    assert runtime.wire.generation_requests()[-1][1]["max_tokens"] == 60


def test_count_and_generation_share_language_prefill_and_authentication(runtime):
    runtime.sidecar.load(MODEL, context_length = 4096)
    runtime.sidecar._api_key = "test-launch-key"
    runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 4096, language = "en-GB")
    count = runtime.wire.count_requests()[-1]
    generation = runtime.wire.generation_requests()[-1]
    assert count[1]["messages"] == generation[1]["messages"]
    assert count[1]["messages"][-1] == {
        "role": "assistant",
        "content": "language English<asr_text>",
    }
    for request in (count, generation):
        assert request[1]["continue_final_message"] is True
        assert request[1]["add_generation_prompt"] is False
        assert request[2]["Authorization"] == "Bearer test-launch-key"


def test_manual_context_can_use_an_older_server_without_token_counting(runtime):
    runtime.wire.count_status = 404
    result = runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 4096)
    assert result["text"] == "Complete sentence."
    assert runtime.wire.generation_requests()
    assert all(_context(process) == 4096 for process in runtime.processes)


@pytest.mark.parametrize("status", [404, 405, 501])
def test_auto_keeps_legacy_capacity_when_runtime_cannot_count_audio(runtime, status):
    runtime.wire.count_status = status
    result = runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert result["text"] == "Complete sentence."
    assert runtime.wire.generation_requests()
    assert runtime.sidecar.context_length == 65536
    assert runtime.sidecar.context_length_preference == 0
    assert _context(runtime.processes[-1]) == 65536
    assert runtime.sidecar._active_requests == 0


def test_manual_context_rejects_input_that_leaves_no_output_room(runtime):
    runtime.wire.input_tokens = 960
    with pytest.raises(SttContextSizeError, match = "(?i)does not fit"):
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 1024)
    assert not runtime.wire.generation_requests()
    assert [_context(process) for process in runtime.processes] == [1024]
    assert runtime.sidecar._active_requests == 0


def test_auto_rejects_recordings_beyond_model_capacity(runtime):
    runtime.wire.input_tokens = 64961  # Input + 512 output + 64 margin exceeds 64K by one.
    with pytest.raises(SttContextSizeError, match = "(?i)64K|65536"):
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert not runtime.wire.generation_requests()
    assert [_context(process) for process in runtime.processes] == [4096]


def test_auto_accepts_exact_model_capacity(runtime):
    runtime.wire.input_tokens = 64960
    assert (
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)["text"]
        == "Complete sentence."
    )
    assert _context(runtime.processes[-1]) == 65536
    assert runtime.wire.generation_requests()[-1][1]["max_tokens"] == 512


def test_auto_resize_uses_the_new_servers_authentication_key(runtime, monkeypatch):
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: True)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "probe_server_capabilities",
        lambda _binary: {"supports_api_key_file": True},
    )
    monkeypatch.setattr(llama_cpp, "_llama_server_key_launch", lambda _key: ([], {}, None))
    keys = iter(("initial-launch-key", "resized-launch-key"))
    monkeypatch.setattr(mtmd.secrets, "token_urlsafe", lambda _length: next(keys))
    runtime.wire.input_tokens = 3521
    runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert runtime.wire.count_requests()[0][2]["Authorization"] == "Bearer initial-launch-key"
    assert runtime.wire.generation_requests()[0][2]["Authorization"] == "Bearer resized-launch-key"
    assert [_context(process) for process in runtime.processes] == [4096, 8192]
    assert runtime.processes[0].poll() is not None


def test_manual_setting_cannot_replace_auto_preference_during_count(runtime):
    runtime.wire.input_tokens = 3521
    runtime.wire.before_count = lambda: runtime.sidecar.load(MODEL, context_length = 4096)
    with pytest.raises(mtmd.SttModelBusyError):
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert not runtime.wire.generation_requests()
    assert [_context(process) for process in runtime.processes] == [4096]
    assert runtime.sidecar.context_length_preference == 0


def test_an_idle_timer_cannot_unload_a_request_while_it_counts(runtime):
    runtime.wire.before_count = lambda: runtime.sidecar._idle_unload(runtime.sidecar._generation)
    result = runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert result["text"] == "Complete sentence."
    assert len(runtime.processes) == 1
    assert runtime.processes[0].poll() is None


def test_forced_unload_during_count_does_not_send_audio_to_a_stale_port(runtime, monkeypatch):
    monkeypatch.setattr(mtmd, "_ACTIVE_REQUEST_DRAIN_TIMEOUT", 0)
    runtime.wire.before_count = lambda: runtime.sidecar.unload(wait = True)
    with pytest.raises(mtmd.SttModelBusyError):
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert not runtime.wire.generation_requests()
    assert runtime.processes[0].poll() is not None
    assert runtime.sidecar._active_requests == 0


def test_cancelled_resize_wait_releases_its_request_without_killing_the_server(
    runtime, monkeypatch
):
    runtime.wire.input_tokens = 3521
    counting, return_count, resizing = threading.Event(), threading.Event(), threading.Event()
    cancelled = threading.Event()
    errors = []
    original_resize = runtime.sidecar._resize_for_request

    def count():
        counting.set()
        assert return_count.wait(3)

    def resize(*args, **kwargs):
        resizing.set()
        return original_resize(*args, **kwargs)

    runtime.wire.before_count = count
    monkeypatch.setattr(runtime.sidecar, "_resize_for_request", resize)

    def transcribe():
        try:
            runtime.sidecar.transcribe(
                b"encoded", model = MODEL, context_length = 0, cancel_event = cancelled
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target = transcribe, daemon = True)
    worker.start()
    acquired = False
    try:
        assert counting.wait(3)
        acquired = runtime.sidecar._start_lock.acquire(timeout = 2)
        assert acquired
        return_count.set()
        assert resizing.wait(3)
        runtime.sidecar.cancel_transcription(cancelled)
        worker.join(timeout = 2)
        assert not worker.is_alive(), "cancellation remained blocked behind another startup"
    finally:
        return_count.set()
        if acquired:
            runtime.sidecar._start_lock.release()
        worker.join(timeout = 3)
    assert len(errors) == 1 and isinstance(errors[0], mtmd.SttTranscriptionCancelledError)
    assert runtime.sidecar._active_requests == 0
    assert not runtime.wire.generation_requests()
    assert len(runtime.processes) == 1 and runtime.processes[0].poll() is None


@pytest.mark.parametrize("second_input_tokens, second_succeeds", [(8000, False), (200, True)])
def test_context_resize_does_not_interrupt_a_sibling_transcription(
    runtime, second_input_tokens, second_succeeds
):
    first_started, release_first = threading.Event(), threading.Event()
    first_results, first_errors = [], []

    def count():
        runtime.wire.input_tokens = (
            3521 if threading.current_thread().name == "first-recording" else second_input_tokens
        )

    def generate():
        if threading.current_thread().name == "first-recording":
            first_started.set()
            assert release_first.wait(5)

    runtime.wire.before_count = count
    runtime.wire.before_generate = generate

    def first_request():
        try:
            first_results.append(
                runtime.sidecar.transcribe(b"first", model = MODEL, context_length = 0)
            )
        except Exception as exc:
            first_errors.append(exc)

    worker = threading.Thread(target = first_request, name = "first-recording", daemon = True)
    worker.start()
    try:
        assert first_started.wait(3)
        resident = runtime.processes[-1]
        assert _context(resident) == 8192
        if second_succeeds:
            assert (
                runtime.sidecar.transcribe(b"second", model = MODEL, context_length = 0)["text"]
                == "Complete sentence."
            )
        else:
            # This case needs a genuinely larger window than the first recording's 8K.
            with pytest.raises(mtmd.SttModelBusyError):
                runtime.sidecar.transcribe(b"second", model = MODEL, context_length = 0)
        assert runtime.processes[-1] is resident and resident.poll() is None
    finally:
        release_first.set()
        worker.join(timeout = 5)
    assert not worker.is_alive()
    assert not first_errors
    assert first_results[0]["text"] == "Complete sentence."
    assert [_context(process) for process in runtime.processes] == [4096, 8192]
    assert runtime.sidecar._active_requests == 0


def test_setting_changed_between_load_and_request_claim_is_not_overwritten(runtime, monkeypatch):
    """A same-model setting change need not replace the process or bump its generation."""
    original_load = runtime.sidecar.load
    runtime.wire.input_tokens = 3521

    def load_then_select_manual(*args, **kwargs):
        original_load(*args, **kwargs)
        original_load(MODEL, context_length = 4096)

    monkeypatch.setattr(runtime.sidecar, "load", load_then_select_manual)
    with pytest.raises(mtmd.SttModelBusyError):
        runtime.sidecar.transcribe(b"encoded", model = MODEL, context_length = 0)
    assert not runtime.wire.generation_requests()
    assert [_context(process) for process in runtime.processes] == [4096]
    assert runtime.sidecar.context_length_preference == 4096


def test_server_cannot_silently_exceed_the_requested_manual_context(runtime, monkeypatch):
    monkeypatch.setattr(runtime.sidecar, "_read_server_context", lambda *_args, **_kwargs: 8192)
    with pytest.raises(SttContextSizeError, match = "8192.*4096"):
        runtime.sidecar.load(MODEL, context_length = 4096)
    assert runtime.sidecar.loaded_model is None
    assert len(runtime.processes) == 1 and runtime.processes[0].poll() is not None
    assert runtime.sidecar.is_loading() is False


@pytest.mark.parametrize(
    "failure",
    [ConnectionResetError("connection reset"), mtmd.SttUnavailableError("props unavailable")],
)
def test_preemption_during_context_verification_stays_retryable(runtime, monkeypatch, failure):
    """Training can terminate a healthy child while the new /props probe is in flight."""

    def preempt_during_props(*_args, **_kwargs):
        assert runtime.sidecar.cancel_pending_load()
        raise failure

    monkeypatch.setattr(runtime.sidecar, "_read_server_context", preempt_during_props)
    with pytest.raises(mtmd.SttLoadCancelledError):
        runtime.sidecar.load(MODEL, context_length = 4096)
    assert runtime.sidecar.loaded_model is None
    assert runtime.sidecar.is_loading() is False
    assert runtime.processes[0].poll() is not None


def test_cancelling_context_verification_interrupts_a_stalled_http_read(runtime, monkeypatch):
    """An unresponsive child must not turn cancellation into a sixty-second /props timeout."""
    import subprocess

    reading, unblock = threading.Event(), threading.Event()
    cancelled = threading.Event()
    errors = []

    class Socket:
        def shutdown(self, _how):
            unblock.set()

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            reading.set()
            assert unblock.wait(5), "the test did not clean up its blocked HTTP read"
            raise OSError("connection interrupted")

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.sock = Socket()

        def request(self, method, path, **_kwargs):
            assert method == "GET" and path == "/props"

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", Connection)

    def load():
        try:
            runtime.sidecar.load(MODEL, context_length = 4096, request_cancel_event = cancelled)
        except Exception as exc:
            errors.append(exc)

    loader = threading.Thread(target = load, daemon = True)
    loader.start()
    try:
        assert reading.wait(3)
        process = runtime.processes[0]
        # Match an actual native child that ignores graceful shutdown until it is killed.
        monkeypatch.setattr(process, "terminate", lambda: None)

        def wait(timeout = None):
            if process.poll() is None:
                raise subprocess.TimeoutExpired("llama-server", timeout)
            return process.returncode

        monkeypatch.setattr(process, "wait", wait)
        runtime.sidecar.cancel_transcription(cancelled)
        loader.join(timeout = 1)
        assert not loader.is_alive(), "cancelled load is still waiting on the /props HTTP timeout"
    finally:
        unblock.set()
        loader.join(timeout = 3)
    assert len(errors) == 1 and isinstance(errors[0], mtmd.SttLoadCancelledError)
    assert process.poll() is not None
    assert runtime.sidecar.is_loading() is False
