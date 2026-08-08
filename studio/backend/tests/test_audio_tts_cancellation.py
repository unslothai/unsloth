# SPDX-License-Identifier: AGPL-3.0-only
"""Request-scoped cancellation for blocking Transformers TTS generation."""

import asyncio
import base64
import queue
import threading
import time

import pytest

import routes.inference as inference_route
from core.inference import orchestrator as orchestrator_module
from core.inference.orchestrator import InferenceOrchestrator
from core.inference.worker import _handle_generate_audio, _prepare_generate_audio
from models.inference import ChatCompletionRequest


def _bare_orchestrator():
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._active_cancel_lock = threading.Lock()
    orchestrator._active_cancel_events = []
    orchestrator._executing_cancel_events = []
    orchestrator._cancel_event = threading.Event()
    orchestrator._drain_event = threading.Event()
    orchestrator._proc = object()
    orchestrator._cmd_queue = object()
    orchestrator._resp_queue = object()
    orchestrator._dispatcher_thread = None
    orchestrator._dispatcher_stop = threading.Event()
    orchestrator._dispatcher_lifecycle_lock = threading.Lock()
    orchestrator._mailbox_lock = threading.Lock()
    orchestrator._mailboxes = {}
    orchestrator._direct_mailboxes = {}
    orchestrator._request_cancel_events = {}
    orchestrator._unload_pending = False
    orchestrator._exclusive_tts_pending = False
    orchestrator.active_model_name = "model"
    orchestrator.models = {"model": {}}
    orchestrator.loading_models = set()
    return orchestrator


def test_route_passes_request_cancel_event_to_transformers_backend(monkeypatch):
    captured = {}

    class _Llama:
        is_loaded = False
        _is_audio = False

    class _Backend:
        active_model_name = "some/custom-tts"
        models = {"some/custom-tts": {"is_audio": True, "audio_type": "snac"}}

        def generate_audio_response(self, **kwargs):
            captured.update(kwargs)
            return b"RIFFfake", 24000

    async def _noop_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = "some/custom-tts",
        messages = [{"role": "user", "content": "hello"}],
    )

    asyncio.run(
        inference_route._generate_tts_wav("hello", payload, request = None, current_subject = "t")
    )

    assert "cancel_event" in captured
    assert captured["cancel_event"].is_set() is False


def test_audio_response_stopped_while_queued_is_never_sent(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        orchestrator,
        "_send_cmd",
        lambda _cmd: pytest.fail("must not send TTS already stopped"),
    )
    stopped = threading.Event()
    stopped.set()

    with pytest.raises(RuntimeError, match = "cancel"):
        orchestrator.generate_audio_response("hello", cancel_event = stopped)

    assert orchestrator._active_cancel_events == []


def test_audio_response_cancellation_signals_worker_and_drains_terminal_response(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))
    caller_cancel = threading.Event()
    released = []
    reads = 0

    def read_one(*, timeout):
        nonlocal reads
        reads += 1
        if reads == 1:
            caller_cancel.set()
            assert orchestrator._cancel_event.is_set() is False
            return None
        if reads == 2:
            # The worker acknowledges only after clearing stale shared state. The
            # parent must not signal while the TTS command is merely queued.
            assert orchestrator._cancel_event.is_set() is False
            return {
                "type": "audio_started",
                "request_id": sent[0]["request_id"],
            }
        assert orchestrator._cancel_event.is_set() is True
        return {
            "type": "audio_error",
            "request_id": sent[0]["request_id"],
            "error": "cancelled",
        }

    monkeypatch.setattr(
        orchestrator,
        "_direct_reader",
        lambda _request_id: (read_one, lambda **_kwargs: None, lambda: released.append(True)),
    )

    with pytest.raises(RuntimeError, match = "cancel"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert sent and sent[0]["type"] == "generate_audio"
    assert reads == 3
    assert released == [True]
    assert orchestrator._active_cancel_events == []
    assert orchestrator._executing_cancel_events == []


def test_audio_response_cancellation_bounds_an_unresponsive_worker(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 100.0)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_DRAIN_TIMEOUT", 0.03)
    caller_cancel = threading.Event()
    sent = []

    def read_one(*, timeout):
        if not caller_cancel.is_set():
            caller_cancel.set()
            return {
                "type": "audio_started",
                "request_id": sent[0]["request_id"],
            }
        time.sleep(timeout)
        return None

    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))
    cancel_signals = []
    monkeypatch.setattr(orchestrator, "_cancel_generation", lambda: cancel_signals.append(True))
    monkeypatch.setattr(
        orchestrator,
        "_direct_reader",
        lambda _request_id: (
            read_one,
            lambda **_kwargs: pytest.fail("the cancellation drain window was already spent"),
            lambda: None,
        ),
    )
    shutdown_state = []

    def shutdown(*, timeout):
        shutdown_state.append((orchestrator._exclusive_tts_pending, timeout))
        return True

    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", shutdown)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "Audio generation cancelled"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert time.monotonic() - started < 0.5
    assert cancel_signals == [True]
    assert shutdown_state == [(True, 0.03)]
    assert orchestrator.active_model_name is None
    assert orchestrator.models == {}
    assert orchestrator._exclusive_tts_pending is False


def test_audio_response_cancellation_before_worker_start_is_still_bounded(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 100.0)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_DRAIN_TIMEOUT", 0.03)
    caller_cancel = threading.Event()

    def send(_cmd):
        caller_cancel.set()

    def read_one(*, timeout):
        time.sleep(timeout)
        return None

    monkeypatch.setattr(orchestrator, "_send_cmd", send)
    monkeypatch.setattr(
        orchestrator,
        "_cancel_generation",
        lambda: pytest.fail("must not signal shared cancellation before audio_started"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_direct_reader",
        lambda _request_id: (
            read_one,
            lambda **_kwargs: pytest.fail("the cancellation drain window was already spent"),
            lambda: None,
        ),
    )
    shutdown_state = []
    monkeypatch.setattr(
        orchestrator,
        "_shutdown_subprocess",
        lambda *, timeout: shutdown_state.append(timeout) or True,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "Audio generation cancelled"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert time.monotonic() - started < 0.5
    assert shutdown_state == [0.03]
    assert orchestrator._exclusive_tts_pending is False


def test_audio_generation_timeout_scales_with_requested_tokens(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 10.0)

    assert orchestrator_module._audio_generation_timeout(512) == 10.0
    assert orchestrator_module._audio_generation_timeout(2048) == 10.0
    assert orchestrator_module._audio_generation_timeout(8192) == 40.0
    assert orchestrator_module._audio_generation_timeout(10**310) == 40.0


def test_tts_route_bounds_public_token_budget():
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        max_tokens = 10**310,
    )

    assert inference_route._tts_max_new_tokens(payload) == 8192


def test_audio_worker_command_uses_the_bounded_token_budget(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))

    def direct_reader(request_id):
        responses = queue.Queue()
        responses.put(
            {
                "type": "audio_done",
                "request_id": request_id,
                "wav_base64": base64.b64encode(b"RIFFfake").decode("ascii"),
                "sample_rate": 24000,
            }
        )
        return (
            lambda *, timeout: responses.get(timeout = timeout),
            lambda **_kwargs: None,
            lambda: None,
        )

    monkeypatch.setattr(orchestrator, "_direct_reader", direct_reader)

    assert orchestrator.generate_audio_response("hello", max_new_tokens = 10**310) == (
        b"RIFFfake",
        24000,
    )
    assert sent[0]["max_new_tokens"] == 8192


def test_audio_response_timeout_cancels_and_drains_before_releasing(monkeypatch):
    orchestrator = _bare_orchestrator()
    orchestrator._resp_queue = queue.Queue()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 0.05)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_DRAIN_TIMEOUT", 0.2)
    monkeypatch.setattr(
        orchestrator,
        "_shutdown_subprocess",
        lambda **_kwargs: pytest.fail("a drained timeout must not tear the worker down"),
    )

    sent = []

    def send(cmd):
        sent.append(cmd)
        orchestrator._resp_queue.put({"type": "audio_started", "request_id": cmd["request_id"]})

    monkeypatch.setattr(orchestrator, "_send_cmd", send)
    cancel_state = []

    def cancel_generation():
        cancel_state.append(orchestrator._exclusive_tts_pending)
        orchestrator._cancel_event.set()
        orchestrator._resp_queue.put(
            {
                "type": "audio_error",
                "request_id": sent[0]["request_id"],
                "error": "cancelled",
            }
        )

    monkeypatch.setattr(orchestrator, "_cancel_generation", cancel_generation)

    with pytest.raises(RuntimeError, match = "Timeout waiting for audio generation"):
        orchestrator.generate_audio_response("hello")

    assert cancel_state == [True], "timeout cancellation must occur under TTS exclusivity"
    assert orchestrator._exclusive_tts_pending is False
    assert orchestrator._active_cancel_events == []
    assert orchestrator._executing_cancel_events == []


def test_audio_response_timeout_tears_down_unresponsive_worker_before_release(monkeypatch):
    orchestrator = _bare_orchestrator()
    orchestrator._resp_queue = queue.Queue()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 0.03)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_DRAIN_TIMEOUT", 0.03)

    sent = []

    def send(cmd):
        sent.append(cmd)
        orchestrator._resp_queue.put({"type": "audio_started", "request_id": cmd["request_id"]})

    monkeypatch.setattr(orchestrator, "_send_cmd", send)
    monkeypatch.setattr(orchestrator, "_cancel_generation", lambda: None)
    shutdown_state = []

    def shutdown(*, timeout):
        shutdown_state.append((orchestrator._exclusive_tts_pending, timeout))
        return True

    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", shutdown)

    with pytest.raises(RuntimeError, match = "Timeout waiting for audio generation"):
        orchestrator.generate_audio_response("hello")

    assert shutdown_state == [(True, 0.03)]
    assert orchestrator._exclusive_tts_pending is False
    assert orchestrator.active_model_name is None
    assert orchestrator.models == {}


def test_worker_audio_prepare_rechecks_unload_drain_after_clear():
    drain = threading.Event()

    class _Cancel:
        def clear(self):
            # Exact race: unload lands after the first drain check but before the
            # worker clear would otherwise erase its shared cancel.
            drain.set()

    responses = queue.Queue()
    ready = _prepare_generate_audio(
        {"type": "generate_audio", "request_id": "audio-1"},
        responses,
        _Cancel(),
        drain,
    )

    assert ready is False
    response = responses.get_nowait()
    assert response["type"] == "audio_error"
    assert response["request_id"] == "audio-1"
    assert response["cancelled"] is True
    assert response["error"] == "Audio generation cancelled"
    assert responses.empty(), "audio_started must not be emitted for a drained request"


def test_worker_audio_prepare_acknowledges_only_after_cancel_clear():
    operations = []

    class _Cancel:
        def clear(self):
            operations.append("clear")

    class _Responses:
        def put(self, response):
            operations.append(response["type"])

    assert _prepare_generate_audio(
        {"type": "generate_audio", "request_id": "audio-1"},
        _Responses(),
        _Cancel(),
        threading.Event(),
    )
    assert operations == ["clear", "audio_started"]


class _AliveDispatcher:
    def is_alive(self):
        return True


def test_dispatcher_refuses_during_exclusive_tts_and_resumes_after():
    orchestrator = _bare_orchestrator()
    orchestrator._resp_queue = queue.Queue()
    orchestrator._exclusive_tts_pending = True

    assert orchestrator._start_dispatcher() is False
    assert orchestrator._dispatcher_thread is None

    orchestrator._exclusive_tts_pending = False
    try:
        assert orchestrator._start_dispatcher() is True
        assert orchestrator._dispatcher_thread is not None
        assert orchestrator._dispatcher_thread.is_alive()
    finally:
        orchestrator._stop_dispatcher()


def test_tts_waits_for_existing_compare_before_send(monkeypatch):
    orchestrator = _bare_orchestrator()
    orchestrator._dispatcher_thread = _AliveDispatcher()
    orchestrator._mailboxes["compare"] = queue.Queue()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)

    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(
        orchestrator,
        "_stop_dispatcher",
        lambda: setattr(orchestrator, "_dispatcher_thread", None),
    )

    responses = queue.Queue()

    def direct_reader(request_id):
        responses.put({"type": "audio_started", "request_id": request_id})
        responses.put(
            {
                "type": "audio_done",
                "request_id": request_id,
                "wav_base64": base64.b64encode(b"RIFFfake").decode("ascii"),
                "sample_rate": 24000,
            }
        )
        return (
            lambda *, timeout: responses.get(timeout = timeout),
            lambda **_kwargs: None,
            lambda: None,
        )

    monkeypatch.setattr(orchestrator, "_direct_reader", direct_reader)
    result = {}
    thread = threading.Thread(
        target = lambda: result.setdefault("value", orchestrator.generate_audio_response("hello"))
    )
    thread.start()

    deadline = time.monotonic() + 2
    while not orchestrator._exclusive_tts_pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert orchestrator._exclusive_tts_pending is True
    assert sent == [], "TTS must not enqueue behind the active compare request"

    with orchestrator._mailbox_lock:
        orchestrator._mailboxes.pop("compare")
    thread.join(timeout = 3)

    assert thread.is_alive() is False
    assert result["value"] == (b"RIFFfake", 24000)
    assert sent and sent[0]["type"] == "generate_audio"
    assert orchestrator._exclusive_tts_pending is False


def test_tts_cancel_while_waiting_does_not_signal_active_compare(monkeypatch):
    orchestrator = _bare_orchestrator()
    orchestrator._dispatcher_thread = _AliveDispatcher()
    orchestrator._mailboxes["compare"] = queue.Queue()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        orchestrator,
        "_send_cmd",
        lambda _cmd: pytest.fail("cancelled queued TTS must not be sent"),
    )
    cancel_calls = []
    monkeypatch.setattr(orchestrator, "_cancel_generation", lambda: cancel_calls.append(True))
    caller_cancel = threading.Event()
    error = {}

    def run():
        try:
            orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)
        except Exception as exc:  # noqa: BLE001 - assertion captures the thread result
            error["value"] = exc

    thread = threading.Thread(target = run)
    thread.start()
    deadline = time.monotonic() + 2
    while not orchestrator._exclusive_tts_pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert orchestrator._exclusive_tts_pending is True

    caller_cancel.set()
    thread.join(timeout = 2)

    assert thread.is_alive() is False
    assert "cancel" in str(error["value"]).lower()
    assert cancel_calls == []
    assert set(orchestrator._mailboxes) == {"compare"}
    assert orchestrator._exclusive_tts_pending is False


def test_dispatched_generation_rechecks_tts_reservation_before_registration(monkeypatch):
    orchestrator = _bare_orchestrator()
    orchestrator._dispatcher_thread = _AliveDispatcher()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator, "_start_dispatcher", lambda: False)

    def reserve_tts(*_args, **_kwargs):
        orchestrator._exclusive_tts_pending = True
        return {"type": "generate", "request_id": "compare-1"}

    monkeypatch.setattr(orchestrator, "_build_generate_cmd", reserve_tts)
    monkeypatch.setattr(
        orchestrator,
        "_send_cmd",
        lambda _cmd: pytest.fail("compare must not enqueue after TTS reservation"),
    )

    output = list(orchestrator._generate_dispatched(messages = [{"role": "user", "content": "x"}]))

    assert any("audio generation" in str(chunk).lower() for chunk in output)
    assert orchestrator._mailboxes == {}


def test_worker_audio_forwards_shared_cancel_event():
    cancel = threading.Event()
    captured = {}

    class _Backend:
        def generate_audio_response(self, **kwargs):
            captured.update(kwargs)
            return b"RIFFfake", 24000

    responses = queue.Queue()
    _handle_generate_audio(
        _Backend(),
        {"request_id": "audio-1", "text": "hello"},
        responses,
        cancel,
    )

    assert captured["cancel_event"] is cancel
    assert responses.get_nowait()["type"] == "audio_done"


def test_backend_tts_generation_uses_cancel_stopping_criteria(monkeypatch):
    from core.inference.inference import InferenceBackend

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "tts"
    backend._generation_lock = threading.Lock()
    backend.models = {
        "tts": {
            "audio_type": "bicodec",
            "model": object(),
            "tokenizer": object(),
        }
    }
    criteria = object()
    monkeypatch.setattr(backend, "_cancel_stopping_criteria", lambda event: criteria)
    captured = {}

    def _fake_generate(*_args, **kwargs):
        captured.update(kwargs)
        return b"RIFFfake", 24000

    monkeypatch.setattr(backend, "_generate_bicodec", _fake_generate)
    cancel = threading.Event()

    assert backend.generate_audio_response("hello", cancel_event = cancel) == (b"RIFFfake", 24000)
    assert captured["stopping_criteria"] is criteria
