# SPDX-License-Identifier: AGPL-3.0-only
"""Request-scoped cancellation for blocking Transformers TTS generation."""

import asyncio
import base64
import importlib
import json
import queue
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub for a dep the backend pytest job does not install.

    Same helper and reason as test_safetensors_reasoning_stream.py and
    test_audio_type_inconclusive.py: the peft-gated test below imports
    ``core.inference.inference``, which imports ``unsloth`` at module scope, and this
    job installs peft but not unsloth. That import used to be unreachable here because
    the peft gate skipped; now that peft IS installed the gate opens, and the import
    only worked because collection of test_safetensors_reasoning_stream.py had already
    cached the module. Running this file on its own failed. A real install is left alone.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

# Build it while the stubs are live, then drop them, as the sibling files do: a stub
# left in sys.modules is a cross-file leak that
# test_audio_type_inconclusive.py::test_the_stubs_do_not_outlive_this_module asserts
# against. The peft gate below still decides whether the test runs.
try:
    import core.inference.inference  # noqa: E402,F401
except ImportError:  # pragma: no cover - the real dep set imports fine
    pass

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import routes.inference as inference_route  # noqa: E402
from core.inference import orchestrator as orchestrator_module  # noqa: E402
from core.inference.orchestrator import InferenceOrchestrator  # noqa: E402
from core.inference.worker import (  # noqa: E402
    _handle_generate_audio,
    _prepare_generate_audio,
)
from models.inference import ChatCompletionRequest  # noqa: E402


def _bare_orchestrator():
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator._stop_ledger = None
    orchestrator._pending_teardowns = None
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
    orchestrator._worker_released = threading.Condition(orchestrator._dispatcher_lifecycle_lock)
    orchestrator._subprocess_shutdown_lock = threading.Lock()
    orchestrator._mailbox_lock = threading.Lock()
    orchestrator._mailboxes = {}
    orchestrator._direct_mailboxes = {}
    orchestrator._request_cancel_events = {}
    orchestrator._unload_pending = False
    orchestrator._worker_reserved_for = None
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


def test_minimax_prompt_encoder_overflow_is_a_client_error(monkeypatch):
    class _Llama:
        is_loaded = False
        _is_audio = False

    class _Backend:
        active_model_name = "MiniMaxAI/MiniMax-Music3"
        models = {
            "MiniMaxAI/MiniMax-Music3": {
                "is_audio": True,
                "audio_type": "minimax_music3",
            }
        }

        def generate_audio_response(self, **_kwargs):
            raise RuntimeError("The assembled prompt has 5001 tokens; the maximum is 5000")

    async def _noop_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = "MiniMaxAI/MiniMax-Music3",
        messages = [{"role": "user", "content": "lyrics"}],
        audio_instructions = "long music description",
    )

    with pytest.raises(inference_route.HTTPException) as excinfo:
        asyncio.run(
            inference_route._generate_tts_wav("lyrics", payload, request = None, current_subject = "t")
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "The assembled prompt has 5001 tokens; the maximum is 5000"


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
        lambda _request_id, _cancel_event = None: (
            read_one,
            lambda **_kwargs: None,
            lambda: released.append(True),
        ),
    )

    with pytest.raises(RuntimeError, match = "cancel"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert sent and sent[0]["type"] == "generate_audio"
    assert reads == 3
    assert released == [True]
    assert orchestrator._active_cancel_events == []
    assert orchestrator._executing_cancel_events == []


_GPU_TIMEOUT = (
    "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
    "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
)


class _WorkerQueue:
    def __init__(self, sent, *steps):
        self._sent = sent
        self._steps = list(steps)

    def get(self, timeout = None):
        if not self._steps:
            raise queue.Empty
        step = self._steps.pop(0)
        resp = step() if callable(step) else step
        if resp is None:
            raise queue.Empty
        return {**resp, "request_id": self._sent[0]["request_id"]}


def _watch_teardown(orchestrator, monkeypatch):
    torn_down = []

    def shutdown(timeout):
        torn_down.append(timeout)
        orchestrator._proc = None
        return True

    monkeypatch.setattr(orchestrator, "_shutdown_subprocess_locked", shutdown)
    return torn_down


@pytest.mark.parametrize("rtype", ["audio_error", "error"])
def test_a_dead_gpu_queue_during_audio_generation_retires_the_worker(rtype, monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))
    torn_down = _watch_teardown(orchestrator, monkeypatch)
    orchestrator._resp_queue = _WorkerQueue(sent, {"type": rtype, "error": _GPU_TIMEOUT})

    with pytest.raises(RuntimeError, match = "GPU Timeout"):
        orchestrator.generate_audio_response("hello")

    assert torn_down, "a dead GPU queue must retire the worker"
    assert orchestrator.active_model_name is None
    assert orchestrator.models == {}


def test_a_cancelled_audio_request_still_retires_the_poisoned_worker(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))
    torn_down = _watch_teardown(orchestrator, monkeypatch)
    caller_cancel = threading.Event()
    orchestrator._resp_queue = _WorkerQueue(
        sent,
        {"type": "audio_started"},
        caller_cancel.set,
        {"type": "audio_error", "error": _GPU_TIMEOUT},
    )

    with pytest.raises(RuntimeError, match = "cancel"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert torn_down
    assert orchestrator.active_model_name is None


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
        lambda _request_id, _cancel_event = None: (
            read_one,
            lambda **_kwargs: pytest.fail("the cancellation drain window was already spent"),
            lambda: None,
        ),
    )
    shutdown_state = []

    def shutdown(*, timeout):
        shutdown_state.append((orchestrator._worker_reserved_for, timeout))
        return True

    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", shutdown)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "Audio generation cancelled"):
        orchestrator.generate_audio_response("hello", cancel_event = caller_cancel)

    assert time.monotonic() - started < 0.5
    assert cancel_signals == [True]
    assert shutdown_state == [("audio generation is in progress", 0.03)]
    assert orchestrator.active_model_name is None
    assert orchestrator.models == {}
    assert orchestrator._worker_reserved_for is None


def test_audio_response_cancellation_before_worker_start_is_still_bounded(monkeypatch):
    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_GENERATION_TIMEOUT", 100.0)
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_DRAIN_TIMEOUT", 0.03)
    # Before audio_started there is nobody to receive the cancel, so this window is the
    # teardown budget, not the drain: a prefill pass is slow, not unresponsive. Still bounded.
    monkeypatch.setattr(orchestrator_module, "_AUDIO_CANCEL_TEARDOWN_TIMEOUT", 0.05)
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
        lambda _request_id, _cancel_event = None: (
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
    assert orchestrator._worker_reserved_for is None


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


def test_minimax_music_route_allows_its_official_frame_budget():
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        max_tokens = 10**310,
    )

    assert inference_route._tts_max_new_tokens(payload, audio_type = "minimax_music3") == 9000


@pytest.mark.parametrize(
    ("audio_type", "expected_max_tokens"),
    ((None, 8192), ("minimax_music3", 9000)),
)
def test_audio_worker_command_uses_the_model_token_budget(
    monkeypatch, audio_type, expected_max_tokens
):
    orchestrator = _bare_orchestrator()
    orchestrator.models["model"]["audio_type"] = audio_type
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: sent.append(cmd))

    def direct_reader(request_id, cancel_event = None):
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
    assert sent[0]["max_new_tokens"] == expected_max_tokens


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
        cancel_state.append(orchestrator._worker_reserved_for)
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

    assert cancel_state == ["audio generation is in progress"], (
        "timeout cancellation must occur under TTS exclusivity"
    )
    assert orchestrator._worker_reserved_for is None
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
        shutdown_state.append((orchestrator._worker_reserved_for, timeout))
        return True

    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", shutdown)

    with pytest.raises(RuntimeError, match = "Timeout waiting for audio generation"):
        orchestrator.generate_audio_response("hello")

    assert shutdown_state == [("audio generation is in progress", 0.03)]
    assert orchestrator._worker_reserved_for is None
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
    orchestrator._worker_reserved_for = "audio generation is in progress"

    assert orchestrator._start_dispatcher() is None
    assert orchestrator._dispatcher_thread is None

    orchestrator._worker_reserved_for = None
    try:
        assert orchestrator._start_dispatcher() is orchestrator._dispatcher_thread
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

    def direct_reader(request_id, cancel_event = None):
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
    while not orchestrator._worker_reserved_for and time.monotonic() < deadline:
        time.sleep(0.01)
    assert orchestrator._worker_reserved_for == "audio generation is in progress"
    assert sent == [], "TTS must not enqueue behind the active compare request"

    with orchestrator._mailbox_lock:
        orchestrator._mailboxes.pop("compare")
    thread.join(timeout = 3)

    assert thread.is_alive() is False
    assert result["value"] == (b"RIFFfake", 24000)
    assert sent and sent[0]["type"] == "generate_audio"
    assert orchestrator._worker_reserved_for is None


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
    while not orchestrator._worker_reserved_for and time.monotonic() < deadline:
        time.sleep(0.01)
    assert orchestrator._worker_reserved_for == "audio generation is in progress"

    caller_cancel.set()
    thread.join(timeout = 2)

    assert thread.is_alive() is False
    assert "cancel" in str(error["value"]).lower()
    assert cancel_calls == []
    assert set(orchestrator._mailboxes) == {"compare"}
    assert orchestrator._worker_reserved_for is None


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
    # core.inference.inference pulls the training stack in, which the backend-test job does
    # not install; without this the whole job fails on ModuleNotFoundError rather than
    # reporting a skip for a test that cannot run there.
    pytest.importorskip("peft")
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


@pytest.mark.parametrize(("stop_type", "finish_reason"), (("limit", "length"), ("eos", "stop")))
def test_gguf_speech_cut_at_max_tokens_is_reported(monkeypatch, stop_type, finish_reason):
    import core.inference.llama_cpp as llama_cpp

    sent = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"content": "<custom_token_10>", "stop_type": stop_type}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
            **_kwargs,
        ):
            sent.update(json)
            return _Resp()

    class _Llama(llama_cpp.LlamaCppBackend):
        is_loaded = True
        _is_audio = True
        _audio_type = "snac"
        model_identifier = "unsloth/orpheus-3b-0.1-ft-GGUF"
        base_url = "http://127.0.0.1:8080"
        _auth_headers: dict = {}
        _hf_variant = None
        _gguf_path = None

    async def _noop_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(llama_cpp.httpx, "Client", _Client)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_codec_mgr",
        types.SimpleNamespace(
            decode = lambda *_args, **_kwargs: (b"RIFFfake", 24000),
            has_codec = lambda _audio_type: True,
        ),
    )
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama.__new__(_Llama))
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    monkeypatch.setattr(inference_route, "_persist_tts_clip", lambda *_args: None)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "A long paragraph."}],
        max_tokens = 2048,
    )

    response = asyncio.run(
        inference_route.generate_audio(payload, request = None, current_subject = "t")
    )

    assert sent["n_predict"] == 2048
    assert json.loads(response.body)["choices"][0]["finish_reason"] == finish_reason


@pytest.mark.parametrize(("last_token", "truncated"), ((4242, True), (128258, False)))
def test_snac_speech_cut_at_max_tokens_is_reported(monkeypatch, last_token, truncated):
    pytest.importorskip("peft")
    import torch
    from core.inference.inference import InferenceBackend

    class _Tokenizer:
        def __call__(self, text, return_tensors):
            return types.SimpleNamespace(input_ids = torch.tensor([[1, 2, 3]]))

    class _Model:
        device = torch.device("cpu")

        def generate(self, input_ids, max_new_tokens, **_kwargs):
            codes = [4242] * (max_new_tokens - 1) + [last_token]
            return torch.cat([input_ids, torch.tensor([codes])], dim = 1)

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "tts"
    backend._generation_lock = threading.Lock()
    backend.models = {"tts": {"audio_type": "snac", "model": _Model(), "tokenizer": _Tokenizer()}}
    backend._audio_codec_manager = types.SimpleNamespace(
        decode_snac = lambda *_args: (b"RIFFfake", 24000)
    )

    worker_responses = queue.Queue()
    _handle_generate_audio(
        backend,
        {"request_id": "audio-1", "text": "A long paragraph.", "max_new_tokens": 14},
        worker_responses,
        threading.Event(),
    )
    done = worker_responses.get_nowait()

    orchestrator = _bare_orchestrator()
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orchestrator, "_send_cmd", lambda cmd: None)

    def direct_reader(request_id, cancel_event = None):
        return (
            lambda *, timeout: {**done, "request_id": request_id},
            lambda **_kwargs: None,
            lambda: None,
        )

    monkeypatch.setattr(orchestrator, "_direct_reader", direct_reader)
    holder = {}
    orchestrator.generate_audio_response("A long paragraph.", stats_holder = holder)

    assert holder["stats"]["truncated"] is truncated


@pytest.mark.parametrize("audio_type", ("bicodec", "dac"))
@pytest.mark.parametrize(("last_token", "truncated"), ((4242, True), (7, False)))
def test_token_codec_speech_cut_at_max_tokens_is_reported(audio_type, last_token, truncated):
    pytest.importorskip("peft")
    import torch
    from core.inference.inference import InferenceBackend

    class _Inputs(dict):
        def __init__(self):
            super().__init__(input_ids = torch.tensor([[1, 2, 3]]))
            self.input_ids = self["input_ids"]

        def to(self, _device):
            return self

    class _Tokenizer:
        eos_token_id = 7
        pad_token_id = 0

        def __call__(self, texts, return_tensors):
            return _Inputs()

        def batch_decode(self, tokens, skip_special_tokens):
            return [""]

    class _Model:
        device = torch.device("cpu")
        dtype = torch.float32
        generation_config = types.SimpleNamespace(eos_token_id = 7)

        def generate(self, input_ids, max_new_tokens, **_kwargs):
            codes = [4242] * (max_new_tokens - 1) + [last_token]
            return torch.cat([input_ids, torch.tensor([codes])], dim = 1)

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "tts"
    backend._generation_lock = threading.Lock()
    backend.models = {
        "tts": {"audio_type": audio_type, "model": _Model(), "tokenizer": _Tokenizer()}
    }
    backend._audio_codec_manager = types.SimpleNamespace(
        decode_bicodec = lambda *_args: (b"RIFFfake", 16000),
        decode_dac = lambda *_args: (b"RIFFfake", 24000),
    )
    backend._patch_repetition_penalty_processor = lambda: None

    backend.generate_audio_response("A long paragraph.", max_new_tokens = 14)

    assert backend.last_generation_stats["truncated"] is truncated


def test_a_gpu_codec_left_to_cpu_only_slots_moves_to_the_cpu(monkeypatch):
    """The voice slot loaded the shared codec on the GPU; once it unloaded, a zero-VRAM chat
    slot kept that codec alive, holding VRAM training admission does not count."""
    import core.inference.audio_codecs as audio_codecs
    from core.inference.llama_cpp import LlamaCppBackend

    loaded, unloaded = [], []

    class _Manager:
        def __init__(self):
            self._codec_devices = {}

        def load_codec(
            self,
            audio_type,
            device,
            model_repo_path = None,
        ):
            self._codec_devices[audio_type] = device
            loaded.append((audio_type, device, model_repo_path))

        def unload(self):
            unloaded.append(dict(self._codec_devices))

    class _Slot(LlamaCppBackend):
        def __init__(self, zero_vram):
            self._owns_codec = False
            self._zero_vram = zero_vram
            self._audio_type = "snac"
            self._codec_repo_path = None

        @property
        def holds_no_vram(self):
            return self._zero_vram

    monkeypatch.setattr(audio_codecs, "AudioCodecManager", _Manager)
    monkeypatch.setattr(LlamaCppBackend, "_codec_owners", 0)
    monkeypatch.setattr(LlamaCppBackend, "_codec_holders", set())
    gpu_mgr = _Manager()
    gpu_mgr.load_codec("snac", "cuda")
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", gpu_mgr)
    voice, cpu_chat, gpu_chat = _Slot(False), _Slot(True), _Slot(False)
    for slot in (voice, cpu_chat, gpu_chat):
        slot._claim_audio_codec()
    loaded.clear()

    voice._unload_audio_codec()  # a GPU slot still holds it: nothing moves
    assert LlamaCppBackend._codec_mgr is gpu_mgr and unloaded == []

    gpu_chat._unload_audio_codec()  # only the zero-VRAM slot is left
    assert loaded == [("snac", "cpu", None)]
    assert unloaded == [{"snac": "cuda"}]
    assert LlamaCppBackend._codec_mgr._codec_devices == {"snac": "cpu"}
    assert LlamaCppBackend._codec_owners == 1


def test_two_slots_never_load_the_shared_codec_at_once(monkeypatch):
    """The voice slot and the chat slot load under different backend locks, so both could see no
    codec and load it on the GPU together, doubling the peak allocation."""
    import threading
    import time

    import core.inference.audio_codecs as audio_codecs
    from core.inference.llama_cpp import LlamaCppBackend

    active, peak = [0], [0]
    guard = threading.Lock()

    class _Manager:
        def __init__(self):
            self._codec_devices = {}

        def load_codec(
            self,
            audio_type,
            device,
            model_repo_path = None,
        ):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with guard:
                active[0] -= 1
            self._codec_devices[audio_type] = device

    class _Slot(LlamaCppBackend):
        def __init__(self):
            self._owns_codec = False

        @property
        def holds_no_vram(self):
            return True

    monkeypatch.setattr(audio_codecs, "AudioCodecManager", _Manager)
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", None)
    monkeypatch.setattr(LlamaCppBackend, "_codec_owners", 0)
    monkeypatch.setattr(LlamaCppBackend, "_codec_holders", set())
    slots = [_Slot(), _Slot()]
    threads = [threading.Thread(target = s.init_audio_codec, args = ("snac",)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert peak[0] == 1
    assert LlamaCppBackend._codec_owners == 2


def test_a_slot_unloading_waits_for_the_other_slots_codec_load(monkeypatch):
    """Teardown ran under the owner lock only, so it could free the manager between the other
    slot's codec load and its claim, leaving that slot on an unloaded codec."""
    import threading
    import time

    import core.inference.audio_codecs as audio_codecs
    from core.inference.llama_cpp import LlamaCppBackend

    events = []

    class _Manager:
        def __init__(self):
            self._codec_devices = {}

        def load_codec(
            self,
            audio_type,
            device,
            model_repo_path = None,
        ):
            events.append("load start")
            time.sleep(0.3)
            self._codec_devices[audio_type] = device
            events.append("load end")

        def unload(self):
            events.append("unload")

    class _Slot(LlamaCppBackend):
        def __init__(self):
            self._owns_codec = False

        @property
        def holds_no_vram(self):
            return True

    monkeypatch.setattr(audio_codecs, "AudioCodecManager", _Manager)
    monkeypatch.setattr(LlamaCppBackend, "_codec_owners", 0)
    monkeypatch.setattr(LlamaCppBackend, "_codec_holders", set())
    leaving, arriving = _Slot(), _Slot()
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", _Manager())
    leaving._claim_audio_codec()

    loader = threading.Thread(target = arriving.init_audio_codec, args = ("snac",))
    loader.start()
    time.sleep(0.1)  # inside load_codec, before the claim
    leaving._unload_audio_codec()
    loader.join(5)

    assert "unload" not in events
    assert LlamaCppBackend._codec_mgr is not None and LlamaCppBackend._codec_owners == 1
