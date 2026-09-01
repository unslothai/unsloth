# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The out-of-process Transformers dictation engine.

The engine moved into a spawn child because an accelerator context is never
returned while the process holding it lives, so the backend must not be the
process that takes one. These cover both halves: what the child does with a
command, and what the parent-side handle does with a child that answers late,
dies, or is cancelled.
"""

import queue
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import core.inference.stt_transformers_worker as worker_module
from core.inference.stt_sidecar import (
    SttLoadCancelledError,
    SttModelNotDownloadedError,
    SttTranscriptionCancelledError,
)
from core.inference.stt_transformers_worker import SttWorkerError, WhisperWorker

# signal.Signals is per platform, so Windows reads -9 back as a number (TerminateProcess maps to -SIGTERM).
_SIGKILL_TEXT = "SIGKILL" if hasattr(signal, "SIGKILL") else "SIG9"




class _FakeTensor:
    def __init__(self, dtype = None) -> None:
        self.dtype = dtype
        self.moved_to = []

    def to(self, value):
        self.moved_to.append(value)
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.seen_audio = None
        self.seen_rate = None
        self.features = _FakeTensor()

    def __call__(
        self,
        audio,
        sampling_rate = None,
        return_tensors = None,
    ):
        self.seen_audio = audio
        self.seen_rate = sampling_rate
        return SimpleNamespace(input_features = self.features)

    def batch_decode(self, _generated, **_kwargs):
        return ["hello"]


class _FakeModel:
    def __init__(self, dtype = "float16") -> None:
        self.dtype = dtype
        self.device = "cuda"
        self.generation_config = SimpleNamespace(is_multilingual = True)
        self.generate_kwargs = None
        self.moved_to = None
        self.evaluated = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def generate(self, _features, **kwargs):
        self.generate_kwargs = kwargs
        return [[1]]


class _FakeProcess:
    """Stands in for mp.Process; alive until something ends it."""

    def __init__(
        self,
        pid = 4242,
        alive = True,
    ) -> None:
        self.pid = pid
        self._alive = alive
        self.exitcode = None
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._alive

    def join(self, _timeout = None):
        return None

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9


def _wired_worker(process = None):
    """A handle wired to in-process queues, so no child is ever spawned."""
    handle = WhisperWorker()
    handle._process = process if process is not None else _FakeProcess()
    handle._cmd_queue = queue.Queue()
    handle._resp_queue = queue.Queue()
    handle._cancel_event = threading.Event()
    return handle


def _install_fake_transformers(
    monkeypatch,
    model = None,
    processor = None,
):
    fake_model = model if model is not None else _FakeModel()
    fake_processor = processor if processor is not None else _FakeProcessor()
    calls = []

    class FakeWhisperForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return fake_model

    class FakeWhisperProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", path, kwargs))
            return fake_processor

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            WhisperForConditionalGeneration = FakeWhisperForConditionalGeneration,
            WhisperProcessor = FakeWhisperProcessor,
            StoppingCriteriaList = list,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(
            float16 = "float16",
            float32 = "float32",
            device = lambda value: value,
            no_grad = _NoGrad,
        ),
    )
    return calls, fake_model, fake_processor




def test_child_loads_from_the_model_hub_cache_without_an_implicit_download(monkeypatch):
    calls, model, _processor = _install_fake_transformers(monkeypatch)

    worker_module.load_whisper("/cached/model", "cuda", "float16")

    assert {(kind, path) for kind, path, _ in calls} == {
        ("processor", "/cached/model"),
        ("model", "/cached/model"),
    }
    assert all(kwargs.get("local_files_only") is True for _, _, kwargs in calls)
    # The weight load forces safetensors so a pickle checkpoint cannot execute.
    model_kwargs = next(kwargs for kind, _, kwargs in calls if kind == "model")
    assert model_kwargs.get("use_safetensors") is True
    assert model_kwargs.get("torch_dtype") == "float16"
    assert model.moved_to == "cuda"
    assert model.evaluated is True


def test_child_load_stops_at_the_first_checkpoint_after_a_cancel(monkeypatch):
    _calls, model, _processor = _install_fake_transformers(monkeypatch)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(SttLoadCancelledError):
        worker_module.load_whisper("/cached/model", "cuda", "float16", cancel_event)

    assert model.moved_to is None


def test_child_falls_back_to_float32_for_an_unknown_dtype_name(monkeypatch):
    calls, _model, _processor = _install_fake_transformers(monkeypatch)

    worker_module.load_whisper("/cached/model", "cpu", "bfloat9")

    model_kwargs = next(kwargs for kind, _, kwargs in calls if kind == "model")
    assert model_kwargs.get("torch_dtype") == "float32"




def test_child_feeds_decoded_pcm_and_matches_the_model_dtype(monkeypatch):
    _calls, model, processor = _install_fake_transformers(monkeypatch)
    pcm = np.arange(4, dtype = np.float32).tobytes()

    text = worker_module.transcribe_window(
        model, processor, pcm, {"task": "transcribe", "num_beams": 5}
    )

    assert text == "hello"
    assert processor.seen_rate == 16000
    assert np.array_equal(processor.seen_audio, np.arange(4, dtype = np.float32))
    assert processor.features.moved_to == ["cuda", "float16"]
    assert model.generate_kwargs == {"task": "transcribe", "num_beams": 5}


def test_child_only_installs_stopping_criteria_for_a_cancellable_request(monkeypatch):
    _calls, model, processor = _install_fake_transformers(monkeypatch)
    cancel_event = threading.Event()
    pcm = np.zeros(4, dtype = np.float32).tobytes()

    worker_module.transcribe_window(model, processor, pcm, {}, cancel_event)
    criteria = model.generate_kwargs["stopping_criteria"]

    assert criteria[0]() is False
    cancel_event.set()
    assert criteria[0]() is True

    worker_module.transcribe_window(model, processor, pcm, {})
    assert "stopping_criteria" not in model.generate_kwargs




def _run_child(
    monkeypatch,
    commands,
    *,
    load = None,
    transcribe = None,
):
    """Drive run_stt_worker over in-process queues and collect its responses.

    The bootstrap handshake is asserted here and dropped, so each test reads the
    answers to its own commands.
    """
    cmd_queue: queue.Queue = queue.Queue()
    resp_queue: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    if load is not None:
        monkeypatch.setattr(worker_module, "load_whisper", load)
    if transcribe is not None:
        monkeypatch.setattr(worker_module, "transcribe_window", transcribe)
    for command in commands:
        cmd_queue.put(command)
    ready_event = threading.Event()
    thread = threading.Thread(
        target = worker_module.run_stt_worker,
        kwargs = {
            "cmd_queue": cmd_queue,
            "resp_queue": resp_queue,
            "cancel_event": cancel_event,
            "ready_event": ready_event,
            "config": {},
        },
        daemon = True,
    )
    thread.start()
    thread.join(timeout = 10)
    assert thread.is_alive() is False
    assert ready_event.is_set() is True
    responses = []
    while not resp_queue.empty():
        responses.append(resp_queue.get_nowait())
    return responses, cancel_event


def test_child_reports_the_loaded_model_then_transcribes_then_exits(monkeypatch):
    model = _FakeModel()
    model.generation_config = SimpleNamespace(is_multilingual = False)
    responses, _cancel = _run_child(
        monkeypatch,
        [
            {
                "type": "load",
                "snapshot_path": "/cached/model",
                "device": "cuda",
                "dtype": "float16",
            },
            {"type": "transcribe", "audio": b"", "generate_kwargs": {}, "cancellable": False},
            {"type": "shutdown"},
        ],
        load = lambda *_args, **_kwargs: (model, _FakeProcessor()),
        transcribe = lambda *_args, **_kwargs: "hello",
    )

    assert responses == [
        {"type": "loaded", "device": "cuda", "is_multilingual": False},
        {"type": "text", "text": "hello"},
        {"type": "shutdown_ack"},
    ]


def test_child_exits_after_a_failed_load_so_a_half_taken_context_goes_with_it(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("out of memory")

    responses, _cancel = _run_child(
        monkeypatch,
        [
            {
                "type": "load",
                "snapshot_path": "/cached/model",
                "device": "cuda",
                "dtype": "float16",
            },
            {"type": "transcribe", "audio": b"", "generate_kwargs": {}, "cancellable": False},
        ],
        load = boom,
    )

    assert responses == [{"type": "error", "kind": "RuntimeError", "error": "out of memory"}]


def test_child_survives_a_failed_transcription_and_keeps_the_model(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("bad audio")

    responses, _cancel = _run_child(
        monkeypatch,
        [
            {"type": "load", "snapshot_path": "/cached/model", "device": "cpu", "dtype": "float32"},
            {"type": "transcribe", "audio": b"", "generate_kwargs": {}, "cancellable": False},
            {"type": "shutdown"},
        ],
        load = lambda *_args, **_kwargs: (_FakeModel(), _FakeProcessor()),
        transcribe = boom,
    )

    assert [response["type"] for response in responses] == ["loaded", "error", "shutdown_ack"]
    assert responses[1]["error"] == "bad audio"


def test_child_reports_a_cancelled_generation_rather_than_partial_text(monkeypatch):
    def stop_early(
        _model,
        _processor,
        _pcm,
        _kwargs,
        cancel_event = None,
    ):
        cancel_event.set()
        return "half a sen"

    responses, _cancel = _run_child(
        monkeypatch,
        [
            {"type": "load", "snapshot_path": "/cached/model", "device": "cpu", "dtype": "float32"},
            {"type": "transcribe", "audio": b"", "generate_kwargs": {}, "cancellable": True},
            {"type": "shutdown"},
        ],
        load = lambda *_args, **_kwargs: (_FakeModel(), _FakeProcessor()),
        transcribe = stop_early,
    )

    assert responses[1]["kind"] == "SttTranscriptionCancelledError"


def test_child_answers_an_unknown_command_instead_of_dropping_it(monkeypatch):
    responses, _cancel = _run_child(
        monkeypatch,
        [{"type": "explode"}, {"type": "shutdown"}],
    )

    assert responses[0]["type"] == "error"
    assert "explode" in responses[0]["error"]




def test_a_local_cache_miss_crosses_as_a_not_downloaded_error():
    class LocalEntryNotFoundError(RuntimeError):
        pass

    response = worker_module._error_response(LocalEntryNotFoundError("not cached"))

    assert response["kind"] == "SttModelNotDownloadedError"
    with pytest.raises(SttModelNotDownloadedError):
        worker_module._raise_worker_error(response)


def test_cancellation_keeps_its_class_across_the_process_boundary():
    response = worker_module._error_response(
        SttTranscriptionCancelledError("Transcription cancelled.")
    )

    with pytest.raises(SttTranscriptionCancelledError, match = "cancelled"):
        worker_module._raise_worker_error(response)


def test_an_unknown_failure_arrives_as_a_worker_error_carrying_its_message():
    # The exception object is never sent: a torch error that will not pickle costs the caller its timeout.
    response = worker_module._error_response(TypeError("weird"))

    assert response == {"type": "error", "kind": "TypeError", "error": "weird"}
    with pytest.raises(SttWorkerError, match = "weird"):
        worker_module._raise_worker_error(response)




def test_handle_sends_one_window_and_returns_its_text():
    handle = _wired_worker()
    handle._resp_queue.put({"type": "text", "text": "hello"})

    text = handle.transcribe_window(b"\x00\x00\x00\x00", {"num_beams": 1})

    assert text == "hello"
    command = handle._cmd_queue.get_nowait()
    assert command["type"] == "transcribe"
    assert command["generate_kwargs"] == {"num_beams": 1}
    assert command["cancellable"] is False


def test_handle_reports_a_dead_child_instead_of_waiting_out_its_timeout():
    process = _FakeProcess(alive = False)
    process.exitcode = -9
    handle = _wired_worker(process)

    with pytest.raises(SttWorkerError, match = _SIGKILL_TEXT):
        handle.transcribe_window(b"", {})


def test_handle_kills_a_child_that_stops_answering():
    process = _FakeProcess()
    handle = _wired_worker(process)

    with pytest.raises(SttWorkerError, match = "stopped responding"):
        handle._await("text", 0.0, None, "transcribe")

    assert process.killed or process.terminated


def test_handle_mirrors_a_request_cancel_into_the_child():
    handle = _wired_worker()
    cancel_event = threading.Event()
    cancel_event.set()

    def answer_once():
        time.sleep(0.2)
        handle._resp_queue.put(
            {
                "type": "error",
                "kind": "SttTranscriptionCancelledError",
                "error": "Transcription cancelled.",
            }
        )

    thread = threading.Thread(target = answer_once, daemon = True)
    thread.start()
    with pytest.raises(SttTranscriptionCancelledError):
        handle._await("text", 30.0, cancel_event, "transcribe")
    thread.join(timeout = 5)

    assert handle._cancel_event.is_set()


def test_a_cancelled_load_that_never_answers_is_killed_rather_than_waited_on(monkeypatch):
    monkeypatch.setattr(worker_module, "_CANCEL_GRACE_SECONDS", 0.0)
    process = _FakeProcess()
    handle = _wired_worker(process)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(SttLoadCancelledError):
        handle._await("loaded", 30.0, cancel_event, "load")

    assert handle.is_alive() is False


def test_the_cancel_grace_is_not_followed_by_a_second_shutdown_wait(monkeypatch):
    # The grace IS the graceful shutdown: a child too busy inside from_pretrained to read the cancel
    # event will not read a shutdown command either, and another wait doubles the 10s stall.
    monkeypatch.setattr(worker_module, "_CANCEL_GRACE_SECONDS", 0.0)
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Recording(_FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.joins = []

        def join(self, timeout = None):
            self.joins.append(timeout)

    process = _Recording()
    handle = _wired_worker(process)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(SttLoadCancelledError):
        handle._await("loaded", 30.0, cancel_event, "load")

    assert worker_module._SHUTDOWN_TIMEOUT_SECONDS not in process.joins
    assert process.terminated is True
    assert handle.is_alive() is False


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("load", SttLoadCancelledError), ("transcribe", SttTranscriptionCancelledError)],
)
def test_a_cancel_that_lands_near_the_command_timeout_keeps_its_cancellation(
    monkeypatch, phase, expected
):
    # A cancel in the last seconds of the timeout is still a cancellation: the caller is owed 409 or 499.
    monkeypatch.setattr(worker_module, "_CANCEL_GRACE_SECONDS", 30.0)
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Recording(_FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.joins = []

        def join(self, timeout = None):
            self.joins.append(timeout)

    process = _Recording()
    handle = _wired_worker(process)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(expected):
        handle._await("text" if phase == "transcribe" else "loaded", 0.0, cancel_event, phase)

    assert worker_module._SHUTDOWN_TIMEOUT_SECONDS not in process.joins
    assert handle.is_alive() is False


def test_closing_a_handle_normally_still_asks_the_child_to_exit_first(monkeypatch):
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Recording(_FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.joins = []

        def join(self, timeout = None):
            self.joins.append(timeout)
            self._alive = False

    process = _Recording()
    handle = _wired_worker(process)
    cmd_queue = handle._cmd_queue

    handle.close()

    assert cmd_queue.get_nowait() == {"type": "shutdown"}
    assert process.joins[0] == worker_module._SHUTDOWN_TIMEOUT_SECONDS
    assert process.terminated is False


def test_closing_the_handle_ends_the_child_and_drops_its_pid(monkeypatch):
    forgotten = []
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda pid: forgotten.append(pid))
    process = _FakeProcess()
    handle = _wired_worker(process)

    handle.close()

    assert forgotten == [4242]
    assert handle.is_alive() is False
    assert handle._cmd_queue is None


def test_a_child_that_survives_terminate_and_kill_keeps_its_pid_and_handle(monkeypatch):
    # A child wedged in a driver call outlives SIGKILL and holds its memory, so its pid must be remembered.
    forgotten = []
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda pid: forgotten.append(pid))

    class _Unkillable(_FakeProcess):
        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    process = _Unkillable()
    handle = _wired_worker(process)

    closed = handle.close()

    assert forgotten == []
    assert closed is False
    assert handle._process is process
    assert handle.is_alive() is True
    assert handle._cmd_queue is not None


def test_a_child_that_outlived_a_cancelled_command_marks_its_handle_unusable(monkeypatch):
    # The handle is kept for its memory but must say it is spent: the child answers no later command,
    # its terminate leaves the queues liable to corruption, and the cancel is raised over close().
    monkeypatch.setattr(worker_module, "_CANCEL_GRACE_SECONDS", 0.0)
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Unkillable(_FakeProcess):
        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    handle = _wired_worker(_Unkillable())
    assert handle.survived_kill is False
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(SttTranscriptionCancelledError):
        handle._await("text", 30.0, cancel_event, "transcribe")

    assert handle.is_alive() is True
    assert handle.survived_kill is True


def test_a_handle_whose_child_did_exit_is_still_usable(monkeypatch):
    # The flag is only for a child that outlived both signals; an ordinary close must not retire a handle.
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    handle = _wired_worker()

    assert handle.close() is True
    assert handle.survived_kill is False


def test_closing_a_handle_that_ignores_shutdown_escalates_to_a_kill(monkeypatch):
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Stubborn(_FakeProcess):
        def terminate(self):
            self.terminated = True

    process = _Stubborn()
    handle = _wired_worker(process)

    handle.close()

    assert process.terminated is True
    assert process.killed is True




class _RefusingProcess:
    """A child that cannot be created: a sandbox, or a frozen POSIX build."""

    def __init__(self, error) -> None:
        self.pid = None
        self.exitcode = None
        self._error = error

    def is_alive(self):
        return False

    def start(self):
        raise self._error

    def join(self, _timeout = None):
        return None


class _RefusingContext:
    def __init__(self, error = None) -> None:
        self.error = error or PermissionError("spawn is not permitted here")

    def Queue(self):
        return queue.Queue()

    def Event(self):
        return threading.Event()

    def Process(self, **_kwargs):
        return _RefusingProcess(self.error)


def test_dictation_still_loads_and_transcribes_when_no_child_can_be_started(monkeypatch):
    # This may only move work out of the backend, never remove a working configuration: dictation survives.
    from core.inference.stt_sidecar import WhisperSttSidecar

    monkeypatch.setattr(worker_module, "_CTX", _RefusingContext())
    _calls, _model, _processor = _install_fake_transformers(monkeypatch)

    engine = WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
        "/cached/model", "cpu", "float32", threading.Event()
    )

    assert isinstance(engine, worker_module.InProcessWhisperEngine)
    assert engine.device == "cpu"
    assert engine.is_alive() is True
    assert engine.transcribe_window(np.zeros(4, dtype = np.float32).tobytes(), {}) == "hello"


def test_a_spawn_failure_on_an_accelerator_leaves_the_cpu_retry_to_the_sidecar(monkeypatch):
    # An in-process load takes the context this module avoids, so the fallback is CPU and the sidecar retries first.
    from core.inference.stt_sidecar import WhisperSttSidecar

    monkeypatch.setattr(worker_module, "_CTX", _RefusingContext())
    monkeypatch.setattr(
        worker_module,
        "load_whisper",
        lambda *_args, **_kwargs: pytest.fail("no in-process load on an accelerator"),
    )

    with pytest.raises(worker_module.SttWorkerSpawnError, match = "not permitted"):
        WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
            "/cached/model", "cuda", "float16", threading.Event()
        )


class _StillbornProcess:
    """A child that starts but whose fresh interpreter never comes up.

    A frozen POSIX build re-runs its own binary rather than an interpreter, so
    start() returns and the child is gone before it can read a command.
    """

    def __init__(self, exitcode = 1) -> None:
        self.pid = 4243
        self.exitcode = None
        self._exitcode = exitcode

    def start(self):
        self.exitcode = self._exitcode

    def is_alive(self):
        return False

    def join(self, _timeout = None):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass


class _StillbornContext:
    def __init__(self, exitcode = 1) -> None:
        self.exitcode = exitcode

    def Queue(self):
        return queue.Queue()

    def Event(self):
        return threading.Event()

    def Process(self, **_kwargs):
        return _StillbornProcess(self.exitcode)


def test_a_child_that_never_bootstraps_reads_as_a_host_that_cannot_spawn(monkeypatch):
    # start() succeeding says only that exec worked, and a child that dies before answering took no device.
    from core.inference.stt_sidecar import WhisperSttSidecar

    monkeypatch.setattr(worker_module, "_CTX", _StillbornContext())
    _calls, _model, _processor = _install_fake_transformers(monkeypatch)

    engine = WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
        "/cached/model", "cpu", "float32", threading.Event()
    )

    assert isinstance(engine, worker_module.InProcessWhisperEngine)
    assert engine.device == "cpu"


def test_a_child_killed_by_a_signal_keeps_its_crash_instead_of_falling_back(monkeypatch):
    # A child the box killed under memory pressure bootstrapped fine, so spawn works and the backend would repeat it.
    monkeypatch.setattr(worker_module, "_CTX", _StillbornContext(exitcode = -9))
    monkeypatch.setattr(
        worker_module,
        "load_whisper",
        lambda *_args, **_kwargs: pytest.fail("no in-process load after a real crash"),
    )

    handle = WhisperWorker()
    with pytest.raises(SttWorkerError, match = _SIGKILL_TEXT) as caught:
        handle.start("/cached/model", "cpu", "float32")

    assert isinstance(caught.value, worker_module.SttWorkerSpawnError) is False


class _NativeCrashProcess:
    """A child that bootstraps and then dies inside the native model load.

    Runs the real child entrypoint, whose load neither returns nor reports
    anything, exactly as a fault in native code does not; the process is then
    simply gone. Its exit code is positive because Windows has no signals to
    report a fault with (0xC0000005 reads as 3221225477), which is what a child
    that never bootstrapped looks like from the exit code alone.
    """

    def __init__(self, kwargs, faulted: threading.Event) -> None:
        self.pid = 4244
        self.exitcode = None
        self._kwargs = kwargs
        self._faulted = faulted

    def start(self):
        thread = threading.Thread(
            target = worker_module.run_stt_worker,
            kwargs = self._kwargs,
            daemon = True,
        )
        thread.start()

    def is_alive(self):
        if self._faulted.is_set():
            self.exitcode = 3221225477
            return False
        return True

    def join(self, _timeout = None):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass


class _NativeCrashContext:
    """Spawns a child that comes up and then faults in the model load."""

    def __init__(self, faulted: threading.Event) -> None:
        self._faulted = faulted
        self._queues: list = []

    def Queue(self):
        made: queue.Queue = queue.Queue()
        self._queues.append(made)
        return made

    def Event(self):
        return threading.Event()

    def Process(self, **kwargs):
        # Forward what start() passed, ready_event included: rebuilding the kwargs would drop the readiness signal.
        process = _NativeCrashProcess(dict(kwargs.get("kwargs") or {}), self._faulted)
        self._process = process
        return process


def _fault_in_the_native_load(monkeypatch, faulted: threading.Event, forever: threading.Event):
    def _fault(*_args, **_kwargs):
        faulted.set()
        forever.wait(30)
        raise AssertionError("the crashed child was resumed")

    monkeypatch.setattr(worker_module, "load_whisper", _fault)


def test_a_child_that_crashed_in_the_load_is_not_read_as_a_host_that_cannot_spawn(monkeypatch):
    # A native crash under the load kills the child with a positive exit code on Windows, and that
    # child bootstrapped, so reading it as a host that cannot spawn would repeat the native load.
    faulted = threading.Event()
    forever = threading.Event()
    monkeypatch.setattr(worker_module, "_CTX", _NativeCrashContext(faulted))
    _fault_in_the_native_load(monkeypatch, faulted, forever)

    handle = WhisperWorker()
    try:
        with pytest.raises(SttWorkerError) as caught:
            handle.start("/cached/model", "cpu", "float32")
    finally:
        forever.set()

    assert faulted.is_set()
    assert isinstance(caught.value, worker_module.SttWorkerSpawnError) is False


def test_a_crash_in_the_child_load_is_never_repeated_inside_the_backend(monkeypatch):
    # The fallback exists for a host that cannot bring a child up: a crashing load crashes the backend too.
    from core.inference.stt_sidecar import WhisperSttSidecar

    faulted = threading.Event()
    forever = threading.Event()
    monkeypatch.setattr(worker_module, "_CTX", _NativeCrashContext(faulted))
    _fault_in_the_native_load(monkeypatch, faulted, forever)
    monkeypatch.setattr(
        worker_module.InProcessWhisperEngine,
        "start",
        lambda *_args, **_kwargs: pytest.fail("no in-process load after a crash in the child"),
    )

    try:
        with pytest.raises(SttWorkerError):
            WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
                "/cached/model", "cpu", "float32", threading.Event()
            )
    finally:
        forever.set()


def test_the_child_says_it_is_ready_before_it_touches_a_command(monkeypatch):
    # The handshake separates a host that cannot spawn from a child that failed, so it precedes the load.
    cmd_queue: queue.Queue = queue.Queue()
    resp_queue: queue.Queue = queue.Queue()

    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(worker_module, "load_whisper", boom)
    cmd_queue.put(
        {"type": "load", "snapshot_path": "/cached/model", "device": "cpu", "dtype": "float32"}
    )
    ready_event = threading.Event()
    worker_module.run_stt_worker(
        cmd_queue = cmd_queue,
        resp_queue = resp_queue,
        cancel_event = threading.Event(),
        ready_event = ready_event,
        config = {},
    )

    assert ready_event.is_set() is True
    assert resp_queue.get_nowait()["kind"] == "RuntimeError"


def test_the_in_process_fallback_reports_the_checkpoint_language_support(monkeypatch):
    _calls, model, _processor = _install_fake_transformers(monkeypatch)
    model.generation_config = SimpleNamespace(is_multilingual = False)

    engine = worker_module.InProcessWhisperEngine()
    engine.start("/cached/model", "cpu", "float32")

    assert engine.generation_config.is_multilingual is False
    engine.close()
    assert engine.is_alive() is False


class _LosesTheReadyMessage(queue.Queue):
    """A response queue that drops the ready word, as a real one does.

    multiprocessing.Queue.put only hands the object to a feeder thread. A child
    that faults before that thread drains the buffer delivers nothing, and the
    load command is already queued when the child reaches get(), so it faults
    almost immediately: measured at 17 losses in 20 runs, against 0 for an
    Event. A thread queue.Queue delivers in the caller, which is why a queued
    handshake looks sound in tests and is not.
    """

    def put(self, item, *args, **kwargs):
        if isinstance(item, dict) and item.get("type") == "ready":
            return
        return super().put(item, *args, **kwargs)


class _LossyNativeCrashContext(_NativeCrashContext):
    def Queue(self):
        made = _LosesTheReadyMessage()
        self._queues.append(made)
        return made


def test_a_crashed_child_whose_ready_word_was_lost_is_still_not_read_as_a_bad_host(monkeypatch):
    # The child faulted in the native load with no ready reaching the backend; calling that a spawn failure repeats it.
    faulted = threading.Event()
    forever = threading.Event()
    monkeypatch.setattr(worker_module, "_CTX", _LossyNativeCrashContext(faulted))
    _fault_in_the_native_load(monkeypatch, faulted, forever)

    handle = WhisperWorker()
    try:
        with pytest.raises(SttWorkerError) as caught:
            handle.start("/cached/model", "cpu", "float32")
    finally:
        forever.set()

    assert faulted.is_set()
    assert isinstance(caught.value, worker_module.SttWorkerSpawnError) is False
