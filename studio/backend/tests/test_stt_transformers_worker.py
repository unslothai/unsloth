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


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Child: loading
# ---------------------------------------------------------------------------


def test_child_loads_from_the_model_hub_cache_without_an_implicit_download(monkeypatch):
    calls, model, _processor = _install_fake_transformers(monkeypatch)

    worker_module.load_whisper("/cached/model", "cuda", "float16")

    assert {(kind, path) for kind, path, _ in calls} == {
        ("processor", "/cached/model"),
        ("model", "/cached/model"),
    }
    # Never fetch weights implicitly; the Model Hub owns downloads.
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

    # Cancelled before the weights could reach the accelerator.
    assert model.moved_to is None


def test_child_falls_back_to_float32_for_an_unknown_dtype_name(monkeypatch):
    calls, _model, _processor = _install_fake_transformers(monkeypatch)

    worker_module.load_whisper("/cached/model", "cpu", "bfloat9")

    model_kwargs = next(kwargs for kind, _, kwargs in calls if kind == "model")
    assert model_kwargs.get("torch_dtype") == "float32"


# ---------------------------------------------------------------------------
# Child: transcription
# ---------------------------------------------------------------------------


def test_child_feeds_decoded_pcm_and_matches_the_model_dtype(monkeypatch):
    _calls, model, processor = _install_fake_transformers(monkeypatch)
    pcm = np.arange(4, dtype = np.float32).tobytes()

    text = worker_module.transcribe_window(
        model, processor, pcm, {"task": "transcribe", "num_beams": 5}
    )

    assert text == "hello"
    assert processor.seen_rate == 16000
    assert np.array_equal(processor.seen_audio, np.arange(4, dtype = np.float32))
    # to(device) then to(dtype): features must match the weights they meet.
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


# ---------------------------------------------------------------------------
# Child: command loop
# ---------------------------------------------------------------------------


def _run_child(
    monkeypatch,
    commands,
    *,
    load = None,
    transcribe = None,
):
    """Drive run_stt_worker over in-process queues and collect its responses."""
    cmd_queue: queue.Queue = queue.Queue()
    resp_queue: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    if load is not None:
        monkeypatch.setattr(worker_module, "load_whisper", load)
    if transcribe is not None:
        monkeypatch.setattr(worker_module, "transcribe_window", transcribe)
    for command in commands:
        cmd_queue.put(command)
    thread = threading.Thread(
        target = worker_module.run_stt_worker,
        kwargs = {
            "cmd_queue": cmd_queue,
            "resp_queue": resp_queue,
            "cancel_event": cancel_event,
            "config": {},
        },
        daemon = True,
    )
    thread.start()
    thread.join(timeout = 10)
    assert thread.is_alive() is False
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
        # The transcribe would be answered if the child stayed in its loop.
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
        cancel_event.set()  # what StoppingCriteria does to a running generate
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


# ---------------------------------------------------------------------------
# Error transport
# ---------------------------------------------------------------------------


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
    # The exception object is never sent: a torch error that will not pickle
    # would cost the caller its whole timeout instead of an error.
    response = worker_module._error_response(TypeError("weird"))

    assert response == {"type": "error", "kind": "TypeError", "error": "weird"}
    with pytest.raises(SttWorkerError, match = "weird"):
        worker_module._raise_worker_error(response)


# ---------------------------------------------------------------------------
# Parent handle
# ---------------------------------------------------------------------------


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

    with pytest.raises(SttWorkerError, match = "SIGKILL"):
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
        # The child sees the shared event and reports the cancellation itself.
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
    # from_pretrained reaches no checkpoint, and training is waiting for the memory.
    monkeypatch.setattr(worker_module, "_CANCEL_GRACE_SECONDS", 0.0)
    process = _FakeProcess()
    handle = _wired_worker(process)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(SttLoadCancelledError):
        handle._await("loaded", 30.0, cancel_event, "load")

    assert handle.is_alive() is False


def test_the_cancel_grace_is_not_followed_by_a_second_shutdown_wait(monkeypatch):
    # The grace IS the graceful shutdown: a child too busy inside from_pretrained to
    # read the cancel event is equally too busy to read a shutdown command, so giving
    # it another _SHUTDOWN_TIMEOUT_SECONDS would block the waiting training run for
    # twice the documented 10 seconds.
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

    # No graceful join, no shutdown command queued for a child that cannot read it.
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
    # A cancel arriving in the last seconds of the load or transcribe timeout is
    # still a cancellation: the caller is owed the 409 or the 499, not a 500 for
    # a worker that "stopped responding", and not another full shutdown wait.
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
            self._alive = False  # an idle child consumes the shutdown and exits

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
    # A child wedged in a driver call outlives SIGKILL and still holds its
    # accelerator memory. Forgetting its pid would leave terminate_all and the
    # next startup sweep nothing to find it by.
    forgotten = []
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda pid: forgotten.append(pid))

    class _Unkillable(_FakeProcess):
        def terminate(self):
            self.terminated = True  # neither signal reaches it

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


def test_closing_a_handle_that_ignores_shutdown_escalates_to_a_kill(monkeypatch):
    monkeypatch.setattr("utils.process_lifetime.forget_pid", lambda _pid: None)

    class _Stubborn(_FakeProcess):
        def terminate(self):
            self.terminated = True  # ignores it, unlike _FakeProcess

    process = _Stubborn()
    handle = _wired_worker(process)

    handle.close()

    assert process.terminated is True
    assert process.killed is True


# ---------------------------------------------------------------------------
# Hosts that cannot spawn
# ---------------------------------------------------------------------------


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
    # These changes may only move work out of the backend, never remove a
    # working configuration: a host that forbids spawn had dictation before.
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
    # An in-process load takes the context this module exists to avoid, so the
    # fallback is CPU only; the accelerator attempt must reach the sidecar's own
    # CPU retry first rather than downgrade the user here.
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
    # start() succeeding says only that the exec worked. A child that dies before
    # answering anything took no device and named no model, so it must reach the
    # in-process fallback rather than the same failure on a second child.
    from core.inference.stt_sidecar import WhisperSttSidecar

    monkeypatch.setattr(worker_module, "_CTX", _StillbornContext())
    _calls, _model, _processor = _install_fake_transformers(monkeypatch)

    engine = WhisperSttSidecar(keep_alive_seconds = 0)._build_model(
        "/cached/model", "cpu", "float32", threading.Event()
    )

    assert isinstance(engine, worker_module.InProcessWhisperEngine)
    assert engine.device == "cpu"


def test_a_child_killed_by_a_signal_keeps_its_crash_instead_of_falling_back(monkeypatch):
    # A child the box killed under memory pressure bootstrapped fine, so spawn
    # works here; loading the same model in the backend would only repeat it.
    monkeypatch.setattr(worker_module, "_CTX", _StillbornContext(exitcode = -9))
    monkeypatch.setattr(
        worker_module,
        "load_whisper",
        lambda *_args, **_kwargs: pytest.fail("no in-process load after a real crash"),
    )

    handle = WhisperWorker()
    with pytest.raises(SttWorkerError, match = "SIGKILL") as caught:
        handle.start("/cached/model", "cpu", "float32")

    assert isinstance(caught.value, worker_module.SttWorkerSpawnError) is False


def test_the_in_process_fallback_reports_the_checkpoint_language_support(monkeypatch):
    _calls, model, _processor = _install_fake_transformers(monkeypatch)
    model.generation_config = SimpleNamespace(is_multilingual = False)

    engine = worker_module.InProcessWhisperEngine()
    engine.start("/cached/model", "cpu", "float32")

    # The sidecar reads this to drop the kwargs an English-only model rejects.
    assert engine.generation_config.is_multilingual is False
    engine.close()
    assert engine.is_alive() is False
