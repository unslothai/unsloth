# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Third review pass: mtmd startup preemption, reaping and idle-timer safety."""

import subprocess
import threading
import time

import pytest

from core.inference import stt_mtmd_sidecar as mtmd_mod
from core.inference.stt_mtmd_sidecar import MtmdSttSidecar


class _FakeProcess:
    """A child that stays alive until terminated, and can refuse SIGTERM."""

    _next_pid = 9000

    def __init__(self, ignores_sigterm = False):
        _FakeProcess._next_pid += 1
        self.pid = _FakeProcess._next_pid
        self._returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False
        self._ignores_sigterm = ignores_sigterm

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        if not self._ignores_sigterm:
            self._returncode = -15

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout = None):
        self.waited = True
        if self._returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self._returncode


@pytest.fixture
def spawned(monkeypatch):
    """Capture the child a load spawns, with the PID registry stubbed out."""
    made = []
    adopted, forgotten = [], []

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess()
        made.append(process)
        return process

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mtmd_mod, "adopt_pid", adopted.append)
    monkeypatch.setattr(mtmd_mod, "forget_pid", forgotten.append)
    monkeypatch.setattr(mtmd_mod, "ensure_engine_available", lambda: "/bin/llama-server")
    monkeypatch.setattr(mtmd_mod, "_llama_server_child_env", lambda binary: {})
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: False)
    monkeypatch.setattr(
        MtmdSttSidecar, "_ensure_model_downloaded", lambda self, model_id: ("m.gguf", "p.gguf")
    )
    return made, adopted, forgotten


def test_training_preempts_a_startup_before_it_can_publish(spawned, monkeypatch):
    """_process is only set once the server is ready, so training had nothing to
    act on for the whole startup and raced a child that was still allocating."""
    made, _, forgotten = spawned
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    started = threading.Event()

    def never_ready(
        process,
        port,
        cancel_event = None,
    ):
        started.set()
        while not (cancel_event is not None and cancel_event.is_set()):
            time.sleep(0.01)
        return False

    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(never_ready))
    loader = threading.Thread(target = lambda: _swallow(sidecar, "qwen3-asr-0.6b"))
    loader.start()
    try:
        assert started.wait(2), "startup never began"
        assert sidecar.is_loading()
        assert sidecar.cancel_pending_load() is True
        sidecar.wait_for_load_to_settle()
    finally:
        loader.join(timeout = 5)

    assert made[0].terminated, "the starting llama-server was left allocating"
    assert forgotten == [made[0].pid], "a reaped PID must leave the registry"


def test_cancel_pending_load_reports_when_there_is_nothing_to_cancel():
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    assert sidecar.cancel_pending_load() is False


def test_a_child_that_ignores_sigterm_is_killed_not_abandoned(monkeypatch):
    forgotten = []
    monkeypatch.setattr(mtmd_mod, "forget_pid", forgotten.append)
    process = _FakeProcess(ignores_sigterm = True)

    mtmd_mod._reap(process)

    assert process.terminated and process.killed, "SIGTERM alone leaves it holding the GPU"
    assert forgotten == [process.pid]


def test_the_idle_timer_cannot_fire_during_a_long_transcription():
    """Audio may run longer than the keep-alive, and posting happens outside the
    lock, so an armed timer would kill llama-server mid-request."""
    sidecar = MtmdSttSidecar(keep_alive_seconds = 300)
    with sidecar._lock:
        sidecar._active_requests = 1
        sidecar._schedule_idle_unload_locked()
        assert sidecar._idle_timer is None

        sidecar._active_requests = 0
        sidecar._schedule_idle_unload_locked()
        assert sidecar._idle_timer is not None
        sidecar._cancel_idle_unload_locked()


def test_an_update_blocks_new_dictation_loads(spawned):
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    with sidecar.update_maintenance() as was_active:
        assert was_active is False
        with pytest.raises(mtmd_mod.SttUnavailableError, match = "being updated"):
            sidecar.load("qwen3-asr-0.6b")
    # Released again once the install finishes.
    assert sidecar._update_in_progress is False


def test_the_llama_updater_holds_the_dictation_guard(monkeypatch):
    from contextlib import ExitStack, contextmanager

    from utils import llama_cpp_update

    entered, exited = [], []

    class _Sidecar:
        @contextmanager
        def update_maintenance(self):
            entered.append(True)
            try:
                yield True
            finally:
                exited.append(True)

    monkeypatch.setattr("core.inference.stt_mtmd_sidecar.get_mtmd_stt_sidecar", lambda: _Sidecar())
    stack = ExitStack()
    assert llama_cpp_update._block_mtmd_sidecar(stack) is True
    assert entered and not exited, "the guard must be held across the install"
    stack.close()
    assert exited


def test_an_unimportable_sidecar_never_blocks_a_llama_update(monkeypatch):
    from contextlib import ExitStack

    from utils import llama_cpp_update

    def boom():
        raise ImportError("no dictation on this host")

    monkeypatch.setattr("core.inference.stt_mtmd_sidecar.get_mtmd_stt_sidecar", boom)
    assert llama_cpp_update._block_mtmd_sidecar(ExitStack()) is False


def test_a_reaped_download_worker_leaves_the_shutdown_registry(monkeypatch):
    from core.inference import stt_download_worker

    forgotten = []
    monkeypatch.setattr("utils.process_lifetime.forget_pid", forgotten.append)

    class _Worker:
        pid = 4242

        def communicate(self):
            return b"", b"boom"

    assert stt_download_worker.reap_download(_Worker()) == b"boom"
    assert forgotten == [4242], "a PID left adopted can be reused and then signalled"


def _swallow(sidecar, model_id):
    try:
        sidecar.load(model_id)
    except Exception:
        pass


def test_a_switch_to_a_missing_model_keeps_the_working_one(spawned, monkeypatch):
    """Releasing before the cache check cost a usable server on a 409."""
    made, _, _ = spawned
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    sidecar.load("qwen3-asr-0.6b")
    assert sidecar.loaded_model == "qwen3-asr-0.6b"

    def missing(self, model_id):
        raise mtmd_mod.SttModelNotDownloadedError(f"'{model_id}' is not downloaded.")

    monkeypatch.setattr(MtmdSttSidecar, "_ensure_model_downloaded", missing)
    with pytest.raises(mtmd_mod.SttModelNotDownloadedError):
        sidecar.load("qwen3-asr-1.7b")

    assert sidecar.loaded_model == "qwen3-asr-0.6b", "the working server was torn down"
    assert made[0].poll() is None
    sidecar.unload()


def test_a_model_switch_never_kills_a_running_transcription(spawned, monkeypatch):
    made, _, _ = spawned
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    sidecar.load("qwen3-asr-0.6b")
    with sidecar._lock:
        sidecar._active_requests = 1  # mid _post_transcribe, outside the lock

    with pytest.raises(mtmd_mod.SttModelBusyError):
        sidecar.load("qwen3-asr-1.7b")
    assert made[0].poll() is None, "the in-flight request's server was killed"

    # The same model is a no-op, so a concurrent transcribe is never refused.
    sidecar.load("qwen3-asr-0.6b")
    with sidecar._lock:
        sidecar._active_requests = 0
    sidecar.unload()


def test_a_dead_server_can_still_be_replaced_while_a_request_is_pending(spawned, monkeypatch):
    made, _, _ = spawned
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    sidecar.load("qwen3-asr-0.6b")
    made[0]._returncode = 1  # crashed under the request
    with sidecar._lock:
        sidecar._active_requests = 1

    sidecar.load("qwen3-asr-1.7b")
    assert sidecar.loaded_model == "qwen3-asr-1.7b"
    with sidecar._lock:
        sidecar._active_requests = 0
    sidecar.unload()


def test_the_output_cap_follows_the_length_of_the_audio():
    budget = mtmd_mod._transcript_token_budget
    # A fixed 2048 truncated anything past roughly ten minutes of speech.
    assert budget(30 * 60) > 2048
    assert budget(1.0) == mtmd_mod._MIN_TRANSCRIPT_TOKENS
    assert budget(None) == mtmd_mod._MIN_TRANSCRIPT_TOKENS
    assert budget(0) == mtmd_mod._MIN_TRANSCRIPT_TOKENS
    assert budget(-5) == mtmd_mod._MIN_TRANSCRIPT_TOKENS
    # Bounded, so it stays inside the context that also holds the audio.
    assert budget(10**6) == mtmd_mod._MAX_TRANSCRIPT_TOKENS
    assert budget(120) > budget(60)


def test_the_engine_is_unavailable_without_pyav(monkeypatch):
    """whisper.cpp already refuses here; every transcription 501s on decode."""
    import builtins

    monkeypatch.setattr(mtmd_mod, "find_llama_server_binary", lambda: "/bin/llama-server")
    assert mtmd_mod.is_available() is True

    real_import = builtins.__import__

    def no_av(name, *args, **kwargs):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_av)
    assert mtmd_mod.is_available() is False


def test_a_training_load_pins_the_projector_off_the_gpu_too(spawned, monkeypatch):
    """-ngl 0 covers the main model only; clip.cpp offloads on its own flag."""
    commands = []

    def capture(cmd, **kwargs):
        commands.append([str(a) for a in cmd])
        return _FakeProcess()

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", capture)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: True)

    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar.load("qwen3-asr-0.6b")

    cmd = commands[0]
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert "--mmproj" in cmd, "the projector is what needs pinning"
    assert cmd[-1] == "--no-mmproj-offload", "last flag wins, so it must be last"
    sidecar.unload()


def test_an_ordinary_load_keeps_the_gpu(spawned, monkeypatch):
    commands = []

    def capture(cmd, **kwargs):
        commands.append([str(a) for a in cmd])
        return _FakeProcess()

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", capture)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: False)

    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar.load("qwen3-asr-0.6b")

    cmd = commands[0]
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert "--no-mmproj-offload" not in cmd
    sidecar.unload()


def test_unload_stops_a_startup_instead_of_letting_it_publish(spawned, monkeypatch):
    """_process is unset during startup, so a plain release was a no-op and the
    model came back resident moments after the user unloaded it."""
    made, _, forgotten = spawned
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    started = threading.Event()

    def never_ready(
        process,
        port,
        cancel_event = None,
    ):
        started.set()
        while not (cancel_event is not None and cancel_event.is_set()):
            time.sleep(0.01)
        return False

    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(never_ready))
    loader = threading.Thread(target = lambda: _swallow(sidecar, "qwen3-asr-0.6b"))
    loader.start()
    try:
        assert started.wait(2), "startup never began"
        sidecar.unload()
    finally:
        loader.join(timeout = 5)

    assert made[0].terminated, "the starting server survived an explicit unload"
    assert sidecar.loaded_model is None
    assert forgotten == [made[0].pid]


def test_the_cached_lookup_uses_studios_configured_cache(monkeypatch, tmp_path):
    """A relocated cache is written by the worker and must be read there too."""
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return "/cached/model.gguf"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("core.inference.stt_sidecar._active_hf_hub_cache", lambda: tmp_path)

    assert mtmd_mod._cached_file("qwen3-asr-0.6b", "Qwen3-ASR-0.6B-Q8_0.gguf")
    assert seen["cache_dir"] == str(tmp_path)
    assert seen["local_files_only"] is True


def test_a_server_started_for_training_goes_back_to_the_gpu_after(spawned, monkeypatch):
    """-ngl 0 was sticky: the same model matched, so dictation stayed on CPU
    until the keep-alive expired."""
    commands = []

    def capture(cmd, **kwargs):
        commands.append([str(a) for a in cmd])
        return _FakeProcess()

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", capture)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))

    training = {"active": True}
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: training["active"])
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar.load("qwen3-asr-0.6b")
    assert commands[0][commands[0].index("-ngl") + 1] == "0"

    # Still training: the same server is reused, no restart.
    sidecar.load("qwen3-asr-0.6b")
    assert len(commands) == 1

    training["active"] = False
    sidecar.load("qwen3-asr-0.6b")
    assert len(commands) == 2, "the CPU-only server was never replaced"
    assert commands[1][commands[1].index("-ngl") + 1] == "99"
    assert "--no-mmproj-offload" not in commands[1]
    sidecar.unload()


def test_a_running_transcription_outranks_the_offload_upgrade(spawned, monkeypatch):
    """Swapping to the GPU is an optimisation; it must not kill a request."""
    commands = []

    def capture(cmd, **kwargs):
        commands.append([str(a) for a in cmd])
        return _FakeProcess()

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", capture)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))

    training = {"active": True}
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: training["active"])
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar.load("qwen3-asr-0.6b")

    training["active"] = False
    with sidecar._lock:
        sidecar._active_requests = 1
    sidecar.load("qwen3-asr-0.6b")
    assert len(commands) == 1, "an in-flight transcription was torn down for -ngl"

    with sidecar._lock:
        sidecar._active_requests = 0
    sidecar.load("qwen3-asr-0.6b")
    assert len(commands) == 2, "the upgrade should happen once the request is done"
    sidecar.unload()


def test_audio_is_never_sent_to_a_server_another_client_swapped_in(spawned, monkeypatch):
    """load() returns before the request slot is taken, so the model can change
    in between and the port read would be the other server's."""
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_decode_audio_bounded", lambda audio: b"\x00\x00" * 16000)
    monkeypatch.setattr(mtmd_mod, "_pcm_to_wav_bytes", lambda pcm: b"RIFFwav")

    posted = []
    monkeypatch.setattr(
        MtmdSttSidecar,
        "_post_transcribe",
        lambda self, port, model_id, wav, seconds = None, **kwargs: posted.append((port, model_id))
        or "hi",
    )

    def swap_after_load(
        self,
        model = None,
        request_cancel_event = None,
    ):
        # Stands in for another client switching the singleton in the gap.
        with self._lock:
            self._process = _FakeProcess()
            self._port = 65000
            self._model_id = "qwen3-asr-1.7b"

    monkeypatch.setattr(MtmdSttSidecar, "load", swap_after_load)
    with pytest.raises(mtmd_mod.SttModelBusyError):
        sidecar.transcribe(b"audio", model = "qwen3-asr-0.6b")
    assert posted == [], "audio went to the model the other client loaded"


def test_a_busy_transcription_is_a_retry_not_a_server_error(monkeypatch):
    """The model-switch race is ordinary concurrency, so the client is told to
    try again rather than shown a 500."""
    import asyncio

    import routes.inference as ri
    from core.inference.stt_sidecar import SttModelBusyError
    from fastapi import HTTPException

    def busy(*args, **kwargs):
        raise SttModelBusyError("The dictation model changed. Try again.")

    monkeypatch.setattr(
        ri, "_stt_sidecar_for", lambda engine: type("S", (), {"transcribe": busy})()
    )
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: "mtmd")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            ri._transcribe_audio_bytes(
                b"audio",
                model = "qwen3-asr-0.6b",
                language = None,
                fast = True,
                engine = "mtmd",
            )
        )
    assert excinfo.value.status_code == 409
    assert "Try again" in str(excinfo.value.detail)


def test_dictation_still_works_on_cpu_during_training(spawned, monkeypatch):
    """load() starts at -ngl 0 while a run is active, so refusing here only
    threw away a recording the preload had said was fine."""
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: True)
    monkeypatch.setattr(mtmd_mod, "_decode_audio_bounded", lambda audio: b"\x00\x00" * 16000)
    monkeypatch.setattr(mtmd_mod, "_pcm_to_wav_bytes", lambda pcm: b"RIFFwav")
    monkeypatch.setattr(
        MtmdSttSidecar,
        "_post_transcribe",
        lambda self, port, model_id, wav, seconds = None, **kwargs: "on cpu",
    )

    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    result = sidecar.transcribe(b"audio", model = "qwen3-asr-0.6b")
    assert result["text"] == "on cpu"
    sidecar.unload()


def test_disconnecting_one_mtmd_request_does_not_kill_its_sibling(monkeypatch):
    monkeypatch.setattr(mtmd_mod, "ensure_engine_available", lambda: None)
    monkeypatch.setattr(MtmdSttSidecar, "_ensure_model_downloaded", lambda *args: None)
    monkeypatch.setattr(mtmd_mod, "_decode_audio_bounded", lambda audio: b"\x00\x00" * 16000)
    monkeypatch.setattr(mtmd_mod, "_pcm_to_wav_bytes", lambda pcm: b"RIFFwav")

    class _AliveProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = _AliveProcess()
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar._process = process
    sidecar._port = 65000
    sidecar._model_id = "qwen3-asr-0.6b"
    monkeypatch.setattr(sidecar, "load", lambda model, **kwargs: None)

    first_cancel = threading.Event()
    second_cancel = threading.Event()
    both_started = threading.Event()
    started_lock = threading.Lock()
    started = 0
    release_second = threading.Event()

    def post(
        _port,
        _model,
        _wav,
        _seconds = None,
        *,
        cancel_event = None,
    ):
        nonlocal started
        with started_lock:
            started += 1
            if started == 2:
                both_started.set()
        assert both_started.wait(timeout = 5)
        if cancel_event is first_cancel:
            assert first_cancel.wait(timeout = 5)
            raise mtmd_mod.SttTranscriptionCancelledError("Transcription cancelled.")
        assert release_second.wait(timeout = 5)
        return "sibling survived"

    monkeypatch.setattr(sidecar, "_post_transcribe", post)
    results = []
    errors = []

    def run(cancel_event):
        try:
            results.append(
                sidecar.transcribe(b"audio", model = "qwen3-asr-0.6b", cancel_event = cancel_event)[
                    "text"
                ]
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target = run, args = (first_cancel,))
    second = threading.Thread(target = run, args = (second_cancel,))
    first.start()
    second.start()
    assert both_started.wait(timeout = 5)
    sidecar.cancel_transcription(first_cancel)
    release_second.set()
    first.join(timeout = 5)
    second.join(timeout = 5)

    assert results == ["sibling survived"]
    assert len(errors) == 1 and isinstance(errors[0], mtmd_mod.SttTranscriptionCancelledError)
    assert process.terminated is False
    assert sidecar._active_requests == 0


def test_mtmd_disconnect_closes_the_request_connection(monkeypatch):
    requested = threading.Event()
    shutdown = threading.Event()

    class _Socket:
        def shutdown(self, _how):
            shutdown.set()

    class _Connection:
        def __init__(self, *args, **kwargs):
            self.sock = _Socket()

        def request(self, *args, **kwargs):
            requested.set()

        def getresponse(self):
            assert shutdown.wait(timeout = 5)
            raise OSError("request connection closed")

        def close(self):
            pass

    monkeypatch.setattr(mtmd_mod.http.client, "HTTPConnection", _Connection)
    sidecar = MtmdSttSidecar()
    cancelled = threading.Event()
    errors = []

    def post():
        try:
            sidecar._post_transcribe(
                65000,
                "qwen3-asr-0.6b",
                b"RIFFwav",
                cancel_event = cancelled,
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target = post)
    worker.start()
    assert requested.wait(timeout = 5)
    cancelled.set()
    worker.join(timeout = 5)

    assert shutdown.is_set()
    assert len(errors) == 1 and isinstance(errors[0], OSError)


def test_mtmd_disconnect_cancels_only_its_owned_startup():
    class _StartingProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    sidecar = MtmdSttSidecar()
    owner = threading.Event()
    sibling = threading.Event()
    load_cancel = threading.Event()
    process = _StartingProcess()
    sidecar._loading = True
    sidecar._load_cancel_event = load_cancel
    sidecar._load_owner_cancel_event = owner
    sidecar._starting_process = process

    assert sidecar.cancel_transcription(sibling) is True
    assert sibling.is_set()
    assert not load_cancel.is_set()
    assert process.terminated is False

    assert sidecar.cancel_transcription(owner) is True
    assert owner.is_set() and load_cancel.is_set()
    assert process.terminated is True


def test_a_startup_cancelled_for_training_is_retryable_not_unavailable(spawned, monkeypatch):
    """501 reads as a broken runtime; this is ordinary preemption, so 409."""
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    started = threading.Event()
    raised = []

    def never_ready(
        process,
        port,
        cancel_event = None,
    ):
        started.set()
        while not (cancel_event is not None and cancel_event.is_set()):
            time.sleep(0.01)
        return False

    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(never_ready))

    def load_and_record():
        try:
            sidecar.load("qwen3-asr-0.6b")
        except Exception as exc:
            raised.append(exc)

    loader = threading.Thread(target = load_and_record)
    loader.start()
    try:
        assert started.wait(2), "startup never began"
        sidecar.cancel_pending_load()
    finally:
        loader.join(timeout = 5)

    assert raised and isinstance(raised[0], mtmd_mod.SttLoadCancelledError)
    # Not the 501 class: the route separates them by exception type.
    assert not isinstance(raised[0], mtmd_mod.SttUnavailableError)


def test_training_that_starts_while_the_old_server_is_reaped_still_pins_the_cpu(
    spawned, monkeypatch
):
    """Reaping the previous server can take seconds, and training admission that
    already ran cannot come back to cancel this load, so the offload flags have
    to read training last rather than from the snapshot taken before the reap."""
    commands = []
    training = {"active": False}
    real_reap = mtmd_mod._reap

    def capture(cmd, **kwargs):
        commands.append([str(a) for a in cmd])
        return _FakeProcess()

    def reap_then_train(process):
        # The run that was admitted while this load waited on the reap.
        training["active"] = True
        real_reap(process)

    monkeypatch.setattr(mtmd_mod.subprocess, "Popen", capture)
    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_training_active", lambda: training["active"])

    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    sidecar.load("qwen3-asr-0.6b")
    assert commands[0][commands[0].index("-ngl") + 1] == "99", "nothing was training yet"

    monkeypatch.setattr(mtmd_mod, "_reap", reap_then_train)
    sidecar.load("qwen3-asr-1.7b")

    cmd = commands[1]
    assert cmd[cmd.index("-ngl") + 1] == "0", "spawned onto VRAM the training run had claimed"
    assert "--no-mmproj-offload" in cmd, "the projector would still have been offloaded"
    sidecar.unload()


def test_a_load_is_cancellable_before_it_decides_where_to_run(spawned, monkeypatch):
    """The other order: admission arriving after the snapshot must find a load
    it can cancel, so _loading is published before training is read."""
    seen = []
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)

    def training_active():
        seen.append(sidecar.is_loading())
        return False

    monkeypatch.setattr(MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(mtmd_mod, "_training_active", training_active)

    sidecar.load("qwen3-asr-0.6b")

    assert seen[-1] is True, "the deciding read happened while nothing could cancel the load"
    sidecar.unload()
