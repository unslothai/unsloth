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


def test_training_preempts_a_startup_unload_cannot_reach(spawned, monkeypatch):
    """The bug: _process is only set once the server is ready, so unload() is a
    no-op for the whole startup and training races an -ngl 99 child."""
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
        # unload() alone would return having done nothing at all.
        sidecar.unload()
        assert made[0].poll() is None, "unload should not reach a starting child"

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
