# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A server shutdown asks a running training job to stop and save before anything is killed."""

from __future__ import annotations

import threading
import time

import pytest


class _FakeProc:
    def __init__(self):
        self.alive = True
        self.pid = 4242

    def is_alive(self):
        return self.alive


@pytest.fixture
def backend():
    from core.training.training import TrainingBackend
    return TrainingBackend()


def _running(backend, job_id = "job_1"):
    backend._proc = _FakeProc()
    backend.current_job_id = job_id
    return backend._proc


def _record_stop(
    monkeypatch,
    backend,
    result = True,
    on_stop = None,
):
    calls = []

    def stop_training(**kw):
        calls.append(kw)
        if on_stop is not None:
            on_stop()
        return result

    monkeypatch.setattr(backend, "stop_training", stop_training)
    return calls


def test_an_idle_backend_needs_no_stop(backend, monkeypatch):
    calls = _record_stop(monkeypatch, backend)
    assert backend.stop_for_shutdown(timeout = 1) is True
    assert calls == []


def test_a_running_job_is_stopped_with_save_and_waited_for(backend, monkeypatch):
    proc = _running(backend)
    calls = _record_stop(
        monkeypatch,
        backend,
        on_stop = lambda: threading.Timer(0.2, lambda: setattr(proc, "alive", False)).start(),
    )
    t0 = time.monotonic()
    assert backend.stop_for_shutdown(timeout = 5) is True
    elapsed = time.monotonic() - t0
    assert calls == [{"save": True, "expected_job_id": "job_1"}]
    # Returning before the worker is gone would mean the caller kills a mid-save worker.
    assert not proc.is_alive()
    assert 0.2 <= elapsed < 4


def test_a_saved_run_counts_as_done_even_while_the_worker_lingers(backend, monkeypatch):
    proc = _running(backend)
    _record_stop(
        monkeypatch,
        backend,
        on_stop = lambda: threading.Timer(0.2, backend._complete_seen.set).start(),
    )
    t0 = time.monotonic()
    assert backend.stop_for_shutdown(timeout = 5) is True
    elapsed = time.monotonic() - t0
    assert proc.is_alive() and backend.is_run_finished()
    assert 0.2 <= elapsed < 4


def test_an_already_finished_run_is_not_stopped_again(backend, monkeypatch):
    _running(backend)
    backend._complete_seen.set()
    calls = _record_stop(monkeypatch, backend)
    assert backend.stop_for_shutdown(timeout = 5) is True
    assert calls == []


def test_the_wait_is_bounded(backend, monkeypatch):
    _running(backend)
    _record_stop(monkeypatch, backend)
    t0 = time.monotonic()
    assert backend.stop_for_shutdown(timeout = 0.5) is False
    assert 0.4 < time.monotonic() - t0 < 3


def test_a_refused_stop_is_reported(backend, monkeypatch):
    _running(backend)
    _record_stop(monkeypatch, backend, result = False)
    assert backend.stop_for_shutdown(timeout = 5) is False
