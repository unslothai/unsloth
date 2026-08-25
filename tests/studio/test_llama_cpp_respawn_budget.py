# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Regression tests for bounded llama-server crash recovery."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402


class _Process:
    def __init__(self, returncode: int | None = -9) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def _backend_with_dead_process() -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = _Process()
    backend._healthy = True
    backend._model_identifier = "pressure-sensitive"
    backend._last_load_intent = GgufLoadIntent(
        model_identifier = "pressure-sensitive",
        n_ctx = 16384,
    )
    return backend


def test_repeated_failed_recoveries_stop_replaying_the_same_intent() -> None:
    backend = _backend_with_dead_process()
    loads: list[GgufLoadIntent] = []

    def load_model(intent: GgufLoadIntent) -> bool:
        loads.append(intent)
        backend._process = _Process()
        backend._healthy = True
        return True

    backend.load_model = load_model

    results = [bool(backend._respawn_if_dead()) for _ in range(5)]

    assert results == [True, True, True, False, False]
    assert loads == [backend._last_load_intent] * 3


def test_opened_stream_does_not_restore_the_budget_before_done() -> None:
    backend = _backend_with_dead_process()
    loads: list[GgufLoadIntent] = []

    def dead_load(intent: GgufLoadIntent) -> bool:
        loads.append(intent)
        backend._process = _Process()
        backend._healthy = True
        return True

    backend.load_model = dead_load
    assert backend._respawn_if_dead() is not None
    assert backend._respawn_if_dead() is not None

    stream_attempts = 0

    @contextmanager
    def open_stream(url, payload, cancel_event):
        nonlocal stream_attempts
        stream_attempts += 1
        if stream_attempts == 1:
            raise httpx.ConnectError("forced dead child")
        yield object(), None

    def healthy_load(intent: GgufLoadIntent) -> bool:
        loads.append(intent)
        backend._process = _Process(returncode = None)
        backend._healthy = True
        return True

    backend._open_stream = open_stream
    backend.load_model = healthy_load
    with backend._open_chat_stream_with_respawn_retry({}, None):
        pass

    backend._process.returncode = -9
    backend.load_model = dead_load
    assert backend._respawn_if_dead() is None


def test_caller_driven_new_process_starts_a_fresh_budget() -> None:
    backend = _backend_with_dead_process()
    backend._respawn_attempts = 3
    backend._respawned_process = backend._process
    backend._process = _Process()

    def dead_load(intent: GgufLoadIntent) -> bool:
        backend._process = _Process()
        backend._healthy = True
        return True

    backend.load_model = dead_load
    assert [bool(backend._respawn_if_dead()) for _ in range(4)] == [True, True, True, False]


def test_direct_completion_recovery_restores_the_budget_each_time() -> None:
    backend = _backend_with_dead_process()
    backend._port = 1
    stream_attempts = 0

    @contextmanager
    def open_stream(url, payload, cancel_event):
        nonlocal stream_attempts
        stream_attempts += 1
        if stream_attempts % 2:
            raise httpx.ConnectError("forced dead child")
        yield object(), None

    def healthy_load(intent: GgufLoadIntent) -> bool:
        backend._process = _Process(returncode = None)
        backend._healthy = True
        return True

    backend._open_stream = open_stream
    backend._iter_text_cancellable = lambda *args, **kwargs: iter(["data: [DONE]\n"])
    backend._maybe_recover_from_mtp_crash = lambda exc: False
    backend.load_model = healthy_load

    for _ in range(4):
        list(backend.generate_chat_completion([{"role": "user", "content": "ping"}]))
        backend._process.returncode = -9

    assert stream_attempts == 8


def test_stale_completion_token_cannot_reset_a_newer_recovery() -> None:
    backend = _backend_with_dead_process()

    def dead_load(intent: GgufLoadIntent) -> bool:
        backend._process = _Process()
        backend._healthy = True
        return True

    backend.load_model = dead_load
    first = backend._respawn_if_dead()
    second = backend._respawn_if_dead()
    assert first is not None and second is not None

    backend._complete_respawn_recovery(first)
    assert backend._respawn_attempts == 2

    backend._complete_respawn_recovery(second)
    assert backend._respawn_attempts == 0
