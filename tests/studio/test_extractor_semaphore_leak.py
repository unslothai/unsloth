# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Tests that the bounded extractor semaphore in
``core.chat.document_extractor`` does not leak when multiprocessing
setup raises *after* a slot has already been acquired.

Failure mode the test pins:
    1. ``_run_extract_process_sync`` acquires ``_EXTRACT_SEMAPHORE``.
    2. ``multiprocessing.get_context(...)`` / ``ctx.Queue(...)`` /
       ``ctx.Process(...)`` raises an OSError (fork-resource
       exhaustion, EAGAIN on Windows under pressure, Queue creation
       failure on hardened sandboxes, etc).
    3. The exception escapes before the worker even starts, so the
       finally block does not run -- and the permit is lost forever.

After the patch, the ``try`` is moved up to cover the
``get_context`` / ``Queue`` / ``Process`` calls, so the semaphore is
always released. We assert ``_EXTRACT_SEMAPHORE._value`` is restored
after a forced failure for every plausible call site.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make studio/backend imports resolvable when run from the repo root.
_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# Don't park the test waiting for a slot to free.
os.environ.setdefault("UNSLOTH_STUDIO_EXTRACT_QUEUE_WAIT", "0")


@pytest.fixture
def extractor():
    """Yield the document_extractor module.

    We avoid ``importlib.reload`` here because reloading swaps the
    module-level ``_drain_future_exception`` function object out from
    under ``routes.inference`` (which captured it at import time),
    and other tests assert identity between the two references.
    Instead we snapshot ``_EXTRACT_SEMAPHORE._value`` before each
    test and assert restoration after; no reload required.
    """
    from core.chat import document_extractor as mod

    yield mod


def _semaphore_value(mod) -> int:
    # BoundedSemaphore in CPython exposes the current counter as
    # ``_value`` -- this is a private implementation detail, but the
    # test is explicitly about that counter and the alternatives
    # (probing acquire/release reentrancy) are flakier.
    return mod._EXTRACT_SEMAPHORE._value


def _force_failure(mod, monkeypatch, where: str) -> None:
    """Inject a raising stub at one of the three failure points."""
    import multiprocessing

    real_ctx = multiprocessing.get_context

    def boom(*args, **kwargs):
        raise OSError("simulated multiprocessing failure for test")

    if where == "get_context":
        monkeypatch.setattr(mod.multiprocessing, "get_context", boom)
    elif where == "queue":
        class _Ctx:
            def Queue(self, *_a, **_kw):
                raise OSError("simulated Queue allocation failure")

            def Process(self, *_a, **_kw):  # pragma: no cover - never reached
                return None

        monkeypatch.setattr(mod.multiprocessing, "get_context", lambda *_a, **_kw: _Ctx())
    elif where == "process":
        class _Q:
            def close(self):
                pass

            def join_thread(self):
                pass

        class _Ctx:
            def Queue(self, *_a, **_kw):
                return _Q()

            def Process(self, *_a, **_kw):
                raise OSError("simulated Process construction failure")

        monkeypatch.setattr(mod.multiprocessing, "get_context", lambda *_a, **_kw: _Ctx())
    else:  # pragma: no cover
        raise ValueError(where)


@pytest.mark.parametrize("where", ["get_context", "queue", "process"])
def test_semaphore_released_when_mp_setup_fails(extractor, monkeypatch, where):
    initial = _semaphore_value(extractor)
    _force_failure(extractor, monkeypatch, where)

    with pytest.raises((OSError, RuntimeError)):
        extractor._run_extract_process_sync(
            b"hello world",
            "test.txt",
            {},
            "text/plain",
            timeout_seconds=5,
        )

    assert _semaphore_value(extractor) == initial, (
        f"semaphore leaked one permit at failure point {where!r}: "
        f"expected {initial}, got {_semaphore_value(extractor)}"
    )


def test_repeated_failure_does_not_drain_pool(extractor, monkeypatch):
    """Run the failure path 5x and confirm the pool is still at full
    capacity afterwards -- the regression that hits production is
    sustained: one permit leaked per failed extraction, and the queue
    eventually deadlocks."""
    initial = _semaphore_value(extractor)
    _force_failure(extractor, monkeypatch, "process")

    for _ in range(5):
        with pytest.raises((OSError, RuntimeError)):
            extractor._run_extract_process_sync(
                b"x", "x.txt", {}, "text/plain", timeout_seconds=2,
            )

    assert _semaphore_value(extractor) == initial
