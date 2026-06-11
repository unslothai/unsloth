# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The bounded extractor semaphore must not leak when multiprocessing setup
raises after a slot was acquired.

Pinned regression: get_context / Queue / Process raising (fork exhaustion,
Windows EAGAIN) escaped before the try/finally, losing the permit forever.
The try now covers those calls; assert ``_EXTRACT_SEMAPHORE._value`` is
restored after a forced failure at each call site.
"""

from __future__ import annotations

import os

import pytest


# Don't park the test waiting for a slot to free.
os.environ.setdefault("UNSLOTH_STUDIO_EXTRACT_QUEUE_WAIT", "0")


@pytest.fixture
def extractor():
    """Yield the document_extractor module.

    No ``importlib.reload``: reloading swaps the
    module-level ``_drain_future_exception`` function object out from
    under ``routes.inference`` (which captured it at import time),
    and other tests assert identity between the two references.
    Instead we snapshot ``_EXTRACT_SEMAPHORE._value`` before each
    test and assert restoration after; no reload required.
    """
    from core.chat import document_extractor as mod
    yield mod


def _semaphore_value(mod) -> int:
    # CPython BoundedSemaphore exposes the counter as private ``_value``;
    # the test is about that counter and the alternatives are flakier.
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
            timeout_seconds = 5,
        )

    assert _semaphore_value(extractor) == initial, (
        f"semaphore leaked one permit at failure point {where!r}: "
        f"expected {initial}, got {_semaphore_value(extractor)}"
    )


def test_repeated_failure_does_not_drain_pool(extractor, monkeypatch):
    """Run the failure path 5x and confirm full pool capacity afterwards --
    the production regression is sustained: one permit leaked per failed extraction, and the queue
    eventually deadlocks."""
    initial = _semaphore_value(extractor)
    _force_failure(extractor, monkeypatch, "process")

    for _ in range(5):
        with pytest.raises((OSError, RuntimeError)):
            extractor._run_extract_process_sync(
                b"x",
                "x.txt",
                {},
                "text/plain",
                timeout_seconds = 2,
            )

    assert _semaphore_value(extractor) == initial
