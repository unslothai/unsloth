# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: ``get_inference_backend()`` builds exactly one orchestrator, even
when several threads reach it at once.

The first call is expensive: ``__init__`` runs ``get_default_models()``, which calls
``hw.get_device()`` and so blocks on the torch warm, ~2.9s on a cold GPU host. That is
why the first-paint routes call it through ``asyncio.to_thread``.

Off-loop means genuinely parallel, though, and the getter used to be a plain
check-then-set on a module global. Concurrent first-paint requests all observed
``None`` inside that window, each built an orchestrator, and the last assignment won.
Orchestrator state is per-instance (subprocess handle, ``loading_models``,
``active_model_name``), so a load started on a loser became invisible to every later
call, which read the survivor.

The orchestrator is stubbed out here: these exercise the getter's locking, not the
constructor.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

import core.inference.orchestrator as orch

_ORCHESTRATOR_SRC = Path(orch.__file__)

# Wide enough that every thread is inside the window before the first leaves it, small
# enough to stay a unit test. The real window is ~2.9s.
_BUILD_SECONDS = 0.20
_THREADS = 8


@pytest.fixture
def fresh_singleton(monkeypatch):
    """Reset the module global and restore it afterwards.

    Set directly, not via monkeypatch.setattr, so the getter's ``global`` write is what
    the test observes.
    """
    saved = orch._inference_backend
    orch._inference_backend = None
    try:
        yield
    finally:
        orch._inference_backend = saved


class _StubOrchestrator:
    """Records every construction and holds the window open the way detection does."""

    built: list["_StubOrchestrator"] = []
    _record_lock = threading.Lock()

    def __init__(self):
        with self._record_lock:
            self.__class__.built.append(self)
        # The real constructor spends this time in hw.get_device().
        threading.Event().wait(_BUILD_SECONDS)


@pytest.fixture
def stub_orchestrator(monkeypatch):
    _StubOrchestrator.built = []
    monkeypatch.setattr(orch, "InferenceOrchestrator", _StubOrchestrator)
    return _StubOrchestrator


def test_concurrent_first_calls_build_exactly_one_orchestrator(fresh_singleton, stub_orchestrator):
    """The regression: N threads entering a cold getter together.

    Unlocked, every thread observes None inside the construction window and builds its
    own. Asserts both that one is built and that every caller gets that same one: an
    orphan handed to even one caller is the bug.
    """
    handed_out: list[object] = []
    handed_lock = threading.Lock()
    errors: list[BaseException] = []
    gate = threading.Barrier(_THREADS)

    def worker():
        try:
            gate.wait(timeout = 30)
            backend = orch.get_inference_backend()
            with handed_lock:
                handed_out.append(backend)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target = worker, name = f"getter-{i}") for i in range(_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        # Generous, so a deadlock fails here rather than hanging the suite.
        t.join(timeout = 60)
        assert not t.is_alive(), f"{t.name} never returned from get_inference_backend()"

    assert not errors, f"worker threads raised: {errors}"
    assert len(handed_out) == _THREADS

    assert len(stub_orchestrator.built) == 1, (
        f"{len(stub_orchestrator.built)} orchestrators were constructed; "
        "concurrent first calls must share one"
    )

    survivor = orch._inference_backend
    assert survivor is stub_orchestrator.built[0]
    orphans = [b for b in handed_out if b is not survivor]
    assert not orphans, (
        f"{len(orphans)} callers were handed an orchestrator that is not the "
        "module global; a load started on one would be invisible to later calls"
    )


def test_warm_path_does_not_take_the_lock(fresh_singleton, stub_orchestrator):
    """Once built, the getter must not serialize on the lock.

    Every request path reaches it, so a single ``with`` around the whole body would
    funnel all of them through one mutex.
    """
    first = orch.get_inference_backend()
    assert len(stub_orchestrator.built) == 1

    returned: list[object] = []
    with orch._inference_backend_lock:
        # Lock held by this thread: a warm call from another must still return,
        # which it can only do by skipping the lock.
        t = threading.Thread(target = lambda: returned.append(orch.get_inference_backend()))
        t.start()
        t.join(timeout = 10)
        assert not t.is_alive(), "the warm path blocked on the singleton lock"

    assert returned == [first]
    assert len(stub_orchestrator.built) == 1


def test_getter_constructs_under_a_module_level_lock():
    """Static guard: construction must stay inside a ``with`` on the lock.

    A refactor back to the bare ``if _inference_backend is None: ...`` reads fine and
    passes every single-threaded test, so pin the shape.
    """
    tree = ast.parse(_ORCHESTRATOR_SRC.read_text(encoding = "utf-8"))

    assigns_lock = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_inference_backend_lock" for t in node.targets)
    ]
    assert len(assigns_lock) == 1, "expected one module-level _inference_backend_lock"

    getter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_inference_backend"
    )

    constructions = [
        node
        for node in ast.walk(getter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InferenceOrchestrator"
    ]
    assert constructions, "get_inference_backend no longer constructs the orchestrator"

    guarded: set[int] = set()
    for with_node in ast.walk(getter):
        if not isinstance(with_node, ast.With):
            continue
        holds_lock = any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "_inference_backend_lock"
            for item in with_node.items
        )
        if not holds_lock:
            continue
        for inner in ast.walk(with_node):
            guarded.add(id(inner))

    unguarded = [c for c in constructions if id(c) not in guarded]
    assert not unguarded, (
        "InferenceOrchestrator() is constructed outside "
        "`with _inference_backend_lock:` -- concurrent first callers will each "
        "build their own and orphan all but the last"
    )
