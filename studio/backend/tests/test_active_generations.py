# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel chats: the active-generation registry and the model-swap gate.

A load/unload has to know which streaming chats it would interrupt. Everything
under test is a dict + threading.Lock, so this passes on every platform.
"""

import os
import sys
import threading

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from state import active_generations


@pytest.fixture(autouse = True)
def _clean_registry():
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


# ── registry ──────────────────────────────────────────────────────────


def test_registry_starts_empty():
    assert active_generations.count() == 0
    assert active_generations.snapshot() == []
    assert active_generations.active_thread_ids() == []


def test_entry_lives_only_for_the_block():
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1", model = "m"):
        assert active_generations.count() == 1
        assert active_generations.active_thread_ids() == ["t1"]
    assert active_generations.count() == 0
    assert active_generations.active_thread_ids() == []


def test_entry_is_removed_even_when_the_block_raises():
    ev = threading.Event()
    with pytest.raises(RuntimeError):
        with active_generations.ActiveGeneration(ev, thread_id = "t1"):
            raise RuntimeError("stream blew up")
    assert active_generations.count() == 0


def test_overlapping_runs_on_one_thread_both_register():
    # A tool continuation registers its next leg before the previous unwinds.
    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t1"):
            assert active_generations.count() == 2
            assert active_generations.active_thread_ids() == ["t1"]
        assert active_generations.count() == 1
    assert active_generations.count() == 0


def test_snapshot_is_json_safe_and_ordered_by_start():
    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "first", model = "m1"):
        with active_generations.ActiveGeneration(b, thread_id = "second", model = "m2"):
            snap = active_generations.snapshot()
    assert [e["thread_id"] for e in snap] == ["first", "second"]
    # The threading.Event must not leak into an HTTP response body.
    assert all("event" not in e for e in snap)
    assert {"handle", "thread_id", "model", "kind", "started_at"} == set(snap[0])


def test_thread_ids_are_deduped_and_skip_unnamed_runs():
    a, b, c = threading.Event(), threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t1"):
            # A brand-new chat whose first turn races persistence has no id yet.
            with active_generations.ActiveGeneration(c, thread_id = None):
                assert active_generations.active_thread_ids() == ["t1"]
                assert active_generations.count() == 3


# ── cancellation ──────────────────────────────────────────────────────


def test_cancel_all_sets_every_event():
    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t2"):
            assert active_generations.cancel_all() == 2
            assert a.is_set() and b.is_set()


def test_cancel_all_on_an_empty_registry_is_a_no_op():
    assert active_generations.cancel_all() == 0


def test_cancel_thread_leaves_siblings_alone():
    # Per-thread Stop: the rest keep generating, llama-server is untouched.
    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t2"):
            assert active_generations.cancel_thread("t1") == 1
            assert a.is_set()
            assert not b.is_set()


def test_cancel_thread_with_no_match_is_a_no_op():
    a = threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        assert active_generations.cancel_thread("nope") == 0
        assert active_generations.cancel_thread("") == 0
        assert not a.is_set()


def test_cancel_does_not_unregister_entries():
    # __exit__ owns removal, so a generation mid-cleanup is not lost.
    a = threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        active_generations.cancel_all()
        assert active_generations.count() == 1


# ── concurrency ───────────────────────────────────────────────────────


def test_registry_survives_concurrent_register_unregister():
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout = 10)
            for _ in range(50):
                with active_generations.ActiveGeneration(
                    threading.Event(), thread_id = f"t{i}"
                ):
                    active_generations.snapshot()
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            errors.append(exc)

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 30)

    assert errors == []
    assert active_generations.count() == 0


# ── the model-swap gate ───────────────────────────────────────────────


# The registry needs only the stdlib, so it runs anywhere. The gate lives in
# routes.inference, which pulls the whole inference stack: skip when absent.
def _route_gate():
    pytest.importorskip("fastapi", reason="inference stack not installed")
    routes_inference = pytest.importorskip(
        "routes.inference", reason="inference stack not installed"
    )
    return routes_inference._raise_or_cancel_active_generations


@pytest.fixture
def gate():
    return _route_gate()


def test_gate_allows_a_swap_when_nothing_is_generating(gate):
    assert gate(force = False, action = "Loading a model") == 0


def test_gate_refuses_with_409_and_names_the_chats(gate):
    from fastapi import HTTPException

    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t2"):
            with pytest.raises(HTTPException) as exc:
                gate(force = False, action = "Loading a model")
    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["error"] == "active_generations"
    assert detail["running"] == 2
    assert detail["thread_ids"] == ["t1", "t2"]
    # Refusing must not cancel anything.
    assert not a.is_set() and not b.is_set()


def test_gate_message_is_singular_for_one_chat(gate):
    from fastapi import HTTPException

    with active_generations.ActiveGeneration(threading.Event(), thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            gate(force = False, action = "Unloading the model")
    message = exc.value.detail["message"]
    assert "1 chat that is still generating" in message
    assert "Unloading the model" in message


def test_gate_force_cancels_and_returns_the_count(gate):
    a, b = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(a, thread_id = "t1"):
        with active_generations.ActiveGeneration(b, thread_id = "t2"):
            assert gate(force = True, action = "Loading a model") == 2
            assert a.is_set() and b.is_set()


def test_gate_force_with_nothing_running_is_a_no_op(gate):
    assert gate(force = True, action = "Loading a model") == 0


# ── the route wiring ──────────────────────────────────────────────────


def test_tracked_cancel_registers_the_thread_for_its_block():
    # The single place a generation is recorded, so every streaming path gets it.
    _route_gate()  # skips when the inference stack is unavailable
    from routes.inference import _TrackedCancel

    ev = threading.Event()
    tracker = _TrackedCancel(ev, "cancel-1", thread_id = "t1", model = "m")
    tracker.__enter__()
    try:
        assert active_generations.active_thread_ids() == ["t1"]
        assert active_generations.snapshot()[0]["model"] == "m"
    finally:
        tracker.__exit__(None, None, None)
    assert active_generations.count() == 0


def test_tracked_cancel_shares_its_event_with_the_registry():
    # Reusing the per-run event is what keeps a forced reload off llama-server.
    _route_gate()  # skips when the inference stack is unavailable
    from routes.inference import _TrackedCancel

    ev = threading.Event()
    tracker = _TrackedCancel(ev, "cancel-1", thread_id = "t1")
    tracker.__enter__()
    try:
        active_generations.cancel_all()
        assert ev.is_set()
    finally:
        tracker.__exit__(None, None, None)


def test_load_and_unload_requests_default_to_not_cancelling():
    pytest.importorskip("pydantic", reason="pydantic not installed")
    from models.inference import LoadRequest, UnloadRequest

    assert LoadRequest(model_path = "m").force_cancel_active is False
    assert UnloadRequest(model_path = "m").force_cancel_active is False
    assert (
        LoadRequest(model_path = "m", force_cancel_active = True).force_cancel_active
        is True
    )


def _parallel_constants(path: str) -> dict:
    """Read the _PARALLEL_* constants from a file's source.

    Importing run.py would drag in the whole server to read three integers.
    """
    import ast

    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, "id", "")
            if name.startswith("_PARALLEL_") and isinstance(node.value, ast.Constant):
                found[name] = node.value.value
    return found


def test_studio_defaults_to_more_than_one_decode_slot():
    # With one slot the admission queue serialises every chat.
    consts = _parallel_constants(os.path.join(_backend, "run.py"))

    assert consts["_PARALLEL_DEFAULT_PLAIN"] > 1
    assert (
        consts["_PARALLEL_MIN"]
        <= consts["_PARALLEL_DEFAULT_PLAIN"]
        <= consts["_PARALLEL_MAX"]
    )


def test_cli_and_backend_parallel_defaults_agree():
    # argparse and the typer CLI are separate entry points into the same server.
    backend = _parallel_constants(os.path.join(_backend, "run.py"))
    cli_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(_backend))),
        "unsloth_cli", "commands", "studio.py",
    )
    cli = _parallel_constants(cli_path)

    assert cli["_PARALLEL_DEFAULT_PLAIN"] == backend["_PARALLEL_DEFAULT_PLAIN"]
