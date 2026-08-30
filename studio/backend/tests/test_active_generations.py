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
    assert {"handle", "thread_id", "run_id", "model", "kind", "started_at"} == set(snap[0])


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


def test_cancel_run_targets_only_matching_durable_generation():
    durable, sibling = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(durable, thread_id = "t1", run_id = "run-1"):
        with active_generations.ActiveGeneration(sibling, thread_id = "t1", run_id = "run-2"):
            assert active_generations.cancel_run("run-1") == 1
            assert durable.is_set()
            assert not sibling.is_set()


def test_same_durable_run_and_event_borrows_existing_registration():
    event = threading.Event()
    with active_generations.ActiveGeneration(
        event, run_id = "run-1", thread_id = "stale", model = "stale"
    ):
        with active_generations.ActiveGeneration(
            event, run_id = "run-1", thread_id = "thread-1", model = "local"
        ):
            snapshot = active_generations.snapshot()[0]
            assert (active_generations.count(), snapshot["thread_id"], snapshot["model"]) == (
                1,
                "thread-1",
                "local",
            )
            assert active_generations.cancel_all() == 1
    assert event.is_set()
    assert active_generations.count() == 0


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
                with active_generations.ActiveGeneration(threading.Event(), thread_id = f"t{i}"):
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


# The gate lives in routes.inference, which pulls the whole inference stack.
def _route_gate():
    pytest.importorskip("fastapi", reason = "inference stack not installed")
    routes_inference = pytest.importorskip(
        "routes.inference", reason = "inference stack not installed"
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
    _route_gate()
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
    _route_gate()
    from routes.inference import _TrackedCancel

    ev = threading.Event()
    tracker = _TrackedCancel(ev, "cancel-1", thread_id = "t1")
    tracker.__enter__()
    try:
        active_generations.cancel_all()
        assert ev.is_set()
    finally:
        tracker.__exit__(None, None, None)


def _stub_load_route(monkeypatch, *, active_model_name):
    """Point POST /load at an in-memory safetensors backend.

    active_model_name == the requested path makes the request idempotent, so
    _load_model_impl takes its already_loaded fast return.
    """
    from types import SimpleNamespace

    import routes.inference as inf_mod

    monkeypatch.setattr(inf_mod, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf_mod, "validate_extra_args", lambda args: [])
    monkeypatch.setattr(
        inf_mod,
        "resolve_effective_chat_template_override",
        lambda model_identifier = None, user_override = None: None,
    )
    monkeypatch.setattr(inf_mod, "load_inference_config", lambda name: {})
    monkeypatch.setattr(
        inf_mod,
        "_detect_safetensors_features",
        lambda backend, template, tools = None: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_effort_levels": [],
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )
    monkeypatch.setattr(inf_mod, "_resolve_loaded_trust_remote_code", lambda *a, **k: False)
    monkeypatch.setattr(
        inf_mod,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = active_model_name, models = {}),
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, hf_variant = None, model_identifier = None),
    )
    return inf_mod


def test_idempotent_load_neither_refuses_nor_cancels_running_chats(monkeypatch):
    # Re-applying the resident model hits already_loaded: no llama-server touch, no 409, no stopped chats.
    _route_gate()
    import asyncio

    from models.inference import LoadRequest

    inf_mod = _stub_load_route(monkeypatch, active_model_name = "org/A")

    for force in (False, True):
        ev = threading.Event()
        with active_generations.ActiveGeneration(ev, thread_id = "t1"):
            response = asyncio.run(
                inf_mod.load_model(
                    LoadRequest(model_path = "org/A", force_cancel_active = force),
                    object(),
                    "tester",
                )
            )
        assert response.status == "already_loaded"
        assert not ev.is_set()


def test_a_real_reload_still_refuses_while_chats_stream(monkeypatch):
    # A load that would really replace the model still 409s and names the chats.
    _route_gate()
    import asyncio

    from fastapi import HTTPException

    from models.inference import LoadRequest

    inf_mod = _stub_load_route(monkeypatch, active_model_name = "org/OTHER")

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(inf_mod.load_model(LoadRequest(model_path = "org/A"), object(), "tester"))
    assert exc.value.status_code == 409
    assert exc.value.detail["thread_ids"] == ["t1"]
    assert not ev.is_set()


def test_a_forced_load_that_fails_preflight_leaves_the_chats_alone(monkeypatch):
    # Preflight can still reject after the user confirms, so cancelling first ends chats for nothing.
    _route_gate()
    import asyncio
    import contextlib

    from fastapi import HTTPException

    from models.inference import LoadRequest

    inf_mod = _stub_load_route(monkeypatch, active_model_name = "org/OTHER")
    monkeypatch.setattr(inf_mod, "_hf_offline_if_unreachable", contextlib.nullcontext)
    # Stands in for any preflight refusal; a None here is the route's own 400.
    monkeypatch.setattr(inf_mod.ModelConfig, "from_identifier", staticmethod(lambda **kwargs: None))

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inf_mod.load_model(
                    LoadRequest(model_path = "org/A", force_cancel_active = True),
                    object(),
                    "tester",
                )
            )
        # The load was rejected, so the chat must still be streaming.
        assert not ev.is_set()
        assert active_generations.count() == 1
    assert exc.value.status_code == 400


def _stub_standard_load_route(monkeypatch):
    """Drive _load_model_impl down the Unsloth path as far as the pre-teardown drain."""
    import contextlib
    from types import SimpleNamespace

    import routes.inference as inf_mod

    real_sidecar_check = inf_mod._raise_if_sidecar_swap_in_progress
    _stub_load_route(monkeypatch, active_model_name = "org/OTHER")
    # _stub_load_route neutralises the sidecar guard; this test is about it.
    monkeypatch.setattr(inf_mod, "_raise_if_sidecar_swap_in_progress", real_sidecar_check)
    monkeypatch.setattr(inf_mod, "_hf_offline_if_unreachable", contextlib.nullcontext)
    monkeypatch.setattr(inf_mod, "_mlx_distributed_launch_detected", lambda: False)
    monkeypatch.setattr(
        inf_mod.ModelConfig,
        "from_identifier",
        staticmethod(
            lambda **kwargs: SimpleNamespace(
                is_gguf = False,
                identifier = "org/A",
                display_name = "A",
                is_vision = False,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
        ),
    )
    monkeypatch.setattr(inf_mod, "_effective_load_in_4bit", lambda config, requested: False)
    monkeypatch.setattr(inf_mod, "_resolve_inherited_extra_args", lambda *a, **k: None)
    monkeypatch.setattr(inf_mod, "_guard_chat_load_against_training", lambda *a, **k: None)
    return inf_mod


def test_a_sidecar_swap_reserved_during_the_drain_never_strands_cancelled_chats(monkeypatch):
    # A sidecar install can reserve the swap window during the pre-teardown drain, so the recheck
    # after it is the last rejection point and must precede the cancel, else chats die for nothing.
    _route_gate()
    import asyncio
    import time
    from types import SimpleNamespace

    from fastapi import HTTPException

    from core.inference import llama_keepwarm as kw
    from models.inference import LoadRequest

    import utils.transformers_version as tv

    inf_mod = _stub_standard_load_route(monkeypatch)
    reserved = {"v": False}
    monkeypatch.setattr(tv, "sidecar_swap_in_progress", lambda: reserved["v"])

    # Two tracked requests; the install reserves the window mid-drain when the uncancellable one ends.
    monkeypatch.setattr(kw, "_inflight", 2)

    def _installer():
        time.sleep(0.10)
        kw._inflight = 1  # the non-cancellable request finished ...
        reserved["v"] = True  # ... and an install reserved the swap window
        time.sleep(0.35)
        kw._inflight = 0  # the chat's own request drains last

    thread = threading.Thread(target = _installer, daemon = True)
    ev = threading.Event()
    try:
        with active_generations.ActiveGeneration(ev, thread_id = "t1"):
            thread.start()
            with pytest.raises(HTTPException) as exc:
                asyncio.run(
                    inf_mod.load_model(
                        LoadRequest(model_path = "org/A", force_cancel_active = True),
                        SimpleNamespace(
                            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
                        ),
                        "tester",
                    )
                )
            # Rejected, so the chat traded for a model it never got must still stream.
            assert not ev.is_set()
            assert active_generations.count() == 1
        assert exc.value.status_code == 409
        assert "transformers installation" in str(exc.value.detail)
    finally:
        thread.join(timeout = 5)
        kw._inflight = 0


def _stub_unload_backends(monkeypatch, *, llama, backend):
    """Point the /unload route at in-memory backends."""
    import routes.inference as inf_mod
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inf_mod, "is_registered_native_path_label", lambda *a: False)
    monkeypatch.setattr(kw, "note_model_unloaded", lambda: None)
    return inf_mod, kw


def test_unload_rechecks_active_generations_under_the_lifecycle_gate(monkeypatch):
    # Without the recheck, a chat that starts while this queues on the gate is torn down mid-stream.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from models.inference import UnloadRequest

    torn_down: list[str] = []
    inf_mod, kw = _stub_unload_backends(
        monkeypatch,
        llama = SimpleNamespace(
            is_active = True,
            is_loaded = True,
            model_identifier = "org/A-GGUF",
            unload_model = lambda: torn_down.append("gguf"),
        ),
        backend = SimpleNamespace(
            get_loading_model = lambda: None,
            unload_model = lambda path: torn_down.append("unsloth"),
        ),
    )

    ev = threading.Event()
    started = active_generations.ActiveGeneration(ev, thread_id = "t1")

    async def drive():
        # A load holds the lifecycle gate, so the unload queues behind it.
        kw._lifecycle_lock.acquire()
        task = asyncio.create_task(
            inf_mod.unload_model(UnloadRequest(model_path = "org/A-GGUF"), "tester")
        )
        entered = False
        try:
            await asyncio.sleep(0.1)  # the route is polling the gate
            started.__enter__()  # a chat starts in the meantime
            entered = True
        finally:
            kw._lifecycle_lock.release()
        try:
            return await asyncio.wait_for(task, timeout = 5)
        finally:
            if entered:
                started.__exit__(None, None, None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(drive())

    # 409, not the catch-all 500 the route wraps unexpected failures in.
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "active_generations"
    assert torn_down == []
    assert not ev.is_set()


def _run_unload(
    inf_mod,
    monkeypatch,
    *,
    loaded_gguf,
    requested,
    force,
    torn_down,
    unload_model = None,
):
    """Drive POST /unload against a backend pair with ``loaded_gguf`` resident.

    ``unload_model`` overrides the GGUF teardown so a caller can observe what the
    world looked like at the moment of teardown, not just afterwards.
    """
    import asyncio
    from types import SimpleNamespace

    from models.inference import UnloadRequest

    _stub_unload_backends(
        monkeypatch,
        llama = SimpleNamespace(
            is_active = True,
            is_loaded = True,
            model_identifier = loaded_gguf,
            unload_model = unload_model or (lambda: torn_down.append("gguf")),
        ),
        # Nothing on the standard backend: the GGUF above is what is resident.
        backend = SimpleNamespace(
            get_loading_model = lambda: None,
            active_model_name = None,
            models = {},
            unload_model = lambda path: torn_down.append("unsloth"),
        ),
    )
    return asyncio.run(
        inf_mod.unload_model(
            UnloadRequest(model_path = requested, force_cancel_active = force), "tester"
        )
    )


_PINNED_SNAPSHOT = "/tmp/hf/hub/models--Org--Quant/snapshots/" + "d" * 40


def test_unload_finds_a_gguf_loaded_from_a_pinned_snapshot(monkeypatch):
    """A cached row that pins a snapshot is loaded by that path, while the status the client
    reads back reports the repo id. An unload naming the id has to reach the same server, or
    Eject reports success and leaves it resident."""
    _route_gate()
    import routes.inference as inf_mod

    torn_down: list[str] = []
    _run_unload(
        inf_mod,
        monkeypatch,
        loaded_gguf = _PINNED_SNAPSHOT,
        requested = "Org/Quant",
        force = False,
        torn_down = torn_down,
    )
    assert torn_down == ["gguf"]

    # Control: another repo's id names a different model and must leave this one up.
    other: list[str] = []
    _run_unload(
        inf_mod,
        monkeypatch,
        loaded_gguf = _PINNED_SNAPSHOT,
        requested = "Org/Other",
        force = False,
        torn_down = other,
    )
    assert other == ["unsloth"]


def test_stop_loading_cancels_an_in_flight_pinned_load(monkeypatch):
    """Cancel sends the id the picker shows. A pinned row loads under a snapshot path, so a raw
    compare let Stop loading report success while the multi minute load kept running."""
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from models.inference import UnloadRequest

    def _cancel(loading, requested):
        cancelled: list[str] = []
        inf_mod, _kw = _stub_unload_backends(
            monkeypatch,
            llama = SimpleNamespace(is_active = False, is_loaded = False, model_identifier = None),
            backend = SimpleNamespace(
                get_loading_model = lambda: loading,
                cancel_load = lambda path: (cancelled.append(path), True)[1],
                active_model_name = None,
                models = {},
                unload_model = lambda path: None,
            ),
        )
        asyncio.run(
            inf_mod.unload_model(
                UnloadRequest(model_path = requested, force_cancel_active = False), "tester"
            )
        )
        return cancelled

    # Cancelled under the name the load runs as, not the one the client sent.
    assert _cancel(_PINNED_SNAPSHOT, "Org/Quant") == [_PINNED_SNAPSHOT]
    assert _cancel("Org/Quant", "Org/Quant") == ["Org/Quant"]
    # Control: naming another repo cancels nothing.
    assert _cancel(_PINNED_SNAPSHOT, "Org/Other") == []


def test_unload_evicts_a_pinned_standard_model_under_its_registered_name(monkeypatch):
    """The standard backend refuses a name it never loaded, so a pinned load has to be evicted
    under the path it was registered with rather than the id the picker shows."""
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from models.inference import UnloadRequest

    unloaded: list[str] = []
    inf_mod, _kw = _stub_unload_backends(
        monkeypatch,
        llama = SimpleNamespace(is_active = False, is_loaded = False, model_identifier = None),
        backend = SimpleNamespace(
            get_loading_model = lambda: None,
            active_model_name = _PINNED_SNAPSHOT,
            models = {_PINNED_SNAPSHOT: {}},
            unload_model = lambda path: unloaded.append(path),
        ),
    )
    asyncio.run(
        inf_mod.unload_model(
            UnloadRequest(model_path = "Org/Quant", force_cancel_active = False), "tester"
        )
    )
    assert unloaded == [_PINNED_SNAPSHOT]


def test_forced_unload_of_a_stale_model_path_leaves_the_chats_alone(monkeypatch):
    # Eject naming a model another tab swapped out: a no-op success; cancelling first loses runs.
    _route_gate()
    import routes.inference as inf_mod

    torn_down: list[str] = []
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = _run_unload(
            inf_mod,
            monkeypatch,
            loaded_gguf = "org/B-GGUF",  # what the other tab actually loaded
            requested = "org/A-GGUF",  # this tab's stale idea of it
            force = True,
            torn_down = torn_down,
        )
        assert not ev.is_set()
        assert active_generations.count() == 1
    # The resident GGUF was never touched, so nothing was worth cancelling.
    assert "gguf" not in torn_down
    assert response.status == "unloaded"


def test_forced_unload_of_the_loaded_model_still_stops_its_chats(monkeypatch):
    # A real unload must still cancel, or llama-server goes down mid-stream.
    _route_gate()
    import routes.inference as inf_mod

    torn_down: list[str] = []
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = _run_unload(
            inf_mod,
            monkeypatch,
            loaded_gguf = "org/A-GGUF",
            requested = "org/A-GGUF",
            force = True,
            torn_down = torn_down,
        )
        assert ev.is_set()
    assert torn_down == ["gguf"]
    assert response.status == "unloaded"


def test_forced_unload_lets_the_cancelled_chats_unwind_before_teardown(monkeypatch):
    # /unload used to tear down right after the cancel, so a stream told to stop but not yet
    # finished lost its server. Assert the count hits zero BEFORE unload_model runs.
    _route_gate()
    import core.inference.llama_keepwarm as keepwarm
    import routes.inference as inf_mod

    inflight = {"n": 1}
    seen = {}

    def _count(current_request_counted = True, *, include_pending = True):
        # Unwinds one poll after the cancel, like a stream noticing its event.
        if inflight["n"] > 0:
            inflight["n"] -= 1
        return inflight["n"]

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _count)
    monkeypatch.setattr(inf_mod, "_switch_waiter_count", lambda: 0)

    torn_down: list[str] = []
    ev = threading.Event()

    def _record_teardown():
        seen["inflight_at_teardown"] = inflight["n"]
        torn_down.append("gguf")

    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = _run_unload(
            inf_mod,
            monkeypatch,
            loaded_gguf = "org/A-GGUF",
            requested = "org/A-GGUF",
            force = True,
            torn_down = torn_down,
            unload_model = _record_teardown,
        )
        assert ev.is_set()

    assert torn_down == ["gguf"]
    assert seen["inflight_at_teardown"] == 0
    assert response.status == "unloaded"


def test_unload_drains_on_the_middleware_count_not_just_the_registry(monkeypatch):
    # A request past the middleware but not yet at its _TrackedCancel is counted but unregistered, so
    # the drain reads the middleware count, not "did we cancel anything": one poll on a quiet server.
    _route_gate()
    import core.inference.llama_keepwarm as keepwarm
    import routes.inference as inf_mod

    polls = {"n": 0}

    def _count(current_request_counted = True, *, include_pending = True):
        polls["n"] += 1
        return 0

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _count)

    torn_down: list[str] = []
    response = _run_unload(
        inf_mod,
        monkeypatch,
        loaded_gguf = "org/A-GGUF",
        requested = "org/A-GGUF",
        force = True,
        torn_down = torn_down,
    )
    assert torn_down == ["gguf"]
    # Polled, but returned on the first read rather than waiting anything out.
    assert polls["n"] == 1
    assert response.status == "unloaded"


def test_unforced_unload_of_a_stale_model_path_is_still_a_no_op(monkeypatch):
    # Same stale Eject unforced: it reaches no teardown, so refusing strands the stale tab's selection.
    _route_gate()
    import routes.inference as inf_mod

    torn_down: list[str] = []
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = _run_unload(
            inf_mod,
            monkeypatch,
            loaded_gguf = "org/B-GGUF",  # what the other tab actually loaded
            requested = "org/A-GGUF",  # this tab's stale idea of it
            force = False,
            torn_down = torn_down,
        )
        assert not ev.is_set()
        assert active_generations.count() == 1
    # The resident GGUF was untouched; only the standard backend's stale-path no-op ran.
    assert torn_down == ["unsloth"]
    assert response.status == "unloaded"


def test_unforced_unload_of_the_loaded_model_still_refuses_while_chats_stream(monkeypatch):
    # The stale skip above must not disarm the gate for a real replacement.
    _route_gate()
    import routes.inference as inf_mod

    from fastapi import HTTPException

    torn_down: list[str] = []
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            _run_unload(
                inf_mod,
                monkeypatch,
                loaded_gguf = "org/A-GGUF",
                requested = "org/A-GGUF",
                force = False,
                torn_down = torn_down,
            )
    assert exc.value.status_code == 409
    assert exc.value.detail["thread_ids"] == ["t1"]
    assert torn_down == []
    assert not ev.is_set()


def test_unforced_unload_still_refuses_while_a_gguf_load_is_in_flight(monkeypatch):
    # A stale tab's Eject naming the PREVIOUS model while a different one loads. The GGUF branch
    # evicts a live llama-server, so a chat on the previous model must get the 409, not be killed.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from models.inference import UnloadRequest

    torn_down: list[str] = []
    inf_mod, _kw = _stub_unload_backends(
        monkeypatch,
        llama = SimpleNamespace(
            is_active = True,
            is_loaded = False,  # spawned, health check not passed: mid-load
            model_identifier = "org/B-GGUF",
            unload_model = lambda: torn_down.append("gguf"),
        ),
        backend = SimpleNamespace(
            get_loading_model = lambda: None,
            active_model_name = None,
            models = {},
            unload_model = lambda path: torn_down.append("unsloth"),
        ),
    )

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inf_mod.unload_model(
                    UnloadRequest(model_path = "org/A-GGUF", force_cancel_active = False),
                    "tester",
                )
            )
    assert exc.value.status_code == 409
    assert torn_down == []
    assert not ev.is_set()


def test_cancelling_an_in_flight_standard_load_is_not_refused_by_the_chat_gate(monkeypatch):
    # The real cancelLoading shape: unforced /unload naming the still-LOADING model. It replaces
    # nothing, so it cannot interrupt a chat and must not 409 (the frontend would drop the error).
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from models.inference import UnloadRequest

    cancelled: list[str] = []
    torn_down: list[str] = []
    inf_mod, _kw = _stub_unload_backends(
        monkeypatch,
        # Nothing on llama-server: the load in flight is a safetensors one.
        llama = SimpleNamespace(
            is_active = False,
            is_loaded = False,
            model_identifier = None,
            unload_model = lambda: torn_down.append("gguf"),
        ),
        backend = SimpleNamespace(
            get_loading_model = lambda: "org/B",
            cancel_load = lambda path: bool(cancelled.append(path)) or True,
            active_model_name = None,
            models = {},
            unload_model = lambda path: torn_down.append("unsloth"),
        ),
    )

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = asyncio.run(
            inf_mod.unload_model(
                UnloadRequest(model_path = "org/B", force_cancel_active = False), "tester"
            )
        )
        # The chat on the previous model is untouched: the load never reached it.
        assert not ev.is_set()
        assert active_generations.count() == 1
    assert response.status == "unloaded"
    assert cancelled == ["org/B"]
    assert torn_down == []


def test_cancelling_an_in_flight_gguf_load_is_not_refused_by_the_chat_gate(monkeypatch):
    # Same cancelLoading shape on the GGUF fast path: killing that child ends a load, not a chat.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from models.inference import UnloadRequest

    torn_down: list[str] = []
    inf_mod, _kw = _stub_unload_backends(
        monkeypatch,
        llama = SimpleNamespace(
            is_active = True,
            is_loaded = False,  # spawned, health check not passed: mid-load
            model_identifier = "org/B-GGUF",
            unload_model = lambda: torn_down.append("gguf"),
        ),
        backend = SimpleNamespace(
            get_loading_model = lambda: None,
            active_model_name = None,
            models = {},
            unload_model = lambda path: torn_down.append("unsloth"),
        ),
    )

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        response = asyncio.run(
            inf_mod.unload_model(
                UnloadRequest(model_path = "org/B-GGUF", force_cancel_active = False), "tester"
            )
        )
        assert not ev.is_set()
        assert active_generations.count() == 1
    assert response.status == "unloaded"
    assert torn_down == ["gguf"]


def _install_responses_stream_mock(monkeypatch, chunks):
    """Point the direct /v1/responses GGUF pass-through at an in-process
    llama-server. Mirrors the harness in test_responses_tool_passthrough.py."""
    import json
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod

    def handler(request):
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_async_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            context_length = 4096,
            base_url = "http://llama.test",
            supports_reasoning = True,
            reasoning_always_on = False,
            _request_reasoning_kwargs = (
                lambda enable_thinking = None, reasoning_effort = None, preserve_thinking = None: None
            ),
        ),
    )
    return inf_mod


class _NeverDisconnectedRequest:
    async def is_disconnected(self):
        return False


def test_direct_responses_stream_is_visible_to_the_swap_gate(monkeypatch):
    # /v1/responses streams straight to llama-server; unregistered, a non-forced /unload tore it down.
    _route_gate()
    import asyncio

    from models.inference import ChatMessage, ResponsesRequest

    inf_mod = _install_responses_stream_mock(
        monkeypatch, [{"choices": [{"delta": {"content": "33"}}]}]
    )
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]
    seen = {}

    async def run():
        response = await inf_mod._responses_stream(payload, messages, _NeverDisconnectedRequest())
        iterator = response.body_iterator
        await iterator.__anext__()
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        async for _ in iterator:
            pass

    asyncio.run(run())

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    # And it unregisters, or one Codex call would 409 every later reload.
    assert active_generations.count() == 0


def test_forced_reload_stops_a_direct_responses_stream(monkeypatch):
    # The registered event must be the one the stream watches, or a forced reload kills a live decode.
    _route_gate()
    import asyncio

    from models.inference import ChatMessage, ResponsesRequest

    inf_mod = _install_responses_stream_mock(
        monkeypatch,
        [
            {"choices": [{"delta": {"content": "3"}}]},
            {"choices": [{"delta": {"content": "3"}}]},
        ],
    )
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]

    async def run():
        response = await inf_mod._responses_stream(payload, messages, _NeverDisconnectedRequest())
        iterator = response.body_iterator
        chunks = [await iterator.__anext__()]
        assert active_generations.cancel_all() == 1
        async for chunk in iterator:
            chunks.append(chunk)
        return "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

    body = asyncio.run(run())

    # Cancelled mid-stream: the run ends without a completed envelope.
    assert "response.completed" not in body
    assert active_generations.count() == 0


def test_forced_reload_stops_a_responses_stream_still_queued_for_a_slot(monkeypatch):
    # The run registers before it holds a decode slot, so cancel_all() must reach it while queued in
    # admission; watching only the client socket lets it open a generation the swap already revoked.
    _route_gate()
    import asyncio

    from core.inference import llama_admission
    from models.inference import ChatMessage, ResponsesRequest

    for name in (
        llama_admission.ADMISSION_CONTROL_ENV,
        llama_admission.ADMISSION_QUEUE_TIMEOUT_ENV,
        llama_admission.ADMISSION_KEEPALIVE_INTERVAL_ENV,
        llama_admission.ADMISSION_MAX_QUEUE_ENV,
    ):
        monkeypatch.delenv(name, raising = False)

    inf_mod = _install_responses_stream_mock(
        monkeypatch, [{"choices": [{"delta": {"content": "33"}}]}]
    )
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]

    llama_admission.reset_llama_admission_queues()
    try:

        async def run():
            # Hold the backend's only decode slot so the run below has to queue.
            queue = llama_admission.get_llama_admission_queue("http://llama.test")
            holder = queue.reserve(capacity = 1, config = llama_admission.LlamaAdmissionConfig())
            assert holder.lease_nowait() is not None
            response = await inf_mod._responses_stream(
                payload, messages, _NeverDisconnectedRequest()
            )
            chunks = []

            async def drain():
                async for chunk in response.body_iterator:
                    chunks.append(chunk)

            task = asyncio.create_task(drain())
            for _ in range(500):
                if active_generations.count() == 1:
                    break
                await asyncio.sleep(0.01)
            assert active_generations.count() == 1, "the queued run never registered"
            assert active_generations.cancel_all() == 1
            # Unbounded queue by default: without the tracked event this never returns while the slot is held.
            await asyncio.wait_for(task, timeout = 5)
            return chunks

        chunks = asyncio.run(run())
    finally:
        llama_admission.reset_llama_admission_queues()

    body = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)
    # It gave up its place instead of taking the slot: no upstream call, no envelope.
    assert "response.created" not in body
    assert active_generations.count() == 0


def _install_completions_stream_mock(monkeypatch, events):
    """Point the /v1/completions proxy at an in-process llama-server."""
    import json
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod

    def handler(request):
        # One network chunk per SSE event: the relay polls its cancel flag between upstream chunks.
        async def _chunks():
            for event in events:
                yield f"data: {json.dumps(event)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200,
            content = _chunks(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_async_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            context_length = 4096,
            base_url = "http://llama.test",
            model_identifier = "org/M-GGUF",
        ),
    )
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)

    async def _no_auto_switch(request, current_subject, **_kwargs):
        return await request.json()

    monkeypatch.setattr(inf_mod, "_auto_switch_from_request_body", _no_auto_switch)
    return inf_mod


class _CompletionsRequest(_NeverDisconnectedRequest):
    """Minimal stand-in for the Starlette Request /v1/completions reads."""

    def __init__(self, body):
        from types import SimpleNamespace

        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/completions")

    async def json(self):
        return self._body


def test_completions_proxy_stream_is_visible_to_the_swap_gate(monkeypatch):
    # /v1/completions relays from llama-server with no idle drain; unregistered, /unload tore it down.
    _route_gate()
    import asyncio

    inf_mod = _install_completions_stream_mock(monkeypatch, [{"choices": [{"text": "33"}]}])
    request = _CompletionsRequest(
        {"prompt": "hi", "stream": True, "model": "org/M-GGUF", "max_tokens": 8}
    )
    seen = {}

    async def run():
        response = await inf_mod.openai_completions(request, "tester")
        iterator = response.body_iterator
        await iterator.__anext__()
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        async for _ in iterator:
            pass

    asyncio.run(run())

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    # And it unregisters, or one completion would 409 every later reload.
    assert active_generations.count() == 0


def test_forced_reload_stops_a_completions_proxy_stream(monkeypatch):
    # The registered event must be the one the relay watches, or a forced reload kills a live decode.
    _route_gate()
    import asyncio

    inf_mod = _install_completions_stream_mock(
        monkeypatch,
        [{"choices": [{"text": "3"}]}, {"choices": [{"text": "3"}]}],
    )
    request = _CompletionsRequest(
        {"prompt": "hi", "stream": True, "model": "org/M-GGUF", "max_tokens": 8}
    )

    async def run():
        response = await inf_mod.openai_completions(request, "tester")
        iterator = response.body_iterator
        chunks = [await iterator.__anext__()]
        assert active_generations.cancel_all() == 1
        async for chunk in iterator:
            chunks.append(chunk)
        return b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)

    body = asyncio.run(run())

    # Stopped after the first event instead of relaying the rest.
    assert body.count(b'"text"') == 1
    assert active_generations.count() == 0


def test_completions_proxy_non_stream_is_visible_to_the_swap_gate(monkeypatch):
    # ``stream`` defaults to false, so the non-streaming branch is the common shape and holds
    # llama-server throughout: unregistered, /unload counts zero and force_cancel_active has no event.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod

    seen = {}

    def handler(request):
        # Sampled mid-flight: exactly the window a concurrent /unload would tear down in.
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        # And the gate must reach this run, not just see it.
        seen["cancelled"] = active_generations.cancel_all()
        return httpx.Response(200, json = {"id": "cmpl-x", "choices": [{"text": "33"}]})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_async_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    # The pooled client too, so a route that took no per-request one still reaches this transport.
    monkeypatch.setattr(
        inf_mod, "nonstreaming_client", lambda: real_async_client(transport = transport)
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            context_length = 4096,
            base_url = "http://llama.test",
            model_identifier = "org/M-GGUF",
        ),
    )
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)

    async def _no_auto_switch(request, current_subject, **_kwargs):
        return await request.json()

    monkeypatch.setattr(inf_mod, "_auto_switch_from_request_body", _no_auto_switch)

    request = _CompletionsRequest({"prompt": "hi", "model": "org/M-GGUF", "max_tokens": 8})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(inf_mod.openai_completions(request, "tester"))

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert seen["cancelled"] == 1
    # And it unregisters, or one completion would 409 every later reload.
    assert active_generations.count() == 0


class _EmbeddingsRequest(_NeverDisconnectedRequest):
    """Minimal stand-in for the Starlette Request /v1/embeddings reads."""

    def __init__(self, body):
        from types import SimpleNamespace

        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/embeddings")
        self.state = SimpleNamespace(skip_api_monitor = True)

    async def json(self):
        return self._body


def test_embeddings_proxy_is_visible_to_the_swap_gate(monkeypatch):
    # /v1/embeddings holds llama-server for its whole HTTP call: unregistered, a non-forced /unload
    # counts zero and kills the server mid-request (only /load waits on the middleware count).
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod

    seen = {}

    def handler(request):
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        seen["cancelled"] = active_generations.cancel_all()
        return httpx.Response(200, json = {"data": [{"embedding": [0.1, 0.2]}]})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_async_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(
        inf_mod, "nonstreaming_client", lambda: real_async_client(transport = transport)
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            context_length = 4096,
            base_url = "http://llama.test",
            model_identifier = "org/M-GGUF",
        ),
    )
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)

    async def _no_auto_switch(request, current_subject, **_kwargs):
        return await request.json()

    monkeypatch.setattr(inf_mod, "_auto_switch_from_request_body", _no_auto_switch)

    request = _EmbeddingsRequest({"input": "hi", "model": "org/M-GGUF"})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(inf_mod.openai_embeddings(request, "tester"))

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert seen["cancelled"] == 1
    # And it unregisters, or one embedding would 409 every later reload.
    assert active_generations.count() == 0


def test_active_generations_redacts_native_model_paths(monkeypatch):
    # The legacy stream records active_model_name verbatim (an absolute path locally) and is the only
    # place that serialises it: redact like the error paths so a remote client cannot learn host paths.
    _route_gate()
    import asyncio
    import threading
    from types import SimpleNamespace

    import routes.inference as inf_mod
    from utils.native_path_leases import _remember_native_path_for_redaction

    secret_path = "/home/somebody/models/private-model.gguf"
    _remember_native_path_for_redaction(secret_path, "private-model.gguf")

    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4)))
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: SimpleNamespace())

    with active_generations.ActiveGeneration(threading.Event(), thread_id = "t1", model = secret_path):
        body = asyncio.run(inf_mod.get_active_generations(request, "tester"))

    assert body["count"] == 1
    assert secret_path not in str(body)
    assert body["active"][0]["model"] == "<native_path>"


def test_legacy_generate_stream_is_visible_to_the_swap_gate(monkeypatch):
    # The legacy /generate/stream decodes on the standard backend throughout: unregistered it passed
    # the advertised 409 gate then blocked on the generation lock, and a forced swap had no event.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    import routes.inference as inf_mod
    from models.inference import GenerateRequest

    seen = {}

    def _fake_generate_chat_response(**kwargs):
        # Sampled mid-generation: exactly the window an /unload would land in.
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        seen["cancelled"] = active_generations.cancel_all()
        yield "hello"
        yield "world"

    backend = SimpleNamespace(
        active_model_name = "org/M",
        models = {"org/M": {}},
        generate_chat_response = lambda **kw: _fake_generate_chat_response(**kw),
        reset_generation_state = lambda *a: None,
        resize_image = lambda img: img,
    )
    monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: backend)

    async def _drain():
        response = await inf_mod.generate_stream(
            GenerateRequest(messages = [{"role": "user", "content": "hi"}]),
            _NeverDisconnectedRequest(),
            current_subject = "tester",
        )
        async for _ in response.body_iterator:
            pass

    asyncio.run(_drain())

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M"
    assert seen["cancelled"] == 1
    # And it unregisters, or one legacy stream would 409 every later reload.
    assert active_generations.count() == 0


def _anthropic_stream_args(chunks):
    """(request, cancel_event, run_gen) for the local Anthropic stream helpers."""
    cancel_event = threading.Event()

    def run_gen():
        def _gen():
            for chunk in chunks:
                if cancel_event.is_set():
                    return
                yield chunk

        return _gen()

    return _NeverDisconnectedRequest(), cancel_event, run_gen


def test_local_anthropic_plain_stream_is_visible_to_the_swap_gate(monkeypatch):
    # Only the client-tool pass-through registered, so the no-tool /v1/messages path died mid-response.
    _route_gate()
    import asyncio

    import routes.inference as inf_mod

    request, cancel_event, run_gen = _anthropic_stream_args(["3", "33"])
    seen = {}

    async def run():
        response = await inf_mod._anthropic_plain_stream(
            request, cancel_event, run_gen, "msg_1", "org/M-GGUF"
        )
        iterator = response.body_iterator
        await iterator.__anext__()
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        async for _ in iterator:
            pass

    asyncio.run(run())

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert active_generations.count() == 0


def test_forced_reload_stops_a_local_anthropic_plain_stream(monkeypatch):
    # The event registered has to be the one the decode loop watches.
    _route_gate()
    import asyncio

    import routes.inference as inf_mod

    request, cancel_event, run_gen = _anthropic_stream_args(["3", "33", "333"])

    async def run():
        response = await inf_mod._anthropic_plain_stream(
            request, cancel_event, run_gen, "msg_1", "org/M-GGUF"
        )
        iterator = response.body_iterator
        chunks = [await iterator.__anext__()]
        assert active_generations.cancel_all() == 1
        async for chunk in iterator:
            chunks.append(chunk)
        return "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

    body = asyncio.run(run())

    assert cancel_event.is_set()
    # Cancelled mid-stream: no clean message_stop envelope.
    assert "message_stop" not in body
    assert active_generations.count() == 0


def test_local_anthropic_tool_stream_is_visible_to_the_swap_gate(monkeypatch):
    # Same gap on the server-tool path (enable_tools / Anthropic server tools).
    _route_gate()
    import asyncio

    import routes.inference as inf_mod

    request, cancel_event, run_gen = _anthropic_stream_args(
        [{"type": "content", "text": "3"}, {"type": "content", "text": "33"}]
    )
    seen = {}

    async def run():
        response = await inf_mod._anthropic_tool_stream(
            request, cancel_event, run_gen, "msg_1", "org/M-GGUF"
        )
        iterator = response.body_iterator
        await iterator.__anext__()
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        async for _ in iterator:
            pass

    asyncio.run(run())

    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert active_generations.count() == 0


def test_load_and_unload_requests_default_to_not_cancelling():
    pytest.importorskip("pydantic", reason = "pydantic not installed")
    from models.inference import LoadRequest, UnloadRequest

    assert LoadRequest(model_path = "m").force_cancel_active is False
    assert UnloadRequest(model_path = "m").force_cancel_active is False
    assert LoadRequest(model_path = "m", force_cancel_active = True).force_cancel_active is True


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
    assert consts["_PARALLEL_MIN"] <= consts["_PARALLEL_DEFAULT_PLAIN"] <= consts["_PARALLEL_MAX"]


def test_cli_and_backend_parallel_defaults_agree():
    # argparse and the typer CLI are separate entry points into the same server.
    backend = _parallel_constants(os.path.join(_backend, "run.py"))
    cli_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(_backend))),
        "unsloth_cli",
        "commands",
        "studio.py",
    )
    cli = _parallel_constants(cli_path)

    assert cli["_PARALLEL_DEFAULT_PLAIN"] == backend["_PARALLEL_DEFAULT_PLAIN"]


def _run_server_parallel_default(path: str, consts: dict):
    """Resolve run_server()'s llama_parallel_slots default from run.py's source."""
    import ast

    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "run_server":
            continue
        args = node.args.args
        defaults = node.args.defaults
        # defaults align with the tail of the positional arg list.
        for arg, default in zip(args[len(args) - len(defaults) :], defaults):
            if arg.arg != "llama_parallel_slots":
                continue
            if isinstance(default, ast.Constant):
                return default.value
            if isinstance(default, ast.Name):
                return consts.get(default.id)
            return None
    return None


def test_run_server_default_matches_the_cli_parallel_default():
    # colab.py omits llama_parallel_slots, so the signature default is what Colab runs with.
    run_path = os.path.join(_backend, "run.py")
    consts = _parallel_constants(run_path)

    default = _run_server_parallel_default(run_path, consts)

    assert default is not None, "run_server() must keep a llama_parallel_slots default"
    assert default == consts["_PARALLEL_DEFAULT_PLAIN"]
    assert default > 1


def test_colab_launcher_inherits_the_parallel_default():
    # Guard the inheritance itself: an explicit 1 here would resurrect the bug.
    import ast

    colab_path = os.path.join(_backend, "colab.py")
    with open(colab_path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    consts = _parallel_constants(os.path.join(_backend, "run.py"))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_server"
    ]
    assert calls, "colab.py must still launch the backend through run_server()"
    for call in calls:
        for kw in call.keywords:
            if kw.arg != "llama_parallel_slots":
                continue
            value = kw.value.value if isinstance(kw.value, ast.Constant) else None
            assert (
                value is None or value > 1
            ), "colab.py pins llama_parallel_slots to 1; Colab chats would serialise"
    # Whether pinned or inherited, Colab must end up with more than one slot.
    assert consts["_PARALLEL_DEFAULT_PLAIN"] > 1


# ── the point of no return ────────────────────────────────────────────


def test_a_forced_load_that_loses_to_a_sidecar_install_leaves_the_chats_alone(monkeypatch):
    # The destructive cancel is the point of no return: nothing after it may reject the load. A sidecar
    # install can reserve the window during preflight, so its recheck must run before, not after.
    _route_gate()
    import asyncio
    import contextlib
    from types import SimpleNamespace

    from fastapi import HTTPException

    from models.inference import LoadRequest

    inf_mod = _stub_load_route(monkeypatch, active_model_name = "org/OTHER")
    monkeypatch.setattr(inf_mod, "_hf_offline_if_unreachable", contextlib.nullcontext)
    monkeypatch.setattr(
        inf_mod.ModelConfig,
        "from_identifier",
        staticmethod(
            lambda **kwargs: SimpleNamespace(
                is_gguf = False,
                identifier = "org/A",
                display_name = "A",
                is_vision = False,
                is_lora = False,
                path = None,
            )
        ),
    )
    monkeypatch.setattr(inf_mod, "_mlx_distributed_launch_detected", lambda: False)
    monkeypatch.setattr(inf_mod, "_guard_chat_load_against_training", lambda *a, **k: None)
    monkeypatch.setattr(inf_mod, "_resolve_inherited_extra_args", lambda *a, **k: None)

    # The two route-level checks pass, every check after them 409s.
    seen = {"calls": 0}

    def _sidecar_reserved_during_preflight():
        seen["calls"] += 1
        if seen["calls"] > 2:
            raise HTTPException(
                status_code = 409,
                detail = "A transformers installation is in progress. Retry when it completes.",
            )

    monkeypatch.setattr(
        inf_mod, "_raise_if_sidecar_swap_in_progress", _sidecar_reserved_during_preflight
    )

    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )

    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inf_mod.load_model(
                    LoadRequest(
                        model_path = "org/A",
                        load_in_4bit = False,
                        force_cancel_active = True,
                    ),
                    fastapi_request,
                    "tester",
                )
            )
        # The load was rejected, so the chat must still be streaming.
        assert not ev.is_set()
        assert active_generations.count() == 1
    assert exc.value.status_code == 409


def test_anthropic_passthrough_registers_nothing_until_its_body_starts():
    # A pass-through response whose body never starts must leave both registries clean: a never-started
    # async generator runs no body code (PEP 342), so an eagerly entered tracker never unregisters.
    _route_gate()
    import asyncio
    import inspect
    from types import SimpleNamespace

    from starlette.requests import ClientDisconnect

    import routes.inference as inf_mod

    llama_backend = SimpleNamespace(
        base_url = "http://127.0.0.1:8080",
        context_length = 4096,
        count_chat_tokens = lambda messages, _unused, tools, **_kwargs: 7,
    )

    async def _build():
        return await inf_mod._anthropic_passthrough_stream(
            SimpleNamespace(),
            threading.Event(),
            llama_backend,
            [{"role": "user", "content": "hi"}],
            [],
            0.7,
            0.9,
            40,
            128,
            "msg_1",
            "org/A",
            session_id = "s1",
            cancel_id = "c1",
        )

    # Built and abandoned, as when the request task is cancelled before Starlette calls the response.
    asyncio.run(_build())
    assert active_generations.count() == 0
    assert not inf_mod._CANCEL_REGISTRY

    # The client is gone at header time, so the first send fails and the body generator never runs.
    async def _drive():
        response = await _build()

        async def _receive():
            return {"type": "http.disconnect"}

        async def _send(message):
            raise OSError("client disconnected")

        with pytest.raises(ClientDisconnect):
            await response({"type": "http"}, _receive, _send)

    asyncio.run(_drive())
    assert active_generations.count() == 0
    assert not inf_mod._CANCEL_REGISTRY

    # Still tracked once the body runs: the enter stays inside the generator, under the finally.
    src = inspect.getsource(inf_mod._anthropic_passthrough_stream)
    assert src.index("async def _stream()") < src.index("_tracker.__enter__()")
    assert src.index("_tracker.__enter__()") < src.index("_tracker.__exit__(None, None, None)")


def test_audio_generation_is_visible_to_the_swap_gate(monkeypatch):
    # /audio/generate is non-streaming and holds the model for the whole request: unregistered, a
    # non-forced swap counted zero and could tear it down mid-TTS, and a forced one had no entry.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest

    seen = {}

    class _TtsBackend:
        active_model_name = "org/TTS"
        models = {"org/TTS": {"is_audio": True, "audio_type": "snac"}}

        def generate_audio_response(self, **kwargs):
            # Sampled mid-generation: the window a concurrent swap would tear down in.
            seen["count"] = active_generations.count()
            seen["snapshot"] = active_generations.snapshot()
            return (b"RIFFfake", 24000)

    # is_loaded False picks the transformers TTS branch, not the GGUF one.
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, _is_audio = False),
    )
    monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: _TtsBackend())

    async def _no_auto_switch(*a, **k):
        return None

    monkeypatch.setattr(inf_mod, "_maybe_auto_switch_model", _no_auto_switch)

    payload = ChatCompletionRequest(
        model = "org/TTS",
        messages = [{"role": "user", "content": "hi"}],
        thread_id = "thread-tts",
    )
    asyncio.run(inf_mod.generate_audio(payload, request = None, current_subject = "tester"))

    assert seen["count"] == 1
    # Named, so the swap dialog can say which chat it would interrupt.
    assert seen["snapshot"][0]["thread_id"] == "thread-tts"
    # And it unregisters, or one TTS call would 409 every later reload.
    assert active_generations.count() == 0


class _ChatRequest(_NeverDisconnectedRequest):
    """Minimal stand-in for the Starlette Request /v1/chat/completions reads."""

    def __init__(self):
        from types import SimpleNamespace

        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/chat/completions")
        self.state = SimpleNamespace(skip_api_monitor = True)
        self.scope: dict = {}


def _standard_chat_stubs(monkeypatch, backend):
    """Point /v1/chat/completions at a standard (non-GGUF) backend.

    ``supports_tools`` False keeps the request off the safetensors server-tool
    loop, which registers on its own, so the plain default branch is exercised.
    """
    from types import SimpleNamespace

    import routes.inference as inf_mod

    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = False,
            supports_tools = False,
            is_vision = False,
            context_length = None,
        ),
    )
    monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(
        inf_mod, "_detect_safetensors_features", lambda *a, **k: {"supports_tools": False}
    )

    async def _no_auto_switch(*a, **k):
        return None

    monkeypatch.setattr(inf_mod, "_maybe_auto_switch_model", _no_auto_switch)
    return inf_mod


def test_standard_non_stream_chat_is_visible_to_the_swap_gate(monkeypatch):
    # ``stream`` defaults to false, so this is the default shape of a standard chat and it holds the
    # worker throughout. Only the streaming branch registered, so a swap truncated the completion.
    _route_gate()
    import asyncio

    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest

    seen = {}

    class _StandardBackend:
        active_model_name = "org/M"
        models = {"org/M": {"chat_template_info": {"template": "chatml"}}}

        def generate_chat_response(
            self,
            *,
            cancel_event = None,
            stats_holder = None,
            **kwargs,
        ):
            # Sampled mid-generation: exactly the window an /unload lands in.
            seen["count"] = active_generations.count()
            seen["snapshot"] = active_generations.snapshot()
            # And the gate must reach this run, on the event the decode watches.
            seen["cancelled"] = active_generations.cancel_all()
            seen["reached_the_decode"] = cancel_event is not None and cancel_event.is_set()
            yield "33"

        def reset_generation_state(self, caller_cancel_event = None):
            pass

    _standard_chat_stubs(monkeypatch, _StandardBackend())

    payload = ChatCompletionRequest(
        model = "org/M",
        messages = [{"role": "user", "content": "hi"}],
        thread_id = "thread-chat",
    )
    response = asyncio.run(
        inf_mod.openai_chat_completions(payload, _ChatRequest(), current_subject = "tester")
    )

    assert response.status_code == 200
    assert seen["count"] == 1
    # Named, so the swap dialog can say which chat it would interrupt.
    assert seen["snapshot"][0]["thread_id"] == "thread-chat"
    assert seen["cancelled"] == 1
    assert seen["reached_the_decode"]
    # And it unregisters, or one completion would 409 every later reload.
    assert active_generations.count() == 0


def test_standard_non_stream_chat_unregisters_when_it_fails(monkeypatch):
    # A raising backend must not strand an entry: that would 409 every later swap.
    _route_gate()
    import asyncio

    from fastapi import HTTPException

    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest

    class _BrokenBackend:
        active_model_name = "org/M"
        models = {"org/M": {"chat_template_info": {"template": "chatml"}}}

        def generate_chat_response(self, **kwargs):
            raise RuntimeError("decode exploded")
            yield  # pragma: no cover - generator marker

        def reset_generation_state(self, caller_cancel_event = None):
            pass

    _standard_chat_stubs(monkeypatch, _BrokenBackend())

    payload = ChatCompletionRequest(model = "org/M", messages = [{"role": "user", "content": "hi"}])
    with pytest.raises(HTTPException):
        asyncio.run(
            inf_mod.openai_chat_completions(payload, _ChatRequest(), current_subject = "tester")
        )

    assert active_generations.count() == 0


def test_audio_input_non_stream_chat_is_visible_to_the_swap_gate(monkeypatch):
    # An audio-input model with the default stream=false holds the standard worker throughout. Only
    # the streaming sibling registered, so a non-forced swap could unload it mid-transcription.
    _route_gate()
    import asyncio

    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest

    seen = {}

    class _AudioInputBackend:
        active_model_name = "org/AUDIO-IN"
        models = {"org/AUDIO-IN": {"has_audio_input": True}}

        def generate_audio_input_response(
            self,
            *,
            cancel_event = None,
            **kwargs,
        ):
            # Sampled mid-transcription: the window a concurrent swap lands in.
            seen["count"] = active_generations.count()
            seen["snapshot"] = active_generations.snapshot()
            seen["cancelled"] = active_generations.cancel_all()
            seen["reached_the_decode"] = cancel_event is not None and cancel_event.is_set()
            yield "33"

        def reset_generation_state(self, caller_cancel_event = None):
            pass

    _standard_chat_stubs(monkeypatch, _AudioInputBackend())
    monkeypatch.setattr(inf_mod, "_decode_audio_base64", lambda _b64: object())

    payload = ChatCompletionRequest(
        model = "org/AUDIO-IN",
        messages = [{"role": "user", "content": "transcribe this"}],
        audio_base64 = "ZmFrZQ==",
        thread_id = "thread-audio-in",
    )
    response = asyncio.run(
        inf_mod.openai_chat_completions(payload, _ChatRequest(), current_subject = "tester")
    )

    assert response.status_code == 200
    assert seen["count"] == 1
    assert seen["snapshot"][0]["thread_id"] == "thread-audio-in"
    assert seen["cancelled"] == 1
    assert seen["reached_the_decode"]
    # And it unregisters, or one transcription would 409 every later reload.
    assert active_generations.count() == 0


def _anthropic_route_stubs(monkeypatch, **overrides):
    """Minimal GGUF backend + request stub for the /v1/messages route."""
    from types import SimpleNamespace

    import routes.inference as inf_mod
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        supports_tool_passthrough = True,
        model_identifier = "org/M-GGUF",
        base_url = "http://llama.test",
        context_length = 4096,
        count_chat_tokens = lambda *a, **k: 2,
    )
    backend.__dict__.update(overrides)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)
    return inf_mod


class _MessagesRequest(_NeverDisconnectedRequest):
    """Minimal stand-in for the Starlette Request /v1/messages reads."""

    def __init__(self):
        from types import SimpleNamespace

        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/messages")
        self.state = SimpleNamespace(skip_api_monitor = True)


@pytest.mark.parametrize("with_server_tools", [False, True])
def test_local_anthropic_non_stream_is_visible_to_the_swap_gate(monkeypatch, with_server_tools):
    # ``stream`` defaults to false on /v1/messages, so the non-streaming plain and server-tool branches
    # are the common shape and decode throughout. Only their streaming siblings registered.
    _route_gate()
    import asyncio

    from models.inference import AnthropicMessagesRequest

    seen = {}

    def _sample():
        # Sampled mid-generation: exactly the window an /unload lands in.
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        seen["cancelled"] = active_generations.cancel_all()

    def _gen_plain(*, cancel_event = None, **kwargs):
        _sample()
        seen["reached_the_decode"] = cancel_event is not None and cancel_event.is_set()
        yield "ok"

    def _gen_tools(*, cancel_event = None, **kwargs):
        _sample()
        seen["reached_the_decode"] = cancel_event is not None and cancel_event.is_set()
        yield {"type": "content", "text": "ok"}

    inf_mod = _anthropic_route_stubs(
        monkeypatch,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
    )

    fields = {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    if with_server_tools:
        fields["enable_tools"] = True
        fields["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    payload = AnthropicMessagesRequest(**fields)

    response = asyncio.run(
        inf_mod.anthropic_messages(payload, request = _MessagesRequest(), current_subject = "tester")
    )

    assert response.status_code == 200
    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert seen["cancelled"] == 1
    # The event registered is the one the decode watches, so a forced swap lands.
    assert seen["reached_the_decode"]
    # And it unregisters, or one message would 409 every later reload.
    assert active_generations.count() == 0


def test_anthropic_passthrough_non_stream_is_visible_to_the_swap_gate(monkeypatch):
    # The client-tool pass-through holds llama-server for one non-streaming POST. Its streaming sibling
    # registers inside the body generator; this branch had none, so /unload tore the server down.
    _route_gate()
    import asyncio

    import httpx

    from models.inference import AnthropicMessagesRequest

    seen = {}

    def handler(request):
        seen["count"] = active_generations.count()
        seen["snapshot"] = active_generations.snapshot()
        seen["cancelled"] = active_generations.cancel_all()
        return httpx.Response(
            200,
            json = {
                "choices": [
                    {"message": {"role": "assistant", "content": "33"}, "finish_reason": "stop"}
                ]
            },
        )

    inf_mod = _anthropic_route_stubs(monkeypatch)
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    # The pass-through takes a per-request client, so a Stop or forced swap can close it mid-POST.
    monkeypatch.setattr(
        inf_mod,
        "_cancelable_nonstreaming_client",
        lambda: real_async_client(transport = transport),
    )

    # enable_tools False keeps the server-tool loop out, so the client tool takes the pass-through.
    payload = AnthropicMessagesRequest(
        max_tokens = 16,
        messages = [{"role": "user", "content": "hi"}],
        enable_tools = False,
        tools = [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
    )

    response = asyncio.run(
        inf_mod.anthropic_messages(payload, request = _MessagesRequest(), current_subject = "tester")
    )

    assert response.status_code == 200
    assert seen["count"] == 1
    assert seen["snapshot"][0]["model"] == "org/M-GGUF"
    assert seen["cancelled"] == 1
    # And it unregisters, or one message would 409 every later reload.
    assert active_generations.count() == 0


def test_anthropic_passthrough_non_stream_stops_when_the_swap_cancels_it(monkeypatch):
    # Registering is half the job: a pooled client cannot be closed, so the run was cancelled while the
    # POST carried on. The watcher closes a per-request client; the set event makes that error a cancel.
    _route_gate()
    import asyncio

    import httpx

    from models.inference import AnthropicMessagesRequest

    seen = {}

    def handler(request):
        # Stand in for a forced swap mid-decode: cancel, then fail the transport as closing would.
        seen["cancelled"] = active_generations.cancel_all()
        raise httpx.ConnectError("client closed")

    inf_mod = _anthropic_route_stubs(monkeypatch)
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod,
        "_cancelable_nonstreaming_client",
        lambda: real_async_client(transport = transport),
    )

    payload = AnthropicMessagesRequest(
        max_tokens = 16,
        messages = [{"role": "user", "content": "hi"}],
        enable_tools = False,
        tools = [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            inf_mod.anthropic_messages(
                payload, request = _MessagesRequest(), current_subject = "tester"
            )
        )

    assert seen["cancelled"] == 1
    # Cancelled or not, the entry must go, or one message 409s every later reload.
    assert active_generations.count() == 0


def test_audio_generation_unregisters_when_it_fails(monkeypatch):
    # A raising backend must not strand an entry: that would 409 every later load.
    _route_gate()
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest

    class _BrokenTtsBackend:
        active_model_name = "org/TTS"
        models = {"org/TTS": {"is_audio": True, "audio_type": "snac"}}

        def generate_audio_response(self, **kwargs):
            raise RuntimeError("codec exploded")

    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, _is_audio = False),
    )
    monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: _BrokenTtsBackend())

    async def _no_auto_switch(*a, **k):
        return None

    monkeypatch.setattr(inf_mod, "_maybe_auto_switch_model", _no_auto_switch)

    payload = ChatCompletionRequest(
        model = "org/TTS",
        messages = [{"role": "user", "content": "hi"}],
    )
    with pytest.raises(HTTPException):
        asyncio.run(inf_mod.generate_audio(payload, request = None, current_subject = "tester"))

    assert active_generations.count() == 0


# ── sidecar install: carrying a confirmed swap through ─────────────────


def _stub_install_route(monkeypatch, *, in_flight_events):
    """Point POST /install-latest-transformers at an in-memory sidecar install.

    ``in_flight_events`` stands in for the middleware's in-flight count: a
    request is counted until its stream observes the cancel event and unwinds,
    which is the coupling the installer's guard actually reads.
    """
    from types import SimpleNamespace

    import core.inference.llama_keepwarm as keepwarm
    import routes.inference as inf_mod
    import utils.transformers_latest as latest_mod
    import utils.transformers_version as version_mod

    calls = {"installed": [], "released": 0}

    monkeypatch.setattr(version_mod, "try_begin_sidecar_swap", lambda: True)

    def _end_sidecar_swap():
        calls["released"] += 1

    monkeypatch.setattr(version_mod, "end_sidecar_swap", _end_sidecar_swap)

    import core.export as export_mod
    import core.training as training_mod

    monkeypatch.setattr(
        training_mod,
        "get_training_backend",
        lambda: SimpleNamespace(is_training_active = lambda: False),
    )
    monkeypatch.setattr(
        export_mod,
        "get_export_backend",
        lambda: SimpleNamespace(is_export_active = lambda: False, current_checkpoint = None),
    )
    monkeypatch.setattr(
        inf_mod,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None, load_generation = 0),
    )

    def _fake_in_flight(current_request_counted = True, *, include_pending = True):
        return sum(1 for ev in in_flight_events if not ev.is_set())

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _fake_in_flight)

    def _install(version, before_swap, *args, **kwargs):
        calls["installed"].append(version)
        return {"success": True, "version": version, "message": "installed"}

    monkeypatch.setattr(latest_mod, "install_latest_transformers", _install)
    return inf_mod, calls


def test_confirmed_install_stops_the_chats_it_was_given_permission_to_stop(monkeypatch):
    # The install sits between the swap's "stop N chats" prompt and the /load carrying the
    # confirmation, and refuses while those chats run, so a confirmed install cancels them itself.
    _route_gate()
    import asyncio

    from models.inference import InstallLatestTransformersRequest

    ev = threading.Event()
    inf_mod, calls = _stub_install_route(monkeypatch, in_flight_events = [ev])

    with active_generations.ActiveGeneration(ev, thread_id = "t1", model = "org/M-GGUF"):
        response = asyncio.run(
            inf_mod.install_latest_transformers_route(
                InstallLatestTransformersRequest(version = "5.0.0", force_cancel_active = True),
                "tester",
            )
        )
        assert ev.is_set()

    assert response.success is True
    assert calls["installed"] == ["5.0.0"]


def test_unconfirmed_install_still_refuses_while_chats_stream(monkeypatch):
    # Unchanged for every caller that never confirmed (second tab, desktop, curl): no flag, no cancel.
    _route_gate()
    import asyncio

    from fastapi import HTTPException

    from models.inference import InstallLatestTransformersRequest

    ev = threading.Event()
    inf_mod, calls = _stub_install_route(monkeypatch, in_flight_events = [ev])

    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inf_mod.install_latest_transformers_route(
                    InstallLatestTransformersRequest(version = "5.0.0"),
                    "tester",
                )
            )
        assert not ev.is_set()
        assert active_generations.count() == 1

    assert exc.value.status_code == 409
    assert calls["installed"] == []


def test_a_confirmed_install_that_cannot_drain_refuses_instead_of_swapping(monkeypatch):
    # A cancelled request that never observes its event keeps the in-flight count up, so the drain is
    # bounded and cannot wedge the process holding the gate; the recheck behind it still refuses.
    _route_gate()
    import asyncio

    from fastapi import HTTPException

    from models.inference import InstallLatestTransformersRequest

    ev = threading.Event()
    stuck = threading.Event()
    stuck.set()  # already "cancelled", yet still counted: it never unwinds
    inf_mod, calls = _stub_install_route(monkeypatch, in_flight_events = [ev, stuck])
    monkeypatch.setattr(inf_mod, "_POST_CANCEL_DRAIN_TIMEOUT_S", 0.05)

    def _never_unwinds(current_request_counted = True, *, include_pending = True):
        return 1

    import core.inference.llama_keepwarm as keepwarm

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _never_unwinds)

    async def _install():
        # Deadline here too: a regression that drops the drain's bound must fail, not hang the suite.
        return await asyncio.wait_for(
            inf_mod.install_latest_transformers_route(
                InstallLatestTransformersRequest(version = "5.0.0", force_cancel_active = True),
                "tester",
            ),
            timeout = 5,
        )

    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_install())

    assert exc.value.status_code == 409
    assert calls["installed"] == []


def test_confirmed_install_does_not_spend_its_cancel_on_an_install_that_will_refuse(monkeypatch):
    # An unrelated counted request the cancel cannot stop must be waited out BEFORE the cancel: the
    # recheck refuses while it is there, so cancelling first stopped chats for a doomed install.
    _route_gate()
    import asyncio

    from fastapi import HTTPException

    from models.inference import InstallLatestTransformersRequest

    ev = threading.Event()
    inf_mod, calls = _stub_install_route(monkeypatch, in_flight_events = [ev])

    import core.inference.llama_keepwarm as keepwarm

    def _never_drains(current_request_counted = True, *, include_pending = True):
        # Discounting the registered chat still leaves the counted-only stranger: the drain must not clear.
        return 2

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _never_drains)
    monkeypatch.setattr(inf_mod, "_POST_CANCEL_DRAIN_TIMEOUT_S", 0.05)

    async def _install():
        return await asyncio.wait_for(
            inf_mod.install_latest_transformers_route(
                InstallLatestTransformersRequest(version = "5.0.0", force_cancel_active = True),
                "tester",
            ),
            timeout = 5,
        )

    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_install())
        # The refusal is the same as before; what changed is that the chat lives.
        assert not ev.is_set()
        assert active_generations.count() == 1

    assert exc.value.status_code == 409
    assert calls["installed"] == []


# ── draining before teardown ──────────────────────────────────────────


def _drain_with_counts(monkeypatch, counts, **kwargs):
    """Run _wait_for_model_switch_idle against a scripted in-flight count.

    ``counts`` is consumed one entry per poll; the last value repeats, so a
    trailing non-zero stands for a request that never unwinds.
    """
    _route_gate()
    import asyncio

    import core.inference.llama_keepwarm as keepwarm
    import routes.inference as inf_mod

    remaining = list(counts)
    polls = {"n": 0}

    def _count(current_request_counted = True, *, include_pending = True):
        polls["n"] += 1
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _count)
    monkeypatch.setattr(inf_mod, "_switch_waiter_count", lambda: 0)

    async def _run():
        # Hard test-side deadline: a drain that regresses to waiting forever must fail red, not hang.
        await asyncio.wait_for(
            inf_mod._wait_for_model_switch_idle(current_request_counted = False, **kwargs),
            timeout = 5,
        )

    asyncio.run(_run())
    return polls["n"]


def test_forced_swap_does_not_wait_out_the_generations_it_is_about_to_cancel(monkeypatch):
    # cancel_pending discounts the registered generations, since the caller cancels them right after.
    # Drop the discount and the drain waits on a count only that pending cancel can lower: forever.
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        polls = _drain_with_counts(monkeypatch, [1], cancel_pending = True)
    assert polls == 1


def test_the_same_drain_without_the_discount_would_keep_waiting(monkeypatch):
    # The other half: that count really does block, so the previous test passes by the discount.
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev, thread_id = "t1"):
        polls = _drain_with_counts(monkeypatch, [1], timeout_s = 0.05)
    assert polls > 1


def test_post_cancel_drain_gives_up_on_a_request_that_never_unwinds(monkeypatch):
    # TTS on the subprocess backend observes no cancel event, so a forced swap can cancel it and still
    # see it counted forever. The post-cancel drains hold the gate, so they must expire and proceed.
    polls = _drain_with_counts(monkeypatch, [1], timeout_s = 0.05)
    assert polls > 1


def test_drain_returns_as_soon_as_the_cancelled_requests_unwind(monkeypatch):
    # The bound is a backstop: once the count drops the drain returns without sitting out the timeout.
    polls = _drain_with_counts(monkeypatch, [2, 1, 0], timeout_s = 30)
    assert polls == 3


# ── queued chats must not cancel the running one ──────────────────────


def _orchestrator_for_ownership():
    """A real InferenceOrchestrator with just enough stubbed to drive the lock."""
    _route_gate()
    orch_mod = pytest.importorskip(
        "core.inference.orchestrator", reason = "inference stack not installed"
    )
    orch = orch_mod.InferenceOrchestrator.__new__(orch_mod.InferenceOrchestrator)
    orch._gen_lock = threading.Lock()
    orch._active_cancel_events = []
    orch._executing_cancel_events = []
    orch._active_cancel_lock = threading.Lock()
    orch._cancel_event = threading.Event()
    orch._ensure_subprocess_alive = lambda: False  # stop before _send_cmd
    return orch


def test_a_queued_chat_cannot_reset_the_chat_that_is_generating():
    # Safetensors generation serialises on _gen_lock and the worker has ONE cancel event: stopping
    # queued chat B reset that shared event and killed running chat A. Scope the reset to the holder.
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()

    orch._claim_worker(a_event)  # A holds the lock ...
    orch._mark_worker_started(a_event)  # ... and the worker is answering it
    orch.reset_generation_state(b_event)  # B is queued and gets stopped
    assert not orch._cancel_event.is_set()

    orch.reset_generation_state(a_event)  # A's own Stop still works
    assert orch._cancel_event.is_set()


def test_a_global_reset_still_cancels_whatever_is_running():
    # Unload and switch pass nothing: they mean stop everything, else a generation survives teardown.
    orch = _orchestrator_for_ownership()
    _running = threading.Event()
    orch._claim_worker(_running)
    orch._mark_worker_started(_running)
    orch.reset_generation_state()
    assert orch._cancel_event.is_set()


def test_a_reset_with_no_generation_running_is_not_dropped():
    # Nothing holds the lock, so no chat to protect: a reset before any generation must still run.
    orch = _orchestrator_for_ownership()
    orch.reset_generation_state(threading.Event())
    assert orch._cancel_event.is_set()


def test_unload_waits_for_a_request_that_is_admitted_but_not_yet_registered(monkeypatch):
    # The window between the keep-warm middleware and _TrackedCancel: counted in-flight, absent from
    # the registry. Cancelling on the registry alone tore the backend down under an admitted request.
    _route_gate()
    import core.inference.llama_keepwarm as keepwarm
    import routes.inference as inf_mod

    # Counted for two polls, then the request registers/finishes and clears.
    remaining = [1, 1, 0]
    seen = {}

    def _count(current_request_counted = True, *, include_pending = True):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(keepwarm, "other_inference_request_count", _count)
    monkeypatch.setattr(inf_mod, "_switch_waiter_count", lambda: 0)

    torn_down: list[str] = []

    def _record_teardown():
        seen["counted_at_teardown"] = remaining[0]
        torn_down.append("gguf")

    # Registry deliberately empty: this is the unregistered case.
    response = _run_unload(
        inf_mod,
        monkeypatch,
        loaded_gguf = "org/A-GGUF",
        requested = "org/A-GGUF",
        force = True,
        torn_down = torn_down,
        unload_model = _record_teardown,
    )

    assert active_generations.count() == 0
    assert torn_down == ["gguf"]
    assert seen["counted_at_teardown"] == 0
    assert response.status == "unloaded"


def test_a_dispatched_chat_cannot_reset_its_concurrently_dispatched_sibling():
    # Compare-mode / dispatched runs bypass _gen_lock and run concurrently, so with several claimed
    # at once a Stop on one must still leave the others alone.
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()
    c_event = threading.Event()

    orch._claim_worker(a_event)
    orch._mark_worker_started(a_event)
    orch._claim_worker(b_event)
    orch._mark_worker_started(b_event)

    orch.reset_generation_state(c_event)  # a third, unrelated request
    assert not orch._cancel_event.is_set()

    orch.reset_generation_state(b_event)  # one of the running pair
    assert orch._cancel_event.is_set()


def test_releasing_one_generation_leaves_the_other_claimed():
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()
    orch._claim_worker(a_event)
    orch._mark_worker_started(a_event)
    orch._claim_worker(b_event)
    orch._mark_worker_started(b_event)
    orch._release_worker(a_event)

    orch.reset_generation_state(a_event)  # now a stranger
    assert not orch._cancel_event.is_set()

    orch._release_worker(b_event)
    orch.reset_generation_state(a_event)  # nothing running: no one to protect
    assert orch._cancel_event.is_set()


def test_a_dispatched_request_queued_behind_another_is_not_an_owner():
    # The subprocess runs generations one at a time, so admission is not execution: B can be claimed
    # while the worker answers A. Counting B as an owner let its Stop signal the shared event and end A.
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()

    orch._claim_worker(a_event)
    orch._mark_worker_started(a_event)  # the worker answered A
    orch._claim_worker(b_event)  # B is only queued behind it

    orch.reset_generation_state(b_event)
    assert not orch._cancel_event.is_set(), "a queued request must not reset A"

    orch._mark_worker_started(b_event)  # the worker moves on to B
    orch.reset_generation_state(b_event)
    assert orch._cancel_event.is_set()


def test_a_queued_request_cannot_reset_during_the_other_ones_prefill():
    # Between _send_cmd and the first response A is claimed but not executing; treating that as
    # "nobody to protect" let a queued request's Stop kill A mid-prefill.
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()

    orch._claim_worker(a_event)  # A sent its command and is in prefill
    orch._claim_worker(b_event)  # B is queued behind it

    orch.reset_generation_state(b_event)
    assert not orch._cancel_event.is_set(), "B must not reset A during prefill"

    # A's own Stop still works before any token has arrived.
    orch.reset_generation_state(a_event)
    assert orch._cancel_event.is_set()


def test_the_oldest_claim_is_the_one_the_worker_is_prefilling():
    # The command queue is FIFO, so with nothing answering the oldest claim is the executor.
    orch = _orchestrator_for_ownership()
    a_event = threading.Event()
    b_event = threading.Event()
    orch._claim_worker(a_event)
    orch._claim_worker(b_event)
    orch._release_worker(a_event)

    orch.reset_generation_state(b_event)
    assert orch._cancel_event.is_set(), "B is now the oldest claim"


def test_claim_order_matches_send_order_under_concurrent_dispatch():
    # _owns_worker reads claim order to decide who is prefilling, so a claim not atomic with the
    # enqueue can put A first in the list while B is first in the subprocess queue: stopping A kills B.
    _route_gate()
    orch_mod = pytest.importorskip(
        "core.inference.orchestrator", reason = "inference stack not installed"
    )
    orch = orch_mod.InferenceOrchestrator.__new__(orch_mod.InferenceOrchestrator)
    orch._active_cancel_events = []
    orch._executing_cancel_events = []
    orch._active_cancel_lock = threading.Lock()
    orch._send_order_lock = threading.Lock()

    sent: list = []
    barrier = threading.Barrier(4)

    def worker(ev):
        barrier.wait(timeout = 10)
        with orch._send_order_lock:
            orch._claim_worker(ev)
            # Stand in for _send_cmd: the enqueue must not be separable from the claim.
            sent.append(ev)

    events = [threading.Event() for _ in range(4)]
    threads = [threading.Thread(target = worker, args = (e,)) for e in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 30)

    assert orch._active_cancel_events == sent, "claim order must equal send order"


def test_responses_stream_reports_reasoning_ttft_and_stop_reason(monkeypatch):
    # This adapter parses SSE itself, so without its own stamp a reasoning-first
    # turn would time from the visible text instead.
    import asyncio

    from core.inference.api_monitor import api_monitor
    from models.inference import ChatMessage, ResponsesRequest

    inf_mod = _install_responses_stream_mock(
        monkeypatch,
        [
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": "length"}]},
        ],
    )
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]

    monitor_id = api_monitor.start(
        endpoint = "/v1/responses", method = "POST", model = "org/M-GGUF", prompt = "hi"
    )

    async def run():
        response = await inf_mod._responses_stream(
            payload, messages, _NeverDisconnectedRequest(), monitor_id
        )
        async for _ in response.body_iterator:
            pass

    asyncio.run(run())

    rows = [r for r in api_monitor.snapshot() if r["id"] == monitor_id]
    assert rows, "the stream should have opened a monitor row"
    assert rows[0]["ttft_ms"] is not None
    assert rows[0]["stop_reason"] == "length"


def test_responses_stream_stamps_tool_call_deltas(monkeypatch):
    # A tool-call-opening turn already sent client output, so TTFT must stamp there.
    import asyncio

    from core.inference.api_monitor import api_monitor
    from models.inference import ChatMessage, ResponsesRequest

    inf_mod = _install_responses_stream_mock(
        monkeypatch,
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "f", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ],
    )
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]
    monitor_id = api_monitor.start(
        endpoint = "/v1/responses", method = "POST", model = "org/M-GGUF", prompt = "hi"
    )
    # append_reply would stamp late; assert it happens at the delta instead.
    stamped: list[str] = []
    real_mark = api_monitor.mark_first_token
    monkeypatch.setattr(
        api_monitor,
        "mark_first_token",
        lambda mid: (stamped.append(mid), real_mark(mid))[1],
    )

    async def run():
        response = await inf_mod._responses_stream(
            payload, messages, _NeverDisconnectedRequest(), monitor_id
        )
        async for _ in response.body_iterator:
            pass

    asyncio.run(run())

    assert stamped, "the tool-call delta should stamp the first token"
    [row] = [r for r in api_monitor.snapshot() if r["id"] == monitor_id]
    assert row["ttft_ms"] is not None
    assert row["stop_reason"] == "tool_calls"
