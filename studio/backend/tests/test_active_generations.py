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


# The registry needs only the stdlib, so it runs anywhere. The gate lives in
# routes.inference, which pulls the whole inference stack: skip when absent.
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
    # Re-Applying the model that is already loaded returns already_loaded without
    # touching llama-server, so it must not 409 (nor, on the retry the 409 tells
    # the caller to send, stop every chat for a no-op).
    _route_gate()  # skips when the inference stack is unavailable
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
    # The gate itself is intact: a load that would actually replace the running
    # model still 409s, and still names the chats it would stop.
    _route_gate()  # skips when the inference stack is unavailable
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
    # /load repeats the check under the gate; /unload has to as well, or a chat
    # that starts while this request queues on the gate is torn down mid-stream.
    _route_gate()  # skips when the inference stack is unavailable
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
    # /v1/responses streams straight to llama-server (no chat pass-through), so
    # without its own registration a non-forced /unload saw zero generations and
    # tore the server down mid-response.
    _route_gate()  # skips when the inference stack is unavailable
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
    # The registered event has to be the one the stream actually watches, or
    # force_cancel_active would unload the server while it keeps decoding.
    _route_gate()  # skips when the inference stack is unavailable
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
    # colab.py calls run_server() without llama_parallel_slots, so the
    # signature default is what Colab actually runs with. At 1 the admission
    # queue serialises every chat there despite the parallel-chat UI.
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
