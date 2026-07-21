"""Cancel tracker must register BEFORE StreamingResponse returns and clean up
in each async generator's `finally`, else a Stop before the first SSE chunk
leaves a zombie decode (a BackgroundTask would be skipped when stream_response raises).

Structural verifies registration placement and try/finally cleanup; behavioral
verifies the extracted `_TrackedCancel` cleans up across completion/OSError/aclose
and that a pre-set cancel_event breaks the GGUF loop cleanly with final_chunk + [DONE]."""

from __future__ import annotations

import ast
import asyncio
import json
import threading
import time
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
_TREE = ast.parse(SRC)


# ── Structural (AST) helpers ─────────────────────────────────


def _collect_async_functions(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]


def _has_tracker_enter_call(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "__enter__"
            and isinstance(fn.value, ast.Name)
            and fn.value.id.startswith("_tracker")
        ):
            return True
    return False


def _finalbody_has_tracker_exit(finalbody) -> bool:
    for stmt in finalbody:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        fn = call.func
        if (
            fn.attr == "__exit__"
            and isinstance(fn.value, ast.Name)
            and fn.value.id.startswith("_tracker")
        ):
            return True
    return False


def _async_function(name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} handler missing")


def _calls_name(node: ast.AST, name: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id == name:
                return True
    return False


# ── Structural tests ─────────────────────────────────────────


def test_no_tracker_enter_inside_async_generators():
    offenders = []
    for fn in _collect_async_functions(_TREE):
        if fn.name in {
            "gguf_tool_stream",
            "gguf_stream_chunks",
            "stream_chunks",
            "audio_input_stream",
        }:
            if _has_tracker_enter_call(fn):
                offenders.append(fn.name)
    assert not offenders, (
        f"Cancel tracker registration must live OUTSIDE the async generator "
        f"body so a stop POST can find the registry entry before the first "
        f"SSE chunk. Offending generators: {offenders}"
    )


def test_tracker_enter_exists_in_sync_body_of_chat_completions():
    top = None
    for n in ast.walk(_TREE):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "openai_chat_completions":
            top = n
            break
    assert top is not None, "openai_chat_completions handler missing"
    count = 0
    for sub in ast.walk(top):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "__enter__"
            and isinstance(fn.value, ast.Name)
            and fn.value.id.startswith("_tracker")
        ):
            count += 1
    assert count >= 3, (
        f"expected >=3 _tracker.__enter__() calls in openai_chat_completions "
        f"(one per streaming path), got {count}"
    )


def test_async_generators_cleanup_tracker_in_finally():
    required = {
        "gguf_tool_stream",
        "gguf_stream_chunks",
        "stream_chunks",
        "audio_input_stream",
    }
    found: set[str] = set()
    for fn in [n for n in ast.walk(_TREE) if isinstance(n, ast.AsyncFunctionDef)]:
        if fn.name not in required:
            continue
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Try) and sub.finalbody:
                if _finalbody_has_tracker_exit(sub.finalbody):
                    found.add(fn.name)
                    break
    missing = required - found
    assert not missing, (
        f"Cleanup must run via `finally: _tracker.__exit__(None, None, None)` "
        f"inside each streaming generator so ClientDisconnect / OSError paths "
        f"also release registry entries (Starlette skips `background` callbacks "
        f"when stream_response raises). Missing in: {sorted(missing)}"
    )


def test_chat_completions_streams_avoid_starlette_task_group():
    top = _async_function("openai_chat_completions")
    legacy_calls = []
    same_task_calls = 0
    for sub in ast.walk(top):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        if sub.func.id == "StreamingResponse":
            legacy_calls.append(sub.lineno)
        if sub.func.id == "_SameTaskStreamingResponse":
            same_task_calls += 1
    assert not legacy_calls, (
        "Streaming /v1/chat/completions must use _SameTaskStreamingResponse, "
        "not Starlette's legacy task-group StreamingResponse. Lines: "
        f"{legacy_calls}"
    )
    assert same_task_calls >= 5


def test_openai_passthrough_stream_avoids_starlette_task_group():
    functions = [
        _async_function("_openai_passthrough_stream"),
        _async_function("_openai_passthrough_stream_admitted"),
    ]
    legacy_calls = []
    same_task_calls = 0
    for fn in functions:
        for sub in ast.walk(fn):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
                continue
            if sub.func.id == "StreamingResponse":
                legacy_calls.append(sub.lineno)
            if sub.func.id == "_SameTaskStreamingResponse":
                same_task_calls += 1
    assert not legacy_calls, (
        "OpenAI passthrough streams must use _SameTaskStreamingResponse, "
        "not Starlette's legacy task-group StreamingResponse. Lines: "
        f"{legacy_calls}"
    )
    assert same_task_calls >= 2


def test_local_chat_streams_install_same_task_disconnect_watcher():
    top = _async_function("openai_chat_completions")
    assert _calls_name(top, "_await_disconnect_then_cancel"), (
        "Local same-task streams must watch request disconnects themselves; "
        "do not restore Starlette's task-group StreamingResponse for this."
    )


def test_direct_llama_server_streams_install_disconnect_watcher():
    required = {
        "openai_completions",
        "_responses_stream",
        "_anthropic_passthrough_stream",
        "_openai_passthrough_stream_admitted",
    }
    missing = [
        name
        for name in sorted(required)
        if not _calls_name(_async_function(name), "_await_disconnect_then_close")
    ]
    assert not missing, (
        "Direct httpx streams to llama-server must close the upstream response "
        "when the downstream client disconnects during prefill. Missing in: "
        f"{missing}"
    )


def test_audio_input_stream_installs_disconnect_watcher():
    audio = _async_function("audio_input_stream")
    has_watcher = False
    has_cleanup = False
    for sub in ast.walk(audio):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "create_task"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "asyncio"
                and sub.args
                and isinstance(sub.args[0], ast.Call)
                and isinstance(sub.args[0].func, ast.Name)
                and sub.args[0].func.id == "_await_disconnect_then_cancel"
            ):
                has_watcher = True
            if (
                isinstance(fn, ast.Name)
                and fn.id == "_stop_local_disconnect_cancel_watcher"
            ):
                has_cleanup = True
    assert has_watcher, (
        "audio_input_stream must install a disconnect watcher so client "
        "disconnects set cancel_event while asyncio.to_thread(next, ...) is blocked"
    )
    assert has_cleanup, "audio_input_stream must stop its disconnect watcher in finally"


# ── Behavioral helpers ───────────────────────────────────────

_WANTED = {
    "_CANCEL_REGISTRY",
    "_CANCEL_LOCK",
    "_PENDING_CANCELS",
    "_PENDING_CANCEL_TTL_S",
    "_prune_pending",
    "_TrackedCancel",
    "_cancel_by_keys",
    "_cancel_by_cancel_id_or_stash",
}


def _load_registry_module():
    chunks = []
    for n in _TREE.body:
        seg = ast.get_source_segment(SRC, n)
        if seg is None:
            continue
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _WANTED:
            chunks.append(seg)
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if any(name in _WANTED for name in names):
                chunks.append(seg)
        elif (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id in _WANTED
        ):
            chunks.append(seg)
    mod = {}
    exec("import threading, time\n" + "\n\n".join(chunks), mod)
    return mod


def _load_same_task_response_module():
    for n in _TREE.body:
        if isinstance(n, ast.ClassDef) and n.name == "_SameTaskStreamingResponse":
            source = ast.get_source_segment(SRC, n)
            break
    else:
        raise AssertionError("_SameTaskStreamingResponse missing")
    mod = {}
    exec(
        "class StreamingResponse: pass\nclass ClientDisconnect(Exception): pass\n"
        + source,
        mod,
    )
    return mod


def _make_stream(tracker, raise_exc):
    async def gen():
        try:
            try:
                yield "data: first\n\n"
                if raise_exc is not None:
                    raise raise_exc
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise
            except Exception:
                yield "data: error\n\n"
            finally:
                tracker.__exit__(None, None, None)
        except BaseException:
            raise

    return gen()


async def _consume(agen):
    out = []
    try:
        async for ch in agen:
            out.append(ch)
    except BaseException as e:
        out.append(type(e).__name__)
    return out


def _llama_stub_raises_on_preset_cancel(cancel_event):
    # Reproduces llama_cpp.py _stream_with_retry `raise GeneratorExit` when
    # cancel_event is already set at entry.
    if cancel_event.is_set():
        raise GeneratorExit
    yield "cumulative-1"
    yield "cumulative-2"


async def _post_fix_gguf_loop(cancel_event):
    yield "first_chunk"
    gen = _llama_stub_raises_on_preset_cancel(cancel_event)
    sentinel = object()
    while True:
        if cancel_event.is_set():
            break
        cumulative = await asyncio.to_thread(next, gen, sentinel)
        if cumulative is sentinel:
            break
        yield cumulative
    yield "final_chunk"
    yield "[DONE]"


# ── Behavioral tests ─────────────────────────────────────────


def test_finally_cleanup_on_normal_completion():
    m = _load_registry_module()
    m["_CANCEL_REGISTRY"].clear()
    ev = threading.Event()
    tr = m["_TrackedCancel"](ev, "cid-ok", "sid-ok")
    tr.__enter__()
    assert "cid-ok" in m["_CANCEL_REGISTRY"]
    chunks = asyncio.run(_consume(_make_stream(tr, None)))
    assert chunks == ["data: first\n\n", "data: [DONE]\n\n"]
    assert "cid-ok" not in m["_CANCEL_REGISTRY"]
    assert "sid-ok" not in m["_CANCEL_REGISTRY"]


def test_finally_cleanup_on_mid_stream_exception():
    # OSError mid-stream: the exact case where pre-fix `background=BackgroundTask(...)`
    # was skipped and leaked the registry entry.
    m = _load_registry_module()
    m["_CANCEL_REGISTRY"].clear()
    ev = threading.Event()
    tr = m["_TrackedCancel"](ev, "cid-err", "sid-err")
    tr.__enter__()
    assert "cid-err" in m["_CANCEL_REGISTRY"]
    asyncio.run(_consume(_make_stream(tr, OSError("disconnect"))))
    assert "cid-err" not in m["_CANCEL_REGISTRY"]
    assert "sid-err" not in m["_CANCEL_REGISTRY"]


def test_finally_cleanup_on_aclose():
    # Starlette calls aclose() on client disconnect; the finally block must run.
    m = _load_registry_module()
    m["_CANCEL_REGISTRY"].clear()
    ev = threading.Event()
    tr = m["_TrackedCancel"](ev, "cid-abort", "sid-abort")
    tr.__enter__()
    assert "cid-abort" in m["_CANCEL_REGISTRY"]

    async def run():
        gen = _make_stream(tr, None)
        it = gen.__aiter__()
        await it.__anext__()
        await gen.aclose()

    asyncio.run(run())
    assert "cid-abort" not in m["_CANCEL_REGISTRY"]
    assert "sid-abort" not in m["_CANCEL_REGISTRY"]


def test_same_task_response_closes_body_iterator_on_send_disconnect():
    m = _load_same_task_response_module()
    closed = False

    async def body():
        nonlocal closed
        try:
            yield "data: first\n\n"
        finally:
            closed = True

    async def run():
        agen = body()
        await agen.__anext__()
        response = m["_SameTaskStreamingResponse"].__new__(
            m["_SameTaskStreamingResponse"]
        )
        response.body_iterator = agen
        response.background = None
        # __new__ bypasses __init__; __call__'s disconnect branch reads _unstarted_cleanup.
        response._unstarted_cleanup = None

        async def stream_response(_send):
            raise OSError("client disconnected")

        response.stream_response = stream_response
        try:
            await response({}, None, lambda _message: None)
        except m["ClientDisconnect"]:
            pass
        else:
            raise AssertionError("expected ClientDisconnect")

    asyncio.run(run())
    assert closed


def test_preset_cancel_event_exits_cleanly_with_done():
    # Pending-replay: a stashed cancel pre-set cancel_event. The loop must break
    # cleanly with final_chunk + [DONE], not propagate GeneratorExit from the GGUF wrapper.
    ev = threading.Event()
    ev.set()
    chunks = asyncio.run(_consume(_post_fix_gguf_loop(ev)))
    assert "first_chunk" in chunks
    assert "final_chunk" in chunks
    assert "[DONE]" in chunks
    assert "GeneratorExit" not in chunks
    assert "cumulative-1" not in chunks
    assert "cumulative-2" not in chunks


def test_normal_path_streams_all_tokens():
    # Regression: the top-of-loop cancel_event check must not short-circuit when unset.
    ev = threading.Event()
    chunks = asyncio.run(_consume(_post_fix_gguf_loop(ev)))
    assert chunks == [
        "first_chunk",
        "cumulative-1",
        "cumulative-2",
        "final_chunk",
        "[DONE]",
    ]


def test_cancel_during_streaming_stops_iteration_promptly():
    # Setting cancel_event between yields breaks on the next iteration, not draining the generator.
    ev = threading.Event()

    async def _run():
        gen = _post_fix_gguf_loop(ev)
        seen = []
        async for ch in gen:
            seen.append(ch)
            if ch == "cumulative-1":
                ev.set()
        return seen

    seen = asyncio.run(_run())
    assert "first_chunk" in seen
    assert "cumulative-1" in seen
    assert "cumulative-2" not in seen
    assert "final_chunk" in seen
    assert "[DONE]" in seen


# ── Cancel-event responsiveness in the streaming loops ───────


def _loop_has_cancel_event_check(fn) -> bool:
    # An `if cancel_event.is_set():` inside a loop body is sufficient -- without it
    # a cancel POST can't interrupt, since Colab-style proxies drop request.is_disconnected().
    for sub in ast.walk(fn):
        if not isinstance(sub, (ast.While, ast.For, ast.AsyncFor)):
            continue
        for stmt in ast.walk(sub):
            if not isinstance(stmt, ast.If):
                continue
            t = stmt.test
            if (
                isinstance(t, ast.Call)
                and isinstance(t.func, ast.Attribute)
                and t.func.attr == "is_set"
                and isinstance(t.func.value, ast.Name)
                and t.func.value.id == "cancel_event"
            ):
                return True
    return False


def test_streaming_generators_check_cancel_event_in_loop():
    required = {
        "gguf_tool_stream",
        "gguf_stream_chunks",
        "stream_chunks",
        "audio_input_stream",
    }
    missing = []
    for fn in [n for n in ast.walk(_TREE) if isinstance(n, ast.AsyncFunctionDef)]:
        if fn.name not in required:
            continue
        if not _loop_has_cancel_event_check(fn):
            missing.append(fn.name)
    assert not missing, (
        f"Each streaming generator must check `cancel_event.is_set()` inside "
        f"its main loop so `POST /api/inference/cancel` can interrupt the "
        f"stream through proxies that do not forward fetch aborts. "
        f"Missing in: {sorted(missing)}"
    )


def test_audio_input_stream_offloads_blocking_next_to_thread():
    # Guards against regressing to `for chunk_text in audio_input_generate():`, which
    # blocks the event loop per whisper chunk and stalls POST /api/inference/cancel.
    audio = None
    for fn in ast.walk(_TREE):
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "audio_input_stream":
            audio = fn
            break
    assert audio is not None, "audio_input_stream generator missing"

    for sub in ast.walk(audio):
        if isinstance(sub, (ast.For, ast.AsyncFor)):
            it_src = ast.unparse(sub.iter)
            assert "audio_input_generate" not in it_src, (
                "audio_input_stream must not iterate audio_input_generate() "
                "directly -- that blocks the event loop. Use "
                "`await asyncio.to_thread(next, gen, _DONE)` inside a "
                "`while True` loop instead"
            )

    found_to_thread_next = False
    for sub in ast.walk(audio):
        if not isinstance(sub, ast.Call):
            continue
        fn_expr = sub.func
        if not (
            isinstance(fn_expr, ast.Attribute)
            and fn_expr.attr == "to_thread"
            and isinstance(fn_expr.value, ast.Name)
            and fn_expr.value.id == "asyncio"
        ):
            continue
        if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == "next":
            found_to_thread_next = True
            break
    assert found_to_thread_next, (
        "audio_input_stream must call `asyncio.to_thread(next, gen, ...)` "
        "to keep the event loop free while whisper yields the next chunk"
    )


def test_generate_stream_offloads_blocking_next_to_thread():
    outer = None
    for fn in ast.walk(_TREE):
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "generate_stream":
            outer = fn
            break
    assert outer is not None, "generate_stream handler missing"

    inner = None
    for sub in ast.walk(outer):
        if isinstance(sub, ast.AsyncFunctionDef) and sub.name == "stream":
            inner = sub
            break
    assert inner is not None, "generate_stream inner stream() generator missing"

    for sub in ast.walk(inner):
        if isinstance(sub, (ast.For, ast.AsyncFor)):
            it_src = ast.unparse(sub.iter)
            assert "generate_chat_response" not in it_src, (
                "generate_stream's inner stream() must not iterate "
                "backend.generate_chat_response() directly -- that blocks the event "
                "loop on every blocking subprocess read between tokens. Use "
                "`await asyncio.to_thread(next, gen, _DONE)` inside a `while True` "
                "loop instead"
            )

    found_to_thread_next = False
    for sub in ast.walk(inner):
        if not isinstance(sub, ast.Call):
            continue
        fn_expr = sub.func
        if not (
            isinstance(fn_expr, ast.Attribute)
            and fn_expr.attr == "to_thread"
            and isinstance(fn_expr.value, ast.Name)
            and fn_expr.value.id == "asyncio"
        ):
            continue
        if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == "next":
            found_to_thread_next = True
            break
    assert found_to_thread_next, (
        "generate_stream's inner stream() must call "
        "`asyncio.to_thread(next, gen, _DONE)` to keep the event loop free while the "
        "worker subprocess produces the next token"
    )


def test_generate_stream_cancels_backend_on_stream_cancelled_error():
    outer = None
    for fn in ast.walk(_TREE):
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "generate_stream":
            outer = fn
            break
    assert outer is not None, "generate_stream handler missing"

    outer_src = ast.unparse(outer)
    assert "cancel_event = threading.Event()" in outer_src

    inner = None
    for sub in ast.walk(outer):
        if isinstance(sub, ast.AsyncFunctionDef) and sub.name == "stream":
            inner = sub
            break
    assert inner is not None, "generate_stream inner stream() generator missing"

    def _awaits_to_thread_gen_close(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Await):
                continue
            call = sub.value
            if not isinstance(call, ast.Call):
                continue
            fn_expr = call.func
            if not (
                isinstance(fn_expr, ast.Attribute)
                and fn_expr.attr == "to_thread"
                and isinstance(fn_expr.value, ast.Name)
                and fn_expr.value.id == "asyncio"
            ):
                continue
            if not call.args:
                continue
            close_expr = call.args[0]
            if (
                isinstance(close_expr, ast.Attribute)
                and close_expr.attr == "close"
                and isinstance(close_expr.value, ast.Name)
                and close_expr.value.id == "gen"
            ):
                return True
        return False

    found_cancel_kwarg = False
    found_cancel_handler = False
    found_finally_cleanup = False
    for sub in ast.walk(inner):
        if isinstance(sub, ast.Call):
            call_src = ast.unparse(sub.func)
            if call_src.endswith("generate_chat_response"):
                found_cancel_kwarg = any(
                    kw.arg == "cancel_event"
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id == "cancel_event"
                    for kw in sub.keywords
                )
        if isinstance(sub, ast.ExceptHandler):
            exc_src = ast.unparse(sub.type) if sub.type is not None else ""
            if exc_src != "asyncio.CancelledError":
                continue
            body_src = "\n".join(ast.unparse(stmt) for stmt in sub.body)
            found_cancel_handler = (
                "cancel_event.set()" in body_src
                and "backend.reset_generation_state()" in body_src
                and any(
                    isinstance(stmt, ast.Raise) and stmt.exc is None
                    for stmt in sub.body
                )
            )
        if isinstance(sub, ast.Try) and sub.finalbody:
            final_src = "\n".join(ast.unparse(stmt) for stmt in sub.finalbody)
            found_finally_cleanup = (
                "not completed" in final_src
                and "not cancel_event.is_set()" in final_src
                and "cancel_event.set()" in final_src
                and "backend.reset_generation_state()" in final_src
                and _awaits_to_thread_gen_close(sub)
            )

    assert found_cancel_kwarg, (
        "generate_stream must pass cancel_event into backend.generate_chat_response "
        "so cancelled streams can stop backend generation"
    )
    assert found_cancel_handler, (
        "generate_stream must catch asyncio.CancelledError, set cancel_event, "
        "reset backend state, and re-raise"
    )
    assert found_finally_cleanup, (
        "generate_stream cleanup must cancel/reset incomplete streams and "
        "offload gen.close() with asyncio.to_thread so backend joins cannot "
        "block the event loop"
    )


def test_stream_chunks_cancel_branch_resets_backend_state():
    # The cancel branch must call backend.reset_generation_state() to flush
    # GPU/KV-cache state, else cancel-via-POST leaves the subprocess dirty.
    fn = None
    top = None
    for n in ast.walk(_TREE):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "openai_chat_completions":
            top = n
            break
    assert top is not None
    for n in ast.walk(top):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "stream_chunks":
            fn = n
            break
    assert fn is not None, "stream_chunks generator missing"

    for sub in ast.walk(fn):
        if not isinstance(sub, ast.If):
            continue
        t = sub.test
        if not (
            isinstance(t, ast.Call)
            and isinstance(t.func, ast.Attribute)
            and t.func.attr == "is_set"
            and isinstance(t.func.value, ast.Name)
            and t.func.value.id == "cancel_event"
        ):
            continue
        body_src = "\n".join(ast.unparse(s) for s in sub.body)
        if "backend.reset_generation_state()" in body_src:
            return
    raise AssertionError(
        "stream_chunks `if cancel_event.is_set():` branch must call "
        "backend.reset_generation_state() -- matches the existing "
        "request.is_disconnected() / CancelledError cleanup paths and "
        "prevents KV-cache drift after cancel-via-POST"
    )


# ── Behavioral simulations for the iter-1 fixes ──────────────


def test_unsloth_stream_loop_breaks_on_external_cancel_event():
    cancel_event = threading.Event()
    reset_calls = [0]

    class _Backend:
        def reset_generation_state(self):
            reset_calls[0] += 1

    backend = _Backend()

    def _generate():
        for i in range(200):
            time.sleep(0.005)
            yield f"cum-{i}"

    async def _loop():
        _DONE = object()
        loop = asyncio.get_event_loop()
        gen = _generate()
        seen = []
        while True:
            if cancel_event.is_set():
                backend.reset_generation_state()
                break
            cumulative = await loop.run_in_executor(None, next, gen, _DONE)
            if cumulative is _DONE:
                break
            seen.append(cumulative)
        return seen

    async def _fire():
        await asyncio.sleep(0.05)
        cancel_event.set()

    async def _main():
        return await asyncio.gather(_loop(), _fire())

    seen, _ = asyncio.run(_main())
    assert (
        len(seen) < 200
    ), f"loop must not drain the generator after cancel; got {len(seen)} tokens"
    assert reset_calls[0] == 1, (
        f"backend.reset_generation_state() must be called exactly once on "
        f"cancel-via-POST, got {reset_calls[0]}"
    )


def test_generate_stream_stays_responsive_under_blocking_next():
    # Same sync-generator shape as generate_stream, with resp_queue.get modeled
    # by sleep. The output must stay unchanged while next() moves off-loop.
    chunks = ["alpha", "beta", "gamma", "delta"]

    def _generate_chat_response():
        for chunk in chunks:
            time.sleep(0.08)
            yield chunk

    def _sse(chunk):
        return f"data: {json.dumps({'content': chunk})}\n\n"

    async def _direct_loop():
        out = []
        for chunk in _generate_chat_response():
            out.append(_sse(chunk))
        out.append("data: [DONE]\n\n")
        return out

    async def _to_thread_loop():
        _DONE = object()
        gen = _generate_chat_response()
        out = []
        try:
            while True:
                chunk = await asyncio.to_thread(next, gen, _DONE)
                if chunk is _DONE:
                    break
                out.append(_sse(chunk))
            out.append("data: [DONE]\n\n")
            return out
        finally:
            try:
                gen.close()
            except (RuntimeError, ValueError):
                pass

    async def _run_with_heartbeat(loop_coro):
        ticks = 0
        max_gap = 0.0

        async def _heartbeat():
            nonlocal ticks, max_gap
            last = time.monotonic()
            while True:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                max_gap = max(max_gap, now - last)
                last = now
                ticks += 1

        heartbeat = asyncio.create_task(_heartbeat())
        await asyncio.sleep(0)
        try:
            out = await loop_coro()
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        return out, ticks, max_gap

    async def _main():
        direct_out, direct_ticks, direct_max_gap = await _run_with_heartbeat(
            _direct_loop
        )
        threaded_out, threaded_ticks, threaded_max_gap = await _run_with_heartbeat(
            _to_thread_loop
        )
        return (
            direct_out,
            direct_ticks,
            direct_max_gap,
            threaded_out,
            threaded_ticks,
            threaded_max_gap,
        )

    (
        direct_out,
        direct_ticks,
        direct_max_gap,
        threaded_out,
        threaded_ticks,
        threaded_max_gap,
    ) = asyncio.run(_main())

    assert (
        threaded_out
        == direct_out
        == [_sse(chunk) for chunk in chunks] + ["data: [DONE]\n\n"]
    )
    assert direct_ticks == 0, (
        f"direct generate_stream loop should block the event loop; "
        f"got {direct_ticks} heartbeat ticks and max gap {direct_max_gap:.3f}s"
    )
    assert threaded_ticks >= 8, (
        f"to_thread generate_stream loop should let the event loop run; "
        f"got {threaded_ticks} heartbeat ticks"
    )
    assert threaded_max_gap < 0.06, (
        f"to_thread generate_stream loop should avoid long heartbeat gaps; "
        f"got {threaded_max_gap:.3f}s"
    )


def test_audio_stream_stays_responsive_under_blocking_next():
    # Assert the pre-fix `for chunk in audio_input_generate()` pattern blocks the
    # event loop, then confirm the post-fix pattern exits promptly.
    cancel_event = threading.Event()

    def _audio_gen():
        for i in range(8):
            time.sleep(0.15)
            yield f"chunk-{i}"

    async def _prefix_loop():
        seen = []
        for chunk_text in _audio_gen():
            if cancel_event.is_set():
                break
            seen.append(chunk_text)
        return seen

    async def _postfix_loop():
        _DONE = object()
        gen = _audio_gen()
        seen = []
        while True:
            if cancel_event.is_set():
                break
            chunk_text = await asyncio.to_thread(next, gen, _DONE)
            if chunk_text is _DONE:
                break
            seen.append(chunk_text)
        return seen

    async def _fire_early():
        await asyncio.sleep(0.05)
        cancel_event.set()

    async def _run(loop_coro):
        return await asyncio.gather(loop_coro, _fire_early())

    cancel_event.clear()
    t0 = time.monotonic()
    prefix_seen, _ = asyncio.run(_run(_prefix_loop()))
    prefix_elapsed = time.monotonic() - t0
    assert prefix_elapsed >= 0.13, (
        f"pre-fix pattern should block event loop for >=1 chunk time "
        f"(~150ms); got {prefix_elapsed:.3f}s, {len(prefix_seen)} chunks"
    )

    cancel_event.clear()
    t0 = time.monotonic()
    postfix_seen, _ = asyncio.run(_run(_postfix_loop()))
    postfix_elapsed = time.monotonic() - t0
    assert postfix_elapsed < prefix_elapsed, (
        f"post-fix pattern must exit faster than pre-fix (blocking) "
        f"pattern; post={postfix_elapsed:.3f}s vs pre={prefix_elapsed:.3f}s"
    )
    assert (
        len(postfix_seen) < 8
    ), f"post-fix loop must not drain all chunks; got {len(postfix_seen)}"


def test_unsloth_stream_loop_emits_zero_tokens_on_preset_cancel():
    # Pending-cancel replay: cancel_event pre-set, so the top-of-loop check must
    # short-circuit iteration 1 (zero tokens). Catches moving the check below next().
    cancel_event = threading.Event()
    cancel_event.set()
    reset_calls = [0]

    class _Backend:
        def reset_generation_state(self):
            reset_calls[0] += 1

    backend = _Backend()

    next_calls = [0]

    def _generate():
        while True:
            next_calls[0] += 1
            yield f"cum-{next_calls[0]}"

    async def _loop():
        _DONE = object()
        loop = asyncio.get_event_loop()
        gen = _generate()
        seen = []
        while True:
            if cancel_event.is_set():
                backend.reset_generation_state()
                break
            cumulative = await loop.run_in_executor(None, next, gen, _DONE)
            if cumulative is _DONE:
                break
            seen.append(cumulative)
        return seen

    seen = asyncio.run(_loop())
    assert seen == [], (
        f"loop must emit zero tokens when cancel_event is pre-set "
        f"(pending-replay path); got {seen}"
    )
    assert next_calls[0] == 0, (
        f"loop must not call next() at all on pre-set cancel; got "
        f"{next_calls[0]} calls"
    )
    assert reset_calls[0] == 1, (
        f"backend.reset_generation_state() must still fire exactly once "
        f"on pre-set cancel; got {reset_calls[0]}"
    )


def test_audio_stream_emits_zero_chunks_on_preset_cancel():
    # Symmetric to the Unsloth pre-set test: the audio loop must skip
    # asyncio.to_thread(next, ...) when cancel_event was pre-set via pending-replay.
    cancel_event = threading.Event()
    cancel_event.set()

    next_calls = [0]

    def _audio_gen():
        while True:
            next_calls[0] += 1
            yield f"chunk-{next_calls[0]}"

    async def _loop():
        _DONE = object()
        gen = _audio_gen()
        seen = []
        while True:
            if cancel_event.is_set():
                break
            chunk_text = await asyncio.to_thread(next, gen, _DONE)
            if chunk_text is _DONE:
                break
            seen.append(chunk_text)
        return seen

    seen = asyncio.run(_loop())
    assert seen == [], f"audio loop must emit zero chunks on pre-set cancel; got {seen}"
    assert next_calls[0] == 0, (
        f"audio loop must not call next() on pre-set cancel; got "
        f"{next_calls[0]} calls"
    )
