"""
Tests that the cancel tracker is registered BEFORE StreamingResponse is
returned, and that cleanup runs via a `finally` block inside each
async generator.

The zombie-generation scenario is: user clicks Stop during prefill /
warmup / proxy buffering, before the first SSE chunk. If _tracker
__enter__ lives inside the async generator body, the registry is empty
at the moment /api/inference/cancel lands -- so cancel returns 0 and
the decode runs to completion.

The fix moves _tracker = _TrackedCancel(...) and _tracker.__enter__()
to the synchronous body of openai_chat_completions (before the
StreamingResponse is returned) and places _tracker.__exit__ inside
each generator's `finally` block. Using a generator `finally` (rather
than a Starlette BackgroundTask) guarantees cleanup on every
termination path -- normal exhaustion, CancelledError from
ClientDisconnect, and OSError / BrokenPipeError during send() --
because Starlette skips `background` callbacks when stream_response
raises.

Structural verifies:
  - No `async def ...:` body contains `_tracker.__enter__()` in
    routes/inference.py (registration moved to sync body).
  - Each of the four async generators (gguf_tool_stream,
    gguf_stream_chunks, stream_chunks, audio_input_stream) contains
    `_tracker.__exit__(None, None, None)` inside a try/finally block.
  - No StreamingResponse in openai_chat_completions passes
    `background=` (cleanup now lives in the generator finally).

Behavioral verifies (extracting `_TrackedCancel` from source and
exercising the actual runtime semantics):
  - `finally: _tracker.__exit__(...)` runs on normal completion,
    mid-stream exception (OSError / BrokenPipeError from send()),
    and aclose() from Starlette ClientDisconnect.
  - A pre-set cancel_event (from `_TrackedCancel.__enter__` replaying
    a pending cancel POST) lets the GGUF while-loop break cleanly
    and emit final_chunk + [DONE] instead of propagating
    `GeneratorExit` out of `_stream_with_retry` into the async
    generator's `except Exception` (which would not catch it).
"""

from __future__ import annotations

import ast
import asyncio
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
SRC = SOURCE_PATH.read_text()
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


def test_streaming_responses_have_no_background_task():
    top = None
    for n in ast.walk(_TREE):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "openai_chat_completions":
            top = n
            break
    assert top is not None
    for sub in ast.walk(top):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        if sub.func.id != "StreamingResponse":
            continue
        kwargs = {kw.arg for kw in sub.keywords if kw.arg}
        assert "background" not in kwargs, (
            "StreamingResponse in openai_chat_completions must not pass "
            "`background=` -- cleanup now lives in the generator's finally "
            "block; a BackgroundTask would be skipped on abrupt disconnect"
        )


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
    # Reproduces llama_cpp.py _stream_with_retry:2240 `raise GeneratorExit`
    # when cancel_event is already set at entry.
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
    # Simulates OSError / BrokenPipeError from Starlette send() mid-stream --
    # the exact case where pre-fix `background = BackgroundTask(...)` was
    # skipped and leaked the registry entry.
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
    # Starlette calls aclose() on the async generator when the client
    # disconnects mid-stream. The generator's finally block must run.
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


def test_preset_cancel_event_exits_cleanly_with_done():
    # Pending-replay: POST /cancel arrived before the stream registered,
    # was stashed, then consumed by _TrackedCancel.__enter__ which set
    # cancel_event. The generator must break out of the loop cleanly
    # and emit final_chunk + [DONE] rather than calling next(gen) and
    # propagating `GeneratorExit` out of the GGUF stream wrapper.
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
    # Regression: the top-of-loop cancel_event check must not short-circuit
    # when cancel_event is unset.
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
    # Setting cancel_event between yields breaks out on the next iteration
    # rather than draining the stub generator.
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
    # An `if cancel_event.is_set():` statement anywhere inside a
    # `while`/`for` loop body is sufficient -- without it, a cancel POST
    # cannot interrupt the loop because Colab-style proxies do not
    # propagate request.is_disconnected().
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
    # Guards against regression back to `for chunk_text in
    # audio_input_generate():` -- which blocks the event loop on each
    # whisper chunk and prevents POST /api/inference/cancel from being
    # serviced until the chunk yields.
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


def test_stream_chunks_cancel_branch_resets_backend_state():
    # The Unsloth path's cancel branch must flush GPU / KV-cache state
    # via `backend.reset_generation_state()` -- the orchestrator's
    # internal cancel path does not do this, so a cancel-via-POST that
    # only broke the loop would leave the subprocess in a dirty state
    # for the next request.
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


def test_audio_stream_stays_responsive_under_blocking_next():
    # Regression guard: replace the post-fix loop with the pre-fix
    # `for chunk in audio_input_generate()` pattern and assert it blocks
    # the event loop; then confirm the post-fix pattern exits promptly.
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
    # Pending-cancel replay: _TrackedCancel.__enter__ already set
    # cancel_event before the generator body starts iterating. The
    # top-of-loop check must short-circuit the very first iteration so
    # no token is emitted. Catches a regression that moves the check
    # below `next()` -- the mid-loop test would still pass but this
    # test would observe one extra token leak.
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
    # Symmetric to the Unsloth pre-set test: the audio loop's top-of-loop
    # cancel check must skip the asyncio.to_thread(next, ...) call when
    # cancel_event was already set via pending-replay.
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
