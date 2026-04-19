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
