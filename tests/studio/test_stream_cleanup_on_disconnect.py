"""
Behavioral guards for the cancel-cleanup fixes.

Existing tests in this directory are AST-only (structural). These tests
exercise the actual runtime semantics that the fixes rely on:

  1. `finally: _tracker.__exit__(...)` inside each streaming generator
     runs on every termination path -- normal exhaustion, mid-stream
     exception (OSError / BrokenPipeError from send()), and aclose()
     from Starlette ClientDisconnect. This closes the registry-leak
     bug where the prior `background = BackgroundTask(...)` was
     skipped when stream_response raised.

  2. A pre-set cancel_event (from `_TrackedCancel.__enter__` replaying
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
_SRC = SOURCE_PATH.read_text()
_TREE = ast.parse(_SRC)


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
        seg = ast.get_source_segment(_SRC, n)
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
    """Mimic the post-fix generator pattern: try/yield/except/finally."""

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


def _llama_stub_raises_on_preset_cancel(cancel_event):
    """Reproduces llama_cpp.py _stream_with_retry:2240 `raise GeneratorExit`
    when cancel_event is already set at entry."""
    if cancel_event.is_set():
        raise GeneratorExit
    yield "cumulative-1"
    yield "cumulative-2"


async def _post_fix_gguf_loop(cancel_event):
    """Mimics the post-fix GGUF async generator with the top-of-loop cancel
    check (Fix C)."""
    yield "first_chunk"
    gen = _llama_stub_raises_on_preset_cancel(cancel_event)
    sentinel = object()
    while True:
        if cancel_event.is_set():  # Fix C
            break
        cumulative = await asyncio.to_thread(next, gen, sentinel)
        if cumulative is sentinel:
            break
        yield cumulative
    yield "final_chunk"
    yield "[DONE]"


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
    # The stub's "cumulative-*" tokens must NOT appear -- the pre-cancel
    # check must short-circuit before next(gen) is ever called.
    assert "cumulative-1" not in chunks
    assert "cumulative-2" not in chunks


def test_normal_path_streams_all_tokens():
    # Regression: the Fix C top-of-loop check must not short-circuit
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
    # Verify that setting cancel_event between yields breaks out of the
    # loop on the next iteration rather than draining the stub generator.
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
