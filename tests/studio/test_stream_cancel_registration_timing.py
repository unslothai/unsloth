"""
Tests that the cancel tracker is registered BEFORE StreamingResponse is
returned, and that cleanup runs via a Starlette BackgroundTask.

The zombie-generation scenario is: user clicks Stop during prefill /
warmup / proxy buffering, before the first SSE chunk. If _tracker
__enter__ lives inside the async generator body, the registry is empty
at the moment /api/inference/cancel lands -- so cancel returns 0 and
the decode runs to completion.

The fix moves _tracker = _TrackedCancel(...) and _tracker.__enter__()
to the synchronous body of openai_chat_completions (before the
StreamingResponse is returned) and attaches _tracker.__exit__ as a
BackgroundTask so Starlette calls it when the response is drained or
aborted.

Verifies:
  - No `async def ...:` body contains `_tracker.__enter__()` in
    routes/inference.py (registration moved to sync body).
  - All three StreamingResponse calls inside openai_chat_completions
    pass a BackgroundTask bound to `_tracker.__exit__`.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
SRC = SOURCE_PATH.read_text()


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


def test_no_tracker_enter_inside_async_generators():
    tree = ast.parse(SRC)
    offenders = []
    for fn in _collect_async_functions(tree):
        if fn.name in {"gguf_tool_stream", "gguf_stream_chunks", "stream_chunks"}:
            if _has_tracker_enter_call(fn):
                offenders.append(fn.name)
    assert not offenders, (
        f"Cancel tracker registration must live OUTSIDE the async generator "
        f"body so a stop POST can find the registry entry before the first "
        f"SSE chunk. Offending generators: {offenders}"
    )


def test_tracker_enter_exists_in_sync_body_of_chat_completions():
    # The tracker.__enter__() call should occur inside the top-level
    # async def openai_chat_completions function, but NOT nested in its
    # inner async generators.
    tree = ast.parse(SRC)
    top = None
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "openai_chat_completions":
            top = n
            break
    assert top is not None, "openai_chat_completions handler missing"
    # Count _tracker.__enter__() at any nesting level inside
    # openai_chat_completions -- we expect 3 (one per streaming path).
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
        f"expected 3 _tracker.__enter__() calls in openai_chat_completions "
        f"(one per streaming path), got {count}"
    )


def test_streaming_responses_use_background_task_for_cleanup():
    tree = ast.parse(SRC)
    top = None
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "openai_chat_completions":
            top = n
            break
    assert top is not None

    bg_task_calls = 0
    for sub in ast.walk(top):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        if sub.func.id != "StreamingResponse":
            continue
        kwargs = {kw.arg: kw.value for kw in sub.keywords if kw.arg}
        bg = kwargs.get("background")
        if bg is None:
            continue
        if not (isinstance(bg, ast.Call) and isinstance(bg.func, ast.Name)):
            continue
        if bg.func.id != "BackgroundTask":
            continue
        if not bg.args:
            continue
        first = bg.args[0]
        if (
            isinstance(first, ast.Attribute)
            and first.attr == "__exit__"
            and isinstance(first.value, ast.Name)
            and first.value.id.startswith("_tracker")
        ):
            bg_task_calls += 1

    assert bg_task_calls >= 3, (
        f"expected every StreamingResponse in openai_chat_completions to "
        f"pass background=BackgroundTask(_tracker.__exit__, ...); found "
        f"{bg_task_calls}"
    )


def test_background_task_is_imported():
    tree = ast.parse(SRC)
    imported = False
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "starlette.background"
        ):
            for alias in node.names:
                if alias.name == "BackgroundTask":
                    imported = True
    assert imported, (
        "starlette.background.BackgroundTask must be imported for the "
        "stream-cleanup wiring"
    )
