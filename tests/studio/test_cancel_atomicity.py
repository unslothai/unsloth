"""
Regression guards for the TOCTOU race that existed between the cancel
handler and _TrackedCancel.__enter__ before the atomic refactor.

The original mechanism split registry-lookup and pending-stash across
TWO lock acquisitions in cancel_inference, and registry-insertion and
consume-pending across TWO acquisitions in __enter__. Under contention
an interleaving like this drops the cancel silently:

  [cancel thread]    acquire lock
                     bucket empty, release
                     (no stash yet)
  [handler thread]   acquire lock, register, release
  [handler thread]   acquire lock, consume-pending (empty), release
  [cancel thread]    acquire lock, stash (TOO LATE), release
  [handler thread]   stream runs without ever seeing the cancel

The fix folds each side into a single _CANCEL_LOCK critical section:
  - cancel_inference calls _cancel_by_cancel_id_or_stash() which does
    lookup + stash atomically.
  - _TrackedCancel.__enter__ does register + consume-pending atomically.

This file guards both invariants structurally (AST) and behaviorally
(parallel stress) so a future refactor cannot reintroduce the race
silently.
"""

from __future__ import annotations

import ast
import random
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


def _find_function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _find_class(name: str) -> ast.ClassDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _count_with_cancel_lock_blocks(node: ast.AST) -> int:
    n = 0
    for sub in ast.walk(node):
        if not isinstance(sub, ast.With):
            continue
        for item in sub.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Name) and ctx.id == "_CANCEL_LOCK":
                n += 1
                break
    return n


def test_cancel_by_cancel_id_or_stash_is_single_lock_critical_section():
    # The helper MUST acquire _CANCEL_LOCK exactly once and perform both
    # the bucket lookup and the pending-stash inside that block. Two
    # acquisitions reintroduce the TOCTOU race.
    fn = _find_function("_cancel_by_cancel_id_or_stash")
    assert _count_with_cancel_lock_blocks(fn) == 1, (
        "_cancel_by_cancel_id_or_stash must use exactly one `with "
        "_CANCEL_LOCK:` block; splitting into two acquisitions reopens "
        "the TOCTOU race with _TrackedCancel.__enter__"
    )
    src = ast.unparse(fn)
    assert "_CANCEL_REGISTRY.get(cancel_id)" in src, (
        "_cancel_by_cancel_id_or_stash must look up the registry bucket "
        "for the supplied cancel_id under the lock"
    )
    assert "_PENDING_CANCELS[cancel_id]" in src, (
        "_cancel_by_cancel_id_or_stash must stash into _PENDING_CANCELS "
        "when the registry miss path is taken"
    )


def test_tracked_cancel_enter_registers_and_consumes_pending_under_one_lock():
    cls = _find_class("_TrackedCancel")
    enter = None
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__enter__":
            enter = n
            break
    assert enter is not None, "_TrackedCancel.__enter__ missing"
    # Exactly one `with _CANCEL_LOCK:` block inside __enter__.
    assert _count_with_cancel_lock_blocks(enter) == 1, (
        "_TrackedCancel.__enter__ must acquire _CANCEL_LOCK exactly once. "
        "A second acquisition for the pending-consume step lets a "
        "concurrent cancel POST stash after consume sees an empty map, "
        "silently dropping the cancel"
    )
    # The single critical section must both insert into the registry
    # AND pop from _PENDING_CANCELS.
    with_block = None
    for sub in ast.walk(enter):
        if isinstance(sub, ast.With) and any(
            isinstance(i.context_expr, ast.Name) and i.context_expr.id == "_CANCEL_LOCK"
            for i in sub.items
        ):
            with_block = sub
            break
    assert with_block is not None
    block_src = "\n".join(ast.unparse(s) for s in with_block.body)
    assert "_CANCEL_REGISTRY.setdefault" in block_src, (
        "__enter__ critical section must insert into _CANCEL_REGISTRY"
    )
    assert "_PENDING_CANCELS.pop" in block_src, (
        "__enter__ critical section must consume from _PENDING_CANCELS "
        "(pop) inside the same lock, not in a later re-acquisition"
    )


def test_cancel_inference_uses_atomic_helper_for_cancel_id_path():
    fn = _find_function("cancel_inference")
    src = ast.unparse(fn)
    # The cancel_id branch should route through the atomic helper.
    assert "_cancel_by_cancel_id_or_stash" in src, (
        "cancel_inference must route the cancel_id branch through "
        "_cancel_by_cancel_id_or_stash so lookup + stash are atomic"
    )
    # The pre-fix idiom must be gone.
    assert "_remember_pending_cancel(cancel_id)" not in src, (
        "cancel_inference must not call _remember_pending_cancel after a "
        "separate _cancel_by_keys([cancel_id]) lookup; that is the "
        "two-step pattern that produced the TOCTOU race"
    )


# ── Runtime parallel stress ─────────────────────────────────────────


_WANTED = {
    "_CANCEL_REGISTRY",
    "_CANCEL_LOCK",
    "_PENDING_CANCELS",
    "_PENDING_CANCEL_TTL_S",
    "_prune_pending",
    "_remember_pending_cancel",
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
    exec(
        "import threading, time\nfrom typing import Optional\n" + "\n\n".join(chunks),
        mod,
    )
    return mod


def test_parallel_cancel_vs_register_never_drops():
    # Race cancel_by_cancel_id_or_stash against _TrackedCancel.__enter__
    # in separate threads with randomized start order. A dropped event
    # means the TOCTOU race is reintroduced.
    m = _load_registry_module()
    trials = 500
    dropped = 0
    for i in range(trials):
        m["_CANCEL_REGISTRY"].clear()
        m["_PENDING_CANCELS"].clear()
        cid = f"cid-{i}"
        ev = threading.Event()
        tracker = m["_TrackedCancel"](ev, cid, "thread")
        start = threading.Event()

        def do_cancel():
            start.wait()
            m["_cancel_by_cancel_id_or_stash"](cid)

        def do_enter():
            start.wait()
            tracker.__enter__()

        t1 = threading.Thread(target=do_cancel)
        t2 = threading.Thread(target=do_enter)
        threads = [t1, t2]
        random.shuffle(threads)
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        if not ev.is_set():
            dropped += 1
        tracker.__exit__(None, None, None)

    assert dropped == 0, (
        f"TOCTOU regression: {dropped}/{trials} parallel trials silently "
        f"dropped the cancel -- atomic helper may have been split again"
    )


def test_cancel_before_register_replays_atomically():
    # Sequential scenario that MUST be handled by the atomic helper:
    # the cancel arrives first, registers a pending entry, and the
    # subsequent __enter__ replays it.
    m = _load_registry_module()
    cid = "early-cid"
    ev = threading.Event()
    tracker = m["_TrackedCancel"](ev, cid, "thread-x")

    n = m["_cancel_by_cancel_id_or_stash"](cid)
    assert n == 0, "cancel before registration should return 0 (nothing to signal)"
    assert cid in m["_PENDING_CANCELS"], "helper must stash the cancel_id"

    tracker.__enter__()
    assert ev.is_set(), "subsequent __enter__ must replay the pending cancel"
    assert cid not in m["_PENDING_CANCELS"], (
        "pending entry must be consumed by __enter__ (one-shot replay)"
    )
    tracker.__exit__(None, None, None)


def test_cancel_after_register_signals_without_stash():
    # Converse: cancel arriving after registration must signal the
    # registered event and must NOT leave a pending entry behind.
    m = _load_registry_module()
    cid = "post-cid"
    ev = threading.Event()
    tracker = m["_TrackedCancel"](ev, cid, "thread-y")
    tracker.__enter__()

    n = m["_cancel_by_cancel_id_or_stash"](cid)
    assert n == 1
    assert ev.is_set()
    assert cid not in m["_PENDING_CANCELS"], (
        "post-registration cancel must signal directly, not stash"
    )
    tracker.__exit__(None, None, None)
