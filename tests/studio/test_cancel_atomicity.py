"""
TOCTOU atomicity guards for the cancel path.

Structural: cancel_inference, _cancel_by_cancel_id_or_stash, and
_TrackedCancel.__enter__ must each use a single _CANCEL_LOCK critical
section over lookup + stash / register + consume-pending.

Behavioral: parallel cancel-POST vs __enter__ must never drop a cancel.
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
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
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
    fn = _find_function("_cancel_by_cancel_id_or_stash")
    assert _count_with_cancel_lock_blocks(fn) == 1, (
        "_cancel_by_cancel_id_or_stash must use exactly one `with "
        "_CANCEL_LOCK:` block; splitting into two acquisitions reopens "
        "the TOCTOU race with _TrackedCancel.__enter__"
    )
    src = ast.unparse(fn)
    assert "_CANCEL_REGISTRY.get(cancel_id)" in src
    assert "_PENDING_CANCELS[cancel_id]" in src


def test_tracked_cancel_enter_registers_and_consumes_pending_under_one_lock():
    cls = _find_class("_TrackedCancel")
    enter = None
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__enter__":
            enter = n
            break
    assert enter is not None
    assert _count_with_cancel_lock_blocks(enter) == 1, (
        "_TrackedCancel.__enter__ must acquire _CANCEL_LOCK exactly once. "
        "A second acquisition for consume-pending lets a concurrent "
        "cancel POST stash after consume sees an empty map, silently "
        "dropping the cancel"
    )
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
    assert "_CANCEL_REGISTRY.setdefault" in block_src
    assert "_PENDING_CANCELS.pop" in block_src, (
        "__enter__ critical section must consume from _PENDING_CANCELS "
        "inside the same lock, not a later re-acquisition"
    )


def test_cancel_inference_uses_atomic_helper_for_cancel_id_path():
    fn = _find_function("cancel_inference")
    src = ast.unparse(fn)
    assert "_cancel_by_cancel_id_or_stash" in src
    # The pre-fix two-step idiom must be gone.
    assert "_remember_pending_cancel(cancel_id)" not in src, (
        "two-step _cancel_by_keys + _remember_pending_cancel produced "
        "the TOCTOU race and must not return"
    )


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

        threads = [
            threading.Thread(target = do_cancel),
            threading.Thread(target = do_enter),
        ]
        random.shuffle(threads)
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout = 5.0)
            assert not t.is_alive()

        if not ev.is_set():
            dropped += 1
        tracker.__exit__(None, None, None)

    assert dropped == 0, (
        f"TOCTOU regression: {dropped}/{trials} parallel trials silently "
        f"dropped the cancel"
    )


def test_cancel_before_register_replays_atomically():
    m = _load_registry_module()
    cid = "early-cid"
    ev = threading.Event()
    tracker = m["_TrackedCancel"](ev, cid, "thread-x")

    assert m["_cancel_by_cancel_id_or_stash"](cid) == 0
    assert cid in m["_PENDING_CANCELS"]

    tracker.__enter__()
    assert ev.is_set()
    assert cid not in m["_PENDING_CANCELS"]
    tracker.__exit__(None, None, None)


def test_cancel_after_register_signals_without_stash():
    m = _load_registry_module()
    cid = "post-cid"
    ev = threading.Event()
    tracker = m["_TrackedCancel"](ev, cid, "thread-y")
    tracker.__enter__()

    assert m["_cancel_by_cancel_id_or_stash"](cid) == 1
    assert ev.is_set()
    assert cid not in m["_PENDING_CANCELS"]
    tracker.__exit__(None, None, None)
