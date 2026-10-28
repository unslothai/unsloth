"""save.py's repeated full gc.collect() passes stop once a pass that really ran finds nothing.

A pass that finds no unreachable objects frees nothing and runs no finalizer, so the next pass walks the
same heap and finds nothing either; each full pass costs ~0.1 to 0.5 s on an Unsloth heap. Every other
case keeps the old behaviour of running all passes: gc.callbacks registered, a non-zero or None result, or
a pass that was skipped because another thread was collecting (CPython returns 0 without collecting then).
CPU only: the helper is sliced out of save.py with ast, so nothing heavy is imported."""

from __future__ import annotations

import ast
import gc as real_gc
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
# unsloth_save_model 2, the four merged save / push entry points 3 each, GGUF 1, torchao 3 + 4 + 1,
# compressed tensors 2, OpenVINO 1.
CALL_SITES = 26


def _tree():
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"))


def _function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _is_collect(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and getattr(node.func.value, "id", None) == "gc"
    )


def _helper(gc):
    namespace = {"gc": gc}
    fn = _function(_tree(), "_collect_garbage")
    exec(compile(ast.Module([fn], []), str(SAVE_PY), "exec"), namespace)
    return namespace["_collect_garbage"]


class _FakeGC:
    """`script` items: an int or None is what a pass that runs returns; "skip" is a pass that returns 0
    without collecting because another thread is mid-collection (stats unchanged)."""

    def __init__(
        self,
        script,
        callbacks = (),
    ):
        self.script = iter(script)
        self.callbacks = list(callbacks)
        self.stats = {"collections": 0, "collected": 0, "uncollectable": 0}
        self.calls = 0

    def get_stats(self):
        return [{}, {}, dict(self.stats)]

    def collect(self):
        self.calls += 1
        result = next(self.script)
        if result == "skip":
            return 0
        self.stats["collections"] += 1
        self.stats["collected"] += result or 0
        return result


@pytest.mark.parametrize(
    "script,expected",
    [
        ([5, 0, 7], 2),
        ([0, 9, 9], 1),
        ([4, 2, 1], 3),
        ([None, None, None], 3),  # gc.collect replaced (vLLM graph capture): never "found nothing"
        (["skip", 0, 9], 2),  # a skipped pass is not proof the heap is quiet
        (["skip", "skip", "skip"], 3),
    ],
)
def test_stops_after_a_pass_that_ran_and_found_nothing(script, expected):
    gc = _FakeGC(script)
    order = []
    _helper(gc)(before = lambda: order.append("before"), after = lambda: order.append("after"))
    assert gc.calls == expected
    # The cache is still cleared around every collect that ran.
    assert order == ["before", "after"] * expected


def test_keeps_every_pass_with_gc_callbacks():
    gc = _FakeGC([0, 0, 0], callbacks = [print])
    _helper(gc)()
    assert gc.calls == 3


def test_a_pass_skipped_for_another_threads_collection_does_not_end_it():
    # Real gc: thread A is inside a collection, blocked in a finalizer, when the helper starts. The
    # helper's first gc.collect() returns 0 without collecting; stopping there would leave our garbage.
    collect_garbage = _helper(real_gc)
    started, release = threading.Event(), threading.Event()

    class Slow:
        def __init__(self):
            self.me = self

        def __del__(self):
            started.set()
            release.wait(30)

    class Ours:
        def __init__(self):
            self.me = self

    was_enabled = real_gc.isenabled()
    real_gc.disable()
    a = threading.Thread(target = real_gc.collect)
    try:
        real_gc.collect()
        Slow()
        a.start()
        assert started.wait(30)
        for _ in range(5):
            Ours()
        passes = []

        def before():
            # Thread A finishes between our first and second pass.
            if passes:
                release.set()
                a.join(30)
            passes.append(1)

        collect_garbage(before = before)
        # Skipped, then found ours, then quiet.
        assert len(passes) == 3
        assert not any(type(o) is Ours for o in real_gc.get_objects())
    finally:
        release.set()
        a.join(30)
        if was_enabled:
            real_gc.enable()


def test_every_repeated_collect_in_save_py_goes_through_the_helper():
    tree = _tree()
    helper = _function(tree, "_collect_garbage")
    inside = set(map(id, ast.walk(helper)))
    loops = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, (ast.For, ast.While))
        and id(n) not in inside
        and any(_is_collect(c) for c in ast.walk(n))
    ]
    assert loops == []
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_collect_garbage"
    ]
    assert len(calls) == CALL_SITES


def test_unsloth_save_model_releases_state_dict_with_one_collect():
    # The per-entry release loop allocates nothing; the collect after it finds every cycle.
    fn = _function(_tree(), "unsloth_save_model")
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "enumerate(state_dict.items())"
    )
    assert not any(_is_collect(c) for c in ast.walk(loop))
    after = fn.body[fn.body.index(loop) + 1 :]
    assert any(_is_collect(c) for stmt in after[:4] for c in ast.walk(stmt))
