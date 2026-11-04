# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save.py's repeated gc.collect() passes stop once a pass that really ran finds nothing."""

from __future__ import annotations

import ast
import gc as real_gc
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
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
    """ "skip" = a pass that returns 0 without collecting (another thread is mid-collection)."""

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
    assert order == ["before", "after"] * expected


def test_keeps_every_pass_with_gc_callbacks():
    gc = _FakeGC([0, 0, 0], callbacks = [print])
    _helper(gc)()
    assert gc.calls == 3


def test_a_pass_skipped_for_another_threads_collection_does_not_end_it():
    # Thread A is mid-collection, so our first pass returns 0 without collecting.
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
            if passes:
                release.set()
                a.join(30)
            passes.append(1)

        collect_garbage(before = before)
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
    fn = _function(_tree(), "unsloth_save_model")
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "enumerate(state_dict.items())"
    )
    assert not any(_is_collect(c) for c in ast.walk(loop))
    after = fn.body[fn.body.index(loop) + 1 :]
    assert any(_is_collect(c) for stmt in after[:4] for c in ast.walk(stmt))
