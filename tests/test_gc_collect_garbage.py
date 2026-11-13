# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Model loading's repeated full gc.collect() passes stop only after a pass that really ran and found nothing.

CPython's gc.collect() returns 0 without collecting while another thread's collection is in progress, so a
plain `collected == 0` check can end the loop before any pass ran and leave the garbage it was meant to free.
The release sites call unsloth/_gc_passes.collect_garbage, which also checks the full-collection stats.
CPU only: the helper is loaded on its own and the release statements are sliced out of the sources with ast."""

from __future__ import annotations

import ast
import gc as real_gc
import importlib.util
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth"
# file -> number of release sites (load, get_peft_model, patch_peft_model, post_patch)
SITES = {
    "models/vision.py": 5,
    "models/llama.py": 4,
    "models/gemma.py": 1,
    "models/gemma2.py": 1,
    "models/granite.py": 1,
}


def _load_helper():
    path = ROOT / "_gc_passes.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_gc_passes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_collect(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and getattr(node.func.value, "id", None) == "gc"
    )


def _is_helper_call(node):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "collect_garbage"
    )


def _release_statements(rel):
    tree = ast.parse((ROOT / rel).read_text(encoding = "utf-8"))
    found = []
    for node in ast.walk(tree):
        if _is_helper_call(node):
            found.append(node)
        elif isinstance(node, ast.For) and any(_is_collect(c) for c in ast.walk(node)):
            found.append(node)
    return sorted(found, key = lambda n: n.lineno)


CASES = [
    pytest.param(rel, i, id = f"{rel}:{i}") for rel, count in SITES.items() for i in range(count)
]


@pytest.mark.parametrize("rel,count", list(SITES.items()))
def test_every_release_site_uses_the_helper(rel, count):
    statements = _release_statements(rel)
    assert len(statements) == count
    assert all(_is_helper_call(s) for s in statements), [s.lineno for s in statements]
    for s in statements:
        assert [k.arg for k in s.value.keywords] == ["after"]
        assert getattr(s.value.keywords[0].value, "id", None) == "clean_gpu_cache"


@pytest.mark.parametrize("rel,index", CASES)
def test_a_pass_skipped_for_another_threads_collection_does_not_end_it(rel, index):
    # Thread A is inside a collection, blocked in a finalizer, when the release starts. Its first
    # gc.collect() returns 0 without collecting; stopping there would leave our garbage behind.
    statement = _release_statements(rel)[index]
    code = compile(ast.Module([statement], []), rel, "exec")
    helper = _load_helper()
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

    flushes = []

    def clean_gpu_cache():
        # Thread A finishes during the first flush, between our first and second pass.
        release.set()
        a.join(30)
        flushes.append(1)

    namespace = {"gc": real_gc, "clean_gpu_cache": clean_gpu_cache}
    if helper is not None:
        namespace["collect_garbage"] = helper.collect_garbage
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
        exec(code, namespace)
        # Skipped, then found ours, then quiet; the cache is flushed after each.
        assert len(flushes) == 3
        assert not any(type(o) is Ours for o in real_gc.get_objects())
    finally:
        release.set()
        a.join(30)
        if was_enabled:
            real_gc.enable()


class _FakeGC:
    """`script` items: an int or None is what a pass that runs returns; "skip" returns 0 without collecting."""

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
        ([None, None, None], 3),  # gc.collect swapped for `lambda: None`
        (["skip", 0, 9], 2),
        (["skip", "skip", "skip"], 3),
    ],
)
def test_collect_garbage_stops_after_a_pass_that_ran_and_found_nothing(
    monkeypatch, script, expected
):
    helper = _load_helper()
    assert helper is not None
    gc = _FakeGC(script)
    monkeypatch.setattr(helper, "gc", gc)
    order = []
    helper.collect_garbage(
        before = lambda: order.append("before"), after = lambda: order.append("after")
    )
    assert gc.calls == expected
    assert order == ["before", "after"] * expected


def test_collect_garbage_keeps_every_pass_with_gc_callbacks(monkeypatch):
    helper = _load_helper()
    assert helper is not None
    gc = _FakeGC([0, 0, 0], callbacks = [print])
    monkeypatch.setattr(helper, "gc", gc)
    helper.collect_garbage()
    assert gc.calls == 3
