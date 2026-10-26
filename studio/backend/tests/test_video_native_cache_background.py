# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offloaded video loads write the native int8 build after the load is published, not inside it."""

import ast
import inspect
import threading
import types

import pytest

pytest.importorskip("torch")

import core.inference.video as V
from core.inference import diffusion_native_cache as dnc
from core.inference import video_denoiser_prequant as vdp

_ENV = getattr(V, "VIDEO_NATIVE_CACHE_BACKGROUND_ENV", "UNSLOTH_VIDEO_NATIVE_CACHE_BACKGROUND")


class _Write:
    def __init__(self, log, name):
        self.log, self.name = log, name
        self.done = threading.Event()
        self.cancelled = False

    def start(self):
        self.log.append(("start", self.name))
        self.done.set()
        return True

    def wait(self, timeout = None):
        return self.done.wait(timeout)

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def fake(monkeypatch):
    log = []
    monkeypatch.delenv(_ENV, raising = False)
    monkeypatch.setattr(vdp, "denoiser_components", lambda fam: ["transformer", "transformer_2"])
    monkeypatch.setattr(
        V, "finish_background_pins", lambda pipe, cancel = None: log.append(("pins",)) or 0.0
    )

    def save_deferred(transformer, fam, **kw):
        log.append(("deferred", kw["component"], transformer))
        return _Write(log, kw["component"])

    monkeypatch.setattr(dnc, "save_deferred", save_deferred)
    views = [types.SimpleNamespace(transformer = "hi"), types.SimpleNamespace(transformer = "lo")]
    return log, views


def test_the_kill_switches_keep_the_write_inside_the_load(monkeypatch):
    monkeypatch.delenv(_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_CACHE_ASYNC", raising = False)
    assert V._native_cache_background()
    monkeypatch.setenv(_ENV, "0")
    assert not V._native_cache_background()
    monkeypatch.delenv(_ENV)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PREQUANT_CACHE_ASYNC", "0")
    assert not V._native_cache_background()


def test_the_writes_start_after_the_pins_one_expert_at_a_time(fake):
    log, views = fake
    w = V._NativeCacheWrites(
        types.SimpleNamespace(), object(), views, {"scheme": "int8"}, base = "b", source = "s"
    )
    assert log == []  # nothing is written until the load is published
    w.start()
    w._thread.join(10)
    assert log == [
        ("pins",),
        ("deferred", "transformer", "hi"),
        ("start", "transformer"),
        ("deferred", "transformer_2", "lo"),
        ("start", "transformer_2"),
    ]
    w.finish(timeout = 1.0)


def test_an_unload_before_the_start_writes_nothing(fake):
    log, views = fake
    w = V._NativeCacheWrites(
        types.SimpleNamespace(), object(), views, {"scheme": "int8"}, base = "b", source = "s"
    )
    w.finish(timeout = 1.0)
    w.start()
    w._thread.join(10)
    assert not [e for e in log if e[0] in ("deferred", "start")]


def test_a_write_still_running_at_unload_is_waited_for_then_cancelled(fake, monkeypatch):
    log, views = fake
    hang = _Write(log, "transformer")
    hang.start = lambda: log.append(("start", "transformer")) or True  # never completes
    monkeypatch.setattr(dnc, "save_deferred", lambda *a, **k: hang)
    w = V._NativeCacheWrites(
        types.SimpleNamespace(), object(), views[:1] * 2, {"scheme": "int8"}, base = None, source = None
    )
    w.start()
    for _ in range(200):
        if ("start", "transformer") in log:
            break
        threading.Event().wait(0.01)
    w.finish(timeout = 0.05)
    assert hang.cancelled


def test_the_load_defers_the_native_save_and_starts_it_after_the_pins():
    src = inspect.getsource(V.VideoBackend.load_pipeline)
    tree = ast.parse("class _X:\n" + src if src.startswith("    ") else src)
    calls = [
        (
            node.lineno,
            node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    names = [n for _l, n in calls]
    assert "_NativeCacheWrites" in names and "_save_native_quant_to_cache" in names
    # the sync save is only the kill-switch branch
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_native_cache_background" in ast.unparse(node.test):
            assert "_NativeCacheWrites" in ast.unparse(node.body[0])
            assert "_save_native_quant_to_cache" in ast.unparse(node.orelse[0])
            break
    else:
        raise AssertionError("no _native_cache_background branch in load_pipeline")
    pins = min(line for line, n in calls if n == "start_background_pins")
    starts = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and ast.unparse(node.func.value) == "native_writes"
    ]
    assert starts and min(starts) > pins


def test_the_unload_finishes_the_native_writes():
    src = inspect.getsource(V)
    assert '_finish_native_writes(getattr(state, "pipe", None))' in src
