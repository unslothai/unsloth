# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deferred speed profile warm-up: the 2nd image starts the compile pool, fills Triton's key and does the
engagement's first-use imports on a side thread, so the 3rd image does not pay them inside its render. CPU only."""

from __future__ import annotations

import ast
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

from core.inference import diffusion_bg_compile as bg

LOG = logging.getLogger("test_diffusion_deferred_prewarm")
DIFFUSION = Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(bg, "_PREWARM_THREAD", None)
    monkeypatch.delenv(bg.PREWARM_ENV, raising = False)
    yield
    thread = bg._PREWARM_THREAD
    if thread is not None:
        thread.join(10)


@pytest.fixture
def recorder(monkeypatch, fresh):
    calls: list = []
    release = threading.Event()

    def pool(logger = None):
        calls.append(("pool", threading.current_thread().name))
        return True

    def key(logger = None):
        # Blocks until the test lets it go: proves the caller does not wait for the warm-up.
        release.wait(30)
        calls.append(("key", threading.current_thread().name))
        return True

    def imports(logger = None):
        calls.append(("imports", threading.current_thread().name))
        return 3

    monkeypatch.setattr(bg, "warm_compile_pool", pool)
    monkeypatch.setattr(bg, "warm_triton_key", key)
    monkeypatch.setattr(bg, "warm_engage_imports", imports)
    return calls, release


def test_prewarm_runs_off_the_calling_thread_and_never_blocks_it(recorder):
    calls, release = recorder
    assert bg.start_deferred_prewarm(LOG) is True
    # The key step is still blocked, yet the caller (the 2nd image's render) already has control back.
    assert not release.is_set()
    assert ("key", "unsloth-deferred-prewarm") not in calls
    release.set()
    assert bg.wait_prewarm(10) is True
    assert [c[0] for c in calls] == ["pool", "key", "imports"]
    assert {c[1] for c in calls} == {"unsloth-deferred-prewarm"}
    assert threading.current_thread().name not in {c[1] for c in calls}


def test_prewarm_starts_once(recorder):
    calls, release = recorder
    release.set()
    assert bg.start_deferred_prewarm(LOG) is True
    assert bg.start_deferred_prewarm(LOG) is False
    assert bg.wait_prewarm(10) is True
    assert [c[0] for c in calls] == ["pool", "key", "imports"]


@pytest.mark.parametrize("raw", ["0", "off", "false", "no"])
def test_kill_switch_warms_the_pool_inline_as_before(monkeypatch, recorder, raw):
    calls, release = recorder
    release.set()
    monkeypatch.setenv(bg.PREWARM_ENV, raw)
    assert bg.start_deferred_prewarm(LOG) is True
    assert bg._PREWARM_THREAD is None
    assert calls == [("pool", threading.current_thread().name)]


def test_wait_prewarm_without_a_thread_or_from_inside_it(fresh, monkeypatch):
    assert bg.wait_prewarm(0.1) is True
    seen: list = []
    monkeypatch.setattr(
        bg, "warm_compile_pool", lambda logger = None: seen.append(bg.wait_prewarm(5))
    )
    monkeypatch.setattr(bg, "warm_triton_key", lambda logger = None: True)
    monkeypatch.setattr(bg, "warm_engage_imports", lambda logger = None: 0)
    assert bg.start_deferred_prewarm(LOG) is True
    assert bg.wait_prewarm(10) is True
    # Joining itself would deadlock: the thread's own call returns at once.
    assert seen == [True]


def test_wait_prewarm_reports_a_thread_still_running(recorder):
    calls, release = recorder
    assert bg.start_deferred_prewarm(LOG) is True
    assert bg.wait_prewarm(0.05) is False
    release.set()
    assert bg.wait_prewarm(10) is True


def _fake_triton(
    monkeypatch,
    *,
    runtime_key = None,
    compiler_key = None,
):
    triton = types.ModuleType("triton")
    runtime = types.ModuleType("triton.runtime")
    cache = types.ModuleType("triton.runtime.cache")
    compiler = types.ModuleType("triton.compiler")
    compiler_mod = types.ModuleType("triton.compiler.compiler")
    if runtime_key is not None:
        cache.triton_key = runtime_key
    if compiler_key is not None:
        compiler_mod.triton_key = compiler_key
    triton.runtime, runtime.cache = runtime, cache
    triton.compiler, compiler.compiler = compiler, compiler_mod
    for name, module in {
        "triton": triton,
        "triton.runtime": runtime,
        "triton.runtime.cache": cache,
        "triton.compiler": compiler,
        "triton.compiler.compiler": compiler_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_warm_triton_key_fills_the_key(monkeypatch):
    calls: list = []
    _fake_triton(monkeypatch, runtime_key = lambda: calls.append("runtime") or "k")
    assert bg.warm_triton_key(LOG) is True
    assert calls == ["runtime"]


def test_warm_triton_key_falls_back_to_the_compiler_module(monkeypatch):
    calls: list = []
    _fake_triton(monkeypatch, compiler_key = lambda: calls.append("compiler") or "k")
    assert bg.warm_triton_key(LOG) is True
    assert calls == ["compiler"]


def test_warm_triton_key_never_raises(monkeypatch):
    def boom():
        raise OSError("libtriton unreadable")

    _fake_triton(monkeypatch, runtime_key = boom)
    assert bg.warm_triton_key(LOG) is False
    _fake_triton(monkeypatch)
    assert bg.warm_triton_key(LOG) is False


def test_warm_triton_key_real_triton_is_cached():
    triton_cache = pytest.importorskip("triton.runtime.cache")
    key = getattr(triton_cache, "triton_key", None)
    if key is None or not hasattr(key, "cache_info"):
        pytest.skip("this Triton has no cached triton_key")
    assert bg.warm_triton_key(LOG) is True
    assert key.cache_info().currsize == 1


def test_engage_imports_keep_going_and_leave_the_patch_helpers_alone(monkeypatch):
    from core.inference import diffusion_patch_backend

    monkeypatch.setattr(diffusion_patch_backend, "_HELPERS", None)
    imported: list = []
    import importlib

    real = importlib.import_module

    def fake_import(name, *args, **kwargs):
        imported.append(name)
        if name.startswith("unsloth_zoo"):
            raise ImportError("no unsloth_zoo here")
        if name.endswith("diffusion_vae_fused"):
            return types.SimpleNamespace(_kernels = lambda: None)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert bg.warm_engage_imports(LOG) == 1
    assert imported[0] == "unsloth_zoo.temporary_patches.utils"
    assert imported[1].endswith("diffusion_vae_fused")
    # The 3rd image still runs the helpers' own resolution (and its import-unsloth retry).
    assert diffusion_patch_backend._HELPERS is None


def _function(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _bg_calls(node: ast.AST) -> list:
    out = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "bg_compile"
        ):
            out.append((sub.func.attr, sub.lineno))
    return out


def test_second_image_starts_the_side_warm_up():
    tree = ast.parse(DIFFUSION.read_text(encoding = "utf-8"))
    names = [n for n, _ in _bg_calls(_function(tree, "DiffusionBackend", "generate"))]
    assert "start_deferred_prewarm" in names
    # The pool warm-up now runs only inside the side thread (or inline behind the kill switch).
    assert "warm_compile_pool" not in names


def test_engagement_joins_the_warm_up_before_patching():
    tree = ast.parse(DIFFUSION.read_text(encoding = "utf-8"))
    fn = _function(tree, "DiffusionBackend", "_engage_deferred_profile")
    waits = [line for n, line in _bg_calls(fn) if n == "wait_prewarm"]
    patches = [
        sub.lineno
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "install_patches"
    ]
    assert waits and patches and min(waits) < min(patches)
