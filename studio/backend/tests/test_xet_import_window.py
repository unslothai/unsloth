# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A download's first ``unsloth_zoo`` import holds the media import window, so it cannot interleave
with the post-warm worker's torchao and diffusers imports (half-built ``torchao.kernel``)."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_HEALTH = "unsloth_zoo.hf_xet_health"
_SHARED = "unsloth_zoo.hf_xet_fallback"


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(("info", msg % args if args else msg))

    def warning(self, msg, *args):
        self.lines.append(("warning", msg % args if args else msg))


@pytest.fixture
def warm(monkeypatch):
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_dynamo_done", True)
    monkeypatch.setattr(torch_warmup, "_media_import_claimed", False)
    monkeypatch.setattr(torch_warmup, "_media_import_owner", None)
    monkeypatch.delenv(torch_warmup.DYNAMO_GATE_DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(torch_warmup.DYNAMO_GATE_TIMEOUT_ENV_VAR, raising = False)
    return torch_warmup


@pytest.fixture
def shim(monkeypatch):
    from utils import hf_xet_fallback

    monkeypatch.setattr(hf_xet_fallback, "_gpu_present", lambda: True)
    hf_xet_fallback._reset_optional_module_cache()
    yield hf_xet_fallback
    hf_xet_fallback._reset_optional_module_cache()


def _observing_import(warm, seen: dict, module):
    def _import(name, package = None):
        seen["name"] = name
        seen["locked"] = warm._media_import_lock.locked()
        seen["owner"] = warm._media_import_owner == threading.get_ident()
        return module

    return _import


class _RecordingFinder:
    def __init__(self, warm, seen: dict):
        self.warm = warm
        self.seen = seen

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name != _SHARED:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.seen["locked"] = self.warm._media_import_lock.locked()
        self.seen["owner"] = self.warm._media_import_owner == threading.get_ident()


def test_the_optional_import_holds_the_window(warm, shim, monkeypatch):
    seen: dict = {}
    fake = types.ModuleType(_HEALTH)
    monkeypatch.setattr(importlib, "import_module", _observing_import(warm, seen, fake))

    assert shim._load_optional(_HEALTH) is fake

    assert seen == {"name": _HEALTH, "locked": True, "owner": True}
    assert not warm._media_import_lock.locked()
    assert warm._media_import_owner is None


def test_the_shared_import_holds_the_window(warm, shim, monkeypatch):
    seen: dict = {}
    package = types.ModuleType("unsloth_zoo")
    package.__path__ = []
    finder = _RecordingFinder(warm, seen)
    monkeypatch.setattr(shim, "_shared", None)
    monkeypatch.setattr(shim, "_shared_available", None)
    monkeypatch.setattr(shim, "_shared_import_error", None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", package)
    saved = sys.modules.pop(_SHARED, None)
    sys.meta_path.insert(0, finder)
    try:
        assert shim._load_shared() is True
    finally:
        sys.meta_path.remove(finder)
        if saved is None:
            sys.modules.pop(_SHARED, None)
        else:
            sys.modules[_SHARED] = saved

    assert seen == {"locked": True, "owner": True}
    assert shim._shared.__name__ == _SHARED
    assert not warm._media_import_lock.locked()
    assert warm._media_import_owner is None


def test_the_import_waits_for_a_background_media_import_in_flight(warm, shim, monkeypatch):
    seen: dict = {}
    fake = types.ModuleType(_HEALTH)
    monkeypatch.setattr(importlib, "import_module", _observing_import(warm, seen, fake))
    in_window = threading.Event()
    release = threading.Event()

    def _background():
        with warm.background_media_import():
            in_window.set()
            release.wait(10)

    background = threading.Thread(target = _background, daemon = True)
    background.start()
    assert in_window.wait(10)

    result = []
    done = threading.Event()
    loader = threading.Thread(
        target = lambda: (result.append(shim._load_optional(_HEALTH)), done.set()), daemon = True
    )
    loader.start()
    try:
        assert not done.wait(0.5), "unsloth_zoo was imported while a background media import ran"
    finally:
        release.set()
    background.join(10)
    assert done.wait(10)
    loader.join(10)

    assert result == [fake]
    assert seen["locked"] is True and seen["owner"] is True
    assert not warm._media_import_lock.locked()


def test_the_import_still_runs_when_the_wait_expires(warm, shim, monkeypatch):
    monkeypatch.setenv(warm.DYNAMO_GATE_TIMEOUT_ENV_VAR, "0.2")
    seen: dict = {}
    fake = types.ModuleType(_HEALTH)
    monkeypatch.setattr(importlib, "import_module", _observing_import(warm, seen, fake))

    assert warm._media_import_lock.acquire(timeout = 5)
    started = time.perf_counter()
    try:
        assert shim._load_optional(_HEALTH) is fake
        assert warm._media_import_lock.locked()
    finally:
        warm._media_import_lock.release()
    assert time.perf_counter() - started >= 0.15
    assert seen == {"name": _HEALTH, "locked": True, "owner": False}


def test_a_failed_first_attempt_keeps_the_window_for_its_retry(warm, shim, monkeypatch):
    monkeypatch.setattr(shim, "_gpu_present", lambda: False)
    fake = types.ModuleType(_HEALTH)
    attempts: list = []

    def _import(name, package = None):
        attempts.append(
            (warm._media_import_lock.locked(), warm._media_import_owner == threading.get_ident())
        )
        if len(attempts) == 1:
            raise RuntimeError("init raised")
        return fake

    monkeypatch.setattr(importlib, "import_module", _import)

    assert shim._load_optional(_HEALTH) is fake

    assert attempts == [(True, True), (True, True)]
    assert not warm._media_import_lock.locked()
    assert warm._media_import_owner is None


def test_an_expired_wait_is_logged_and_releases_nothing(warm, monkeypatch):
    monkeypatch.setenv(warm.DYNAMO_GATE_TIMEOUT_ENV_VAR, "0.2")
    log = _Log()

    assert warm._media_import_lock.acquire(timeout = 5)
    try:
        with warm.hold_media_import_window("unsloth_zoo import", log):
            assert warm._media_import_owner is None
        assert warm._media_import_lock.locked()
    finally:
        warm._media_import_lock.release()

    assert [level for level, _ in log.lines] == ["info", "warning"], log.lines
    assert "unsloth_zoo import: waiting for a background media import" in log.lines[0][1]
    assert "gave up waiting" in log.lines[1][1]


@pytest.mark.parametrize("setting", ["inf", "1e30"])
def test_an_unbounded_timeout_setting_still_waits_instead_of_raising(warm, monkeypatch, setting):
    monkeypatch.setenv(warm.DYNAMO_GATE_TIMEOUT_ENV_VAR, setting)
    assert warm._dynamo_gate_timeout() == threading.TIMEOUT_MAX

    assert warm._media_import_lock.acquire(timeout = 5)
    entered = threading.Event()

    def _hold():
        with warm.hold_media_import_window("unsloth_zoo import"):
            entered.set()

    waiter = threading.Thread(target = _hold, daemon = True)
    waiter.start()
    try:
        assert not entered.wait(0.2)
    finally:
        warm._media_import_lock.release()
    assert entered.wait(5)
    waiter.join(5)
    assert not waiter.is_alive()
    assert not warm._media_import_lock.locked()


def test_a_failure_inside_the_window_releases_it(warm):
    with pytest.raises(ImportError):
        with warm.hold_media_import_window("unsloth_zoo import"):
            assert warm._media_import_lock.locked()
            raise ImportError("half-built module")

    assert not warm._media_import_lock.locked()
    assert warm._media_import_owner is None


def test_the_kill_switch_skips_the_window(warm, monkeypatch):
    monkeypatch.setenv(warm.DYNAMO_GATE_DISABLE_ENV_VAR, "1")

    with warm.hold_media_import_window("unsloth_zoo import"):
        assert not warm._media_import_lock.locked()
        assert warm._media_import_owner is None


def test_the_loader_runs_unguarded_without_the_warm_module(warm, shim, monkeypatch):
    fake = types.ModuleType(_HEALTH)
    monkeypatch.setattr(importlib, "import_module", lambda name, package = None: fake)
    monkeypatch.setitem(sys.modules, "utils.torch_warmup", None)

    assert shim._load_optional(_HEALTH) is fake
    assert not warm._media_import_lock.locked()


def test_background_work_reaching_the_import_does_not_wait_on_itself(warm):
    result = []

    def _background():
        with warm.background_media_import():
            with warm.hold_media_import_window("unsloth_zoo import"):
                result.append(warm._media_import_lock.locked())

    thread = threading.Thread(target = _background, daemon = True)
    thread.start()
    thread.join(5)

    assert not thread.is_alive(), "background work deadlocked on its own window"
    assert result == [True]
    assert not warm._media_import_lock.locked()


def _reaches_unsloth_zoo(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("unsloth_zoo") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith("unsloth_zoo")
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
    )


def _inside_the_window(fn: ast.FunctionDef) -> set:
    inside = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "_media_import_window"
            for item in node.items
        ):
            continue
        inside.update(id(inner) for inner in ast.walk(node))
    return inside


def _loaders() -> list:
    tree = ast.parse((_BACKEND / "utils" / "hf_xet_fallback.py").read_text(encoding = "utf-8"))
    loaders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_load_shared", "_load_optional"}
    ]
    assert [fn.name for fn in loaders] == ["_load_shared", "_load_optional"]
    return loaders


def _windows(fn: ast.FunctionDef) -> list:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "_media_import_window"
            for item in node.items
        )
    ]


def test_every_zoo_import_in_the_shim_is_inside_the_window():
    for fn in _loaders():
        inside = _inside_the_window(fn)
        imports = [node for node in ast.walk(fn) if _reaches_unsloth_zoo(node)]
        assert len(imports) == 2, fn.name
        assert all(id(node) in inside for node in imports), fn.name


def test_each_loader_opens_the_window_once():
    for fn in _loaders():
        assert len(_windows(fn)) == 1, fn.name


def test_the_dynamo_gate_runs_before_the_window_and_the_window_wraps_the_load_lock():
    for fn in _loaders():
        gate = min(
            (
                node.lineno
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_gate_torch_stack"
            ),
            default = None,
        )
        assert gate is not None, fn.name
        windows = _windows(fn)
        assert windows and gate < min(node.lineno for node in windows), fn.name
        locks = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.With)
            and any(getattr(item.context_expr, "id", None) == "_load_lock" for item in node.items)
        ]
        assert locks, fn.name
        for lock in locks:
            assert any(window.lineno < lock.lineno <= window.end_lineno for window in windows), (
                fn.name,
                lock.lineno,
            )
