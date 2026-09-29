# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""`usable_sync_api` tells the real `playwright.sync_api` from what else answers to that name.

A browser test that skips on `importorskip` alone errors in the CPU job once another test has
put a stub in `sys.modules`, and whether that has happened depends on collection order.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _playwright_robust import usable_sync_api  # noqa: E402


def _sync_api_stub() -> types.ModuleType:
    """The shape test_heavy_thread_measurement_integrity.py installs: every name answers with a
    callable that raises, dunders do not. Rebuilt here rather than imported, since importing
    that module installs its stub and loads its harness as a side effect."""
    module = types.ModuleType("playwright.sync_api")

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(name))

    module.__getattr__ = __getattr__
    return module


def _install(monkeypatch, sync_api):
    package = types.ModuleType("playwright")
    package.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def test_the_collection_time_stub_is_not_playwright(monkeypatch):
    _install(monkeypatch, _sync_api_stub())
    assert usable_sync_api() is None


def test_a_namespace_package_is_not_playwright(monkeypatch):
    namespace = types.ModuleType("playwright.sync_api")
    namespace.__path__ = []
    _install(monkeypatch, namespace)
    assert usable_sync_api() is None


def test_a_missing_package_is_not_playwright(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert usable_sync_api() is None


def test_a_module_with_a_file_behind_it_is_used(monkeypatch, tmp_path):
    real = types.ModuleType("playwright.sync_api")
    real.__file__ = str(tmp_path / "sync_api" / "__init__.py")
    _install(monkeypatch, real)
    assert usable_sync_api() is real
