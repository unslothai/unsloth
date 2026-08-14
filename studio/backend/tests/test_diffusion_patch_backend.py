# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the vetted patch entry point (``diffusion_patch_backend.py``).

Focused on the gate around the ``unsloth`` retry: it exists so a process that never imported
unsloth (the test suite, a worker) still installs patches instead of silently running unpatched,
but it must never fire where the import cannot succeed, because it is expensive enough there to
take a small CI runner down.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_patch_backend as pb

_SENTINEL_ERROR = ImportError("Please install Unsloth via `pip install unsloth`!")


@pytest.fixture(autouse = True)
def _reset_memo(monkeypatch):
    pb._HELPERS = None
    monkeypatch.delenv("UNSLOTH_ALLOW_CPU", raising = False)
    yield
    pb._HELPERS = None


def _torch(*, cuda = False, xpu = False):
    return types.SimpleNamespace(
        cuda = types.SimpleNamespace(is_available = lambda: cuda),
        xpu = types.SimpleNamespace(is_available = lambda: xpu),
    )


def _modules(
    monkeypatch,
    *,
    torch = None,
    unsloth = False,
):
    """Stub sys.modules so the gate sees a chosen torch / unsloth state."""
    mods = dict(sys.modules)
    mods.pop("unsloth", None)
    mods.pop("torch", None)
    if torch is not None:
        mods["torch"] = torch
    if unsloth:
        mods["unsloth"] = types.ModuleType("unsloth")
    monkeypatch.setattr(sys, "modules", mods)


def test_retry_skipped_without_a_supported_accelerator(monkeypatch):
    # A CPU-only or MPS host cannot import unsloth, so paying ~940 MB of RSS to find out is pure cost. Ungated this took down a Linux CI runner and a 7 GB macOS one.
    _modules(monkeypatch, torch = _torch())
    assert pb._retry_could_help(_SENTINEL_ERROR) is False


def test_retry_skipped_when_torch_is_not_loaded(monkeypatch):
    # The retry must never be the thing that loads torch into a process that had avoided it.
    _modules(monkeypatch, torch = None)
    assert pb._retry_could_help(_SENTINEL_ERROR) is False


@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_retry_runs_on_an_accelerator_unsloth_supports(monkeypatch, device):
    # The case the retry exists for: a GPU host whose process has simply not imported unsloth yet.
    _modules(monkeypatch, torch = _torch(**{device: True}))
    assert pb._retry_could_help(_SENTINEL_ERROR) is True


def test_retry_runs_on_cpu_when_explicitly_allowed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    _modules(monkeypatch, torch = _torch())
    assert pb._retry_could_help(_SENTINEL_ERROR) is True


def test_retry_skipped_when_unsloth_is_already_imported(monkeypatch):
    # Then the sentinel would already be set and the first attempt would have worked, so re-importing cannot fix the failure.
    _modules(monkeypatch, torch = _torch(cuda = True), unsloth = True)
    assert pb._retry_could_help(_SENTINEL_ERROR) is False


def test_retry_skipped_for_a_non_import_failure(monkeypatch):
    # A broken patch_function is not fixed by importing unsloth.
    _modules(monkeypatch, torch = _torch(cuda = True))
    assert pb._retry_could_help(RuntimeError("boom")) is False


def test_retry_skipped_when_the_device_probe_raises(monkeypatch):
    # An unprobeable device is not one unsloth can use, so fail closed rather than pay the import.
    broken = types.SimpleNamespace(
        cuda = types.SimpleNamespace(is_available = lambda: (_ for _ in ()).throw(RuntimeError())),
        xpu = None,
    )
    _modules(monkeypatch, torch = broken)
    assert pb._retry_could_help(_SENTINEL_ERROR) is False


def test_helpers_memoises_the_unavailable_result(monkeypatch):
    # Resolution can import unsloth, so it must be attempted at most once per process.
    attempts: list[int] = []

    def _boom():
        attempts.append(1)
        raise _SENTINEL_ERROR

    monkeypatch.setattr(pb, "_retry_could_help", lambda exc: False)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.temporary_patches.utils", None)
    _modules(monkeypatch, torch = _torch())
    assert pb._helpers() is None
    assert pb._helpers() is None


def test_apply_and_revert_are_no_ops_when_helpers_are_unavailable(monkeypatch):
    # The contract the callers rely on: never raise, just report that nothing was patched.
    monkeypatch.setattr(pb, "_helpers", lambda: None)
    target = types.SimpleNamespace(fn = lambda: 1)
    assert pb.apply_patch(target, "fn", lambda: 2) is False
    assert pb.revert_patch(target, "fn") is False
    assert target.fn() == 1
