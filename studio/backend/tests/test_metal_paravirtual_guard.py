# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A virtualised Apple GPU must fall back to CPU; a real one must not.

The second half is the half that matters: forcing gpu_layers=0 on every Mac would
"fix" the corrupt output by throwing away Metal for every real user, so these tests
pin the discrimination and not just the fallback.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import _metal_device_is_paravirtual


@pytest.fixture(autouse = True)
def _clear_cache():
    """The detector is lru_cached so a real machine only pays for the probe once."""
    _metal_device_is_paravirtual.cache_clear()
    yield
    _metal_device_is_paravirtual.cache_clear()


def _fake_mlx(device_name: str):
    module = types.ModuleType("mlx.core")
    module.device_info = lambda: {"device_name": device_name}
    parent = types.ModuleType("mlx")
    parent.core = module
    return parent, module


@pytest.mark.parametrize(
    "device_name, expected",
    [
        ("Apple Paravirtual device", True),
        ("apple paravirtual device", True),  # matching must not be case-sensitive
        ("Apple M1", False),
        ("Apple M3 Max", False),
        ("Apple M4 Pro", False),
    ],
)
def test_only_virtualised_apple_gpus_fall_back(monkeypatch, device_name, expected):
    monkeypatch.setattr(sys, "platform", "darwin")
    parent, core = _fake_mlx(device_name)
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    # Inert probe output so this measures the name matching, not system_profiler.
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(stdout = "Chipset Model: Apple M3 Max"),
    )
    assert _metal_device_is_paravirtual() is expected


def test_non_darwin_never_pays_for_the_probe(monkeypatch):
    """Linux and Windows short-circuit: there is no Metal to be virtualised."""
    monkeypatch.setattr(sys, "platform", "linux")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("probed for a Metal device off macOS")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", explode)
    assert _metal_device_is_paravirtual() is False


def test_system_profiler_catches_it_when_mlx_is_absent(monkeypatch):
    """MLX is not on every Mac; without this fallback a virtualised machine without
    MLX would be treated as bare metal and would emit gibberish."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)  # import raises
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            stdout = "Graphics/Displays:\n  Apple Paravirtual device:\n    Vendor: Apple"
        ),
    )
    assert _metal_device_is_paravirtual() is True


def test_a_broken_probe_leaves_gpu_offload_alone(monkeypatch):
    """If neither source can answer, assume a real Mac. Guessing "virtualised" would
    silently drop everyone the probe fails on down to CPU."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)

    def explode(*args, **kwargs):
        raise OSError("system_profiler not found")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", explode)
    assert _metal_device_is_paravirtual() is False
