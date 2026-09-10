# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the dense quant capability reported by ``/api/system``."""

import ast
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _dense_quant_supported_src() -> str:
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_dense_quant_supported"
    )
    return ast.get_source_segment(src, node)


def _run(monkeypatch, *, device_count, capable_by_ordinal):
    """Run an uncached copy of ``_dense_quant_supported`` with mocked dependencies."""
    scoped: list = []

    fake_torch = types.SimpleNamespace(
        cuda = types.SimpleNamespace(
            is_available = lambda: device_count > 0,
            device_count = lambda: device_count,
        )
    )

    class _Scope:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        def __enter__(self):
            scoped.append(self.ordinal)
            return None

        def __exit__(self, *exc):
            return False

    fake_device = types.ModuleType("core.inference.diffusion_device")
    fake_device.diffusion_device_scope = _Scope
    fake_device.resolve_diffusion_device_target = lambda ordinal = None: ordinal
    fake_quant = types.ModuleType("core.inference.diffusion_transformer_quant")
    fake_quant.dense_quant_host_capable = lambda target: capable_by_ordinal[target]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_device", fake_device)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", fake_quant)

    namespace: dict = {"functools": types.SimpleNamespace(lru_cache = lambda maxsize: (lambda f: f))}
    exec(_dense_quant_supported_src(), namespace)  # noqa: S102 -- the real body, not a copy of it
    return namespace["_dense_quant_supported"](), scoped


def test_a_single_capable_gpu_reports_capable(monkeypatch):
    result, scoped = _run(monkeypatch, device_count = 1, capable_by_ordinal = {None: True})
    assert result is True
    assert scoped == []


def test_a_single_incapable_gpu_reports_incapable(monkeypatch):
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = {None: False})
    assert result is False


def test_every_visible_card_must_be_capable(monkeypatch):
    """Mixed-capability hosts must not advertise dense quant."""
    result, scoped = _run(monkeypatch, device_count = 2, capable_by_ordinal = {0: True, 1: False})
    assert result is False
    # Probe each ordinal under its own device scope.
    assert scoped == [0, 1]


def test_a_homogeneous_multi_gpu_host_still_reports_capable(monkeypatch):
    result, scoped = _run(monkeypatch, device_count = 2, capable_by_ordinal = {0: True, 1: True})
    assert result is True
    assert scoped == [0, 1]


def test_no_gpu_reports_incapable(monkeypatch):
    result, _ = _run(monkeypatch, device_count = 0, capable_by_ordinal = {None: False})
    assert result is False


def test_a_probe_failure_reports_incapable():
    """Probe failures must report the conservative result."""
    src = _dense_quant_supported_src()
    assert "except Exception" in src and "return False" in src.split("except Exception")[-1]


@pytest.mark.parametrize("needle", ["diffusion_device_scope", "device_count"])
def test_the_wiring_stays_in_place(needle):
    assert needle in _dense_quant_supported_src()


def test_the_probe_is_cached_and_published():
    """The capability is cached and included in ``/api/system``."""
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "@functools.lru_cache(maxsize = 1)\ndef _dense_quant_supported" in src
    assert '"dense_quant_supported": _dense_quant_supported()' in src
