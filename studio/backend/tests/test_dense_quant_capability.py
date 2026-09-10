# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the dense quant capability reported by ``/api/system``."""

import ast
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _src(name: str) -> str:
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
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

    namespace: dict = {}
    exec(_src("_dense_quant_supported"), namespace)  # noqa: S102 -- the real body, not a copy of it
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


def test_a_probe_failure_reports_incapable(monkeypatch):
    """A probe that raises must report the conservative result, not fail the status request."""

    class _Boom(dict):
        def __getitem__(self, key):
            raise RuntimeError("driver went away")

    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = _Boom())
    assert result is False


@pytest.mark.parametrize("needle", ["diffusion_device_scope", "device_count"])
def test_the_wiring_stays_in_place(needle):
    assert needle in _src("_dense_quant_supported")


def test_the_capability_is_published_and_never_memoised():
    """`/api/system` carries the bit, and it must be recomputed on every poll.

    `dense_quant_host_capable` counts an UNPROBED scheme as usable, so a cold backend answers yes
    and the first real load can then record a kernel failure. Memoising here would pin that
    optimistic answer for the life of the process.
    """
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '"dense_quant_supported": _dense_quant_supported()' in src
    node = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_dense_quant_supported"
    )
    assert node.decorator_list == []
    assert "lru_cache" not in src.split("def _dense_quant_supported")[0][-400:]


def test_the_published_bit_follows_the_probe_as_it_warms(monkeypatch):
    """A verdict the loader has since paid for must reach the next poll."""
    answers = {None: True}
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert result is True
    answers[None] = False
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert result is False


def test_only_the_capability_bit_is_published():
    """One bit, not a scheme list.

    A per-scheme list has to be kept in step with every input the loader's selector reads
    (precision, speed, memory, the family deny list, a per-card kernel probe, the ladder's own
    ordering). The picker cannot see the request, so it states the capability and lets ``resolved``
    report the precision that actually ran.
    """
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    for gone in ("dense_quant_schemes", "dense_quant_auto_schemes", "dense_quant_probed_schemes"):
        assert gone not in src, gone
