# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the dense quant capability reported by ``/api/system``."""

import ast
import sys
import types
from pathlib import Path
from typing import Optional

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
    exec(_src("_probe_dense_quant_supported"), namespace)  # noqa: S102 -- the real body
    return namespace["_probe_dense_quant_supported"](), scoped


def test_a_single_capable_gpu_reports_capable(monkeypatch):
    result, scoped = _run(monkeypatch, device_count = 1, capable_by_ordinal = {None: True})
    assert result is True
    assert scoped == []


def test_a_single_incapable_gpu_reports_incapable(monkeypatch):
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = {None: False})
    assert result is False


def test_every_visible_card_must_be_capable(monkeypatch):
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
    assert needle in _src("_probe_dense_quant_supported")


def test_the_capability_is_published_and_never_memoised():
    """`/api/system` carries the bit, and the probe must not be pinned.

    `dense_quant_host_capable` counts an UNPROBED scheme as usable, so an early yes can be undone by
    a later load recording a kernel failure. Memoising would pin the optimistic answer forever.
    """
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '"dense_quant_supported": _dense_quant_supported()' in src
    node = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_probe_dense_quant_supported"
    )
    assert node.decorator_list == []


def test_the_polled_route_never_imports_the_ml_stack():
    """torch and torchao cost ~0.8s each and hold the GIL; /api/system is polled through startup.

    The reader answers from `sys.modules` and a value the post-warm worker resolved, so a poll on a
    cold backend imports nothing. `_await_hardware_detection` avoids the same stall.
    """
    reader = _src("_dense_quant_supported")
    assert '"torch" in sys.modules' in reader and '"torchao" in sys.modules' in reader
    # The reader itself must not reach the importing probe except behind that guard.
    assert "_probe_dense_quant_supported" not in reader
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    # ...and something off the polled path has to resolve it, or the label never appears.
    assert "_refresh_dense_quant_capability()" in _src("_post_warm_background_work")


def test_the_probe_follows_the_smoke_cache_as_it_warms(monkeypatch):
    """A verdict the loader has since paid for must reach the next resolution."""
    answers = {None: True}
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert result is True
    answers[None] = False
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert result is False


def test_the_scheme_ladder_is_published_beside_the_bit():
    """``/api/system`` publishes the scheme ladder beside the capability bit, from the same
    ``auto_scheme_candidates`` the loader's selector reads."""
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '"dense_quant_schemes": _dense_quant_schemes()' in src
    assert "auto_scheme_candidates" in _src("_probe_dense_quant_schemes")


def test_the_ladder_is_read_off_the_same_refresh_as_the_bit():
    """The reader beside the bit is a pure read, so the polled route never probes or imports torch a
    second time, and both entries come from one pass."""
    reader = _src("_dense_quant_schemes")
    assert "_probe_dense_quant_schemes" not in reader
    assert "_refresh_dense_quant_capability" not in reader
    refresh = _src("_refresh_dense_quant_capability")
    assert "_probe_dense_quant_schemes()" in refresh
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert src.index('"dense_quant_supported": _dense_quant_supported()') < src.index(
        '"dense_quant_schemes": _dense_quant_schemes()'
    )


def _run_schemes(monkeypatch, *, device_count, schemes_by_ordinal):
    """Run an uncached copy of ``_probe_dense_quant_schemes`` with mocked dependencies."""
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
    fake_quant.auto_scheme_candidates_cached = lambda target: schemes_by_ordinal[target]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_device", fake_device)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", fake_quant)

    namespace: dict = {"Optional": Optional}
    exec(_src("_probe_dense_quant_schemes"), namespace)  # noqa: S102 -- the real body
    return namespace["_probe_dense_quant_schemes"](), scoped


def test_an_ada_host_publishes_fp8_first(monkeypatch):
    result, scoped = _run_schemes(
        monkeypatch, device_count = 1, schemes_by_ordinal = {None: ("fp8", "int8")}
    )
    assert result == ["fp8", "int8"]
    assert scoped == []


def test_an_ampere_host_publishes_int8(monkeypatch):
    result, _ = _run_schemes(monkeypatch, device_count = 1, schemes_by_ordinal = {None: ("int8",)})
    assert result == ["int8"]


def test_a_mixed_host_publishes_only_what_every_card_runs(monkeypatch):
    result, scoped = _run_schemes(
        monkeypatch,
        device_count = 2,
        schemes_by_ordinal = {0: ("fp8", "int8"), 1: ("int8",)},
    )
    assert result == ["int8"]
    assert scoped == [0, 1]


def test_an_unsupported_host_publishes_nothing(monkeypatch):
    result, _ = _run_schemes(monkeypatch, device_count = 0, schemes_by_ordinal = {None: ()})
    assert result == []


def test_a_scheme_probe_failure_publishes_nothing(monkeypatch):
    class _Boom(dict):
        def __getitem__(self, key):
            raise RuntimeError("driver went away")

    result, _ = _run_schemes(monkeypatch, device_count = 1, schemes_by_ordinal = _Boom())
    assert result == []


def test_an_incapable_host_never_publishes_a_ladder():
    refresh = _src("_refresh_dense_quant_capability")
    assert "if _dense_quant_capability else []" in refresh


def test_the_warm_refresh_honours_the_torch_kill_switch():
    """UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 must keep the stack cold.

    `start_background_warm` is then a no-op and `join_background_warm` returns at once, so the
    post-warm worker reaches this point with torch unimported. Refreshing there would import torch
    and torchao and defeat the switch, so it is gated on torch already being up.
    """
    body = _src("_post_warm_background_work")
    refresh = body.index("_refresh_dense_quant_capability()")
    guard = body.rindex('"torch" in sys.modules', 0, refresh)
    assert guard != -1
    # The guard must not swallow the rest of the worker.
    assert "_start_linked_folder_auto_sync" in body[refresh:]


def test_the_polled_ladder_never_runs_the_allocating_smoke_probe(monkeypatch):
    """The ladder on the polled route reads ``_SMOKE_CACHE`` and nothing else.

    ``_scheme_supported`` spawns the up-to-180s child smoke probe, or allocates in this process when
    it cannot, and an allocator failure is deliberately not cached, so reaching it from
    ``/api/system`` would repeat the work on every poll.
    """
    from core.inference import diffusion_transformer_quant as tq

    def _never(*_a, **_k):
        raise AssertionError("the polled ladder ran the smoke probe")

    monkeypatch.setattr(tq, "_scheme_supported", _never)
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda _target: True)
    monkeypatch.setattr(tq, "_capability", lambda: (8, 9))
    monkeypatch.setattr(tq, "_smoke_cache_device_key", lambda _device: "cuda:0")
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {("fp8", "cuda:0"): False})
    # An unprobed scheme counts as usable; a probed failure does not.
    assert tq.auto_scheme_candidates_cached(object()) == ("int8",)
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {})
    assert tq.auto_scheme_candidates_cached(object()) == ("int8", "fp8")
    assert "auto_scheme_candidates_cached" in _src("_probe_dense_quant_schemes")
