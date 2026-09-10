# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the dense quant capability reported by ``/api/system``."""

import ast
import sys
import types
from pathlib import Path

import pytest
from typing import Optional

_BACKEND = Path(__file__).resolve().parent.parent


def _src(name: str) -> str:
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


def _dense_quant_supported_src() -> str:
    return _src("_dense_quant_schemes")


# Arch-nested scheme sets, least capable first, mirroring _SCHEME_MIN_CAPABILITY.
AMPERE = ("int8",)
ADA = ("int8", "fp8")
BLACKWELL = ("int8", "fp8", "nvfp4", "mxfp8")


def _run(monkeypatch, *, device_count, capable_by_ordinal):
    """Run uncached copies of the real ``main.py`` bodies with mocked dependencies.

    ``capable_by_ordinal`` maps each resolved target to its scheme tuple; a bool is accepted as
    shorthand for "every scheme" / "none"."""
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

    def _schemes(target):
        value = capable_by_ordinal[target]
        if isinstance(value, bool):
            return BLACKWELL if value else ()
        return value

    fake_device = types.ModuleType("core.inference.diffusion_device")
    fake_device.diffusion_device_scope = _Scope
    fake_device.resolve_diffusion_device_target = lambda ordinal = None: ordinal
    fake_quant = types.ModuleType("core.inference.diffusion_transformer_quant")
    fake_quant.dense_quant_probed_schemes = _schemes

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_device", fake_device)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", fake_quant)

    namespace: dict = {"Optional": Optional}
    exec(_src("_dense_quant_schemes"), namespace)  # noqa: S102 -- the real body, not a copy of it
    exec(_src("_dense_quant_supported"), namespace)  # noqa: S102
    return namespace["_dense_quant_supported"](), scoped


def _run_schemes(monkeypatch, *, device_count, capable_by_ordinal):
    """The scheme tuple ``/api/system`` publishes, under the same mocks."""
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
    fake_quant.dense_quant_probed_schemes = lambda target: capable_by_ordinal[target]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_device", fake_device)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", fake_quant)

    namespace: dict = {"Optional": Optional}
    exec(_src("_dense_quant_schemes"), namespace)  # noqa: S102
    return namespace["_dense_quant_schemes"](), scoped


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

    schemes, _ = _run_schemes(monkeypatch, device_count = 1, capable_by_ordinal = _Boom())
    assert schemes == ()
    result, _ = _run(monkeypatch, device_count = 1, capable_by_ordinal = _Boom())
    assert result is False


@pytest.mark.parametrize("needle", ["diffusion_device_scope", "device_count"])
def test_the_wiring_stays_in_place(needle):
    assert needle in _dense_quant_supported_src()


def test_the_capability_is_published_and_never_frozen():
    """`/api/system` carries both, and the scheme list must not be memoised.

    `dense_quant_probed_schemes` answers the arch floor until the load path pays for a smoke
    verdict and the cached verdict afterwards, so an lru_cache here would pin the cold answer for
    the life of the process."""
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '"dense_quant_supported": _dense_quant_supported()' in src
    assert '"dense_quant_schemes": list(_dense_quant_schemes())' in src
    node = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_dense_quant_schemes"
    )
    assert node.decorator_list == []
    assert "dense_quant_probed_schemes" in _src("_dense_quant_schemes")


def test_the_published_list_follows_the_probe_as_it_warms(monkeypatch):
    """A verdict the loader has already paid for narrows what the picker advertises."""
    answers = {None: ADA}
    schemes, _ = _run_schemes(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert schemes == ADA
    # The smoke probe later rules fp8 out on this card; the next poll must say so.
    answers[None] = AMPERE
    schemes, _ = _run_schemes(monkeypatch, device_count = 1, capable_by_ordinal = answers)
    assert schemes == AMPERE


# Per-scheme capability: one bit cannot tell an Ampere host from an Ada one.


def test_a_single_card_publishes_its_own_schemes(monkeypatch):
    schemes, scoped = _run_schemes(monkeypatch, device_count = 1, capable_by_ordinal = {None: AMPERE})
    assert schemes == AMPERE
    assert scoped == []


def test_a_mixed_host_publishes_only_what_every_card_runs(monkeypatch):
    """An Ada card beside an Ampere one may not advertise fp8: the picker cannot see the pick."""
    schemes, scoped = _run_schemes(
        monkeypatch, device_count = 2, capable_by_ordinal = {0: ADA, 1: AMPERE}
    )
    assert schemes == AMPERE
    assert scoped == [0, 1]
    # Order of the cards must not change the answer.
    schemes, _ = _run_schemes(monkeypatch, device_count = 2, capable_by_ordinal = {0: AMPERE, 1: ADA})
    assert schemes == AMPERE


def test_one_incapable_card_empties_the_list(monkeypatch):
    schemes, _ = _run_schemes(monkeypatch, device_count = 2, capable_by_ordinal = {0: ADA, 1: ()})
    assert schemes == ()


def test_the_published_bit_is_exactly_a_non_empty_scheme_list(monkeypatch):
    """`dense_quant_supported` must stay the "every card is capable" answer it was."""
    for count, by_ordinal, expected in [
        (1, {None: AMPERE}, True),
        (1, {None: ()}, False),
        (2, {0: ADA, 1: AMPERE}, True),
        (2, {0: ADA, 1: ()}, False),
        (0, {None: ()}, False),
    ]:
        result, _ = _run(monkeypatch, device_count = count, capable_by_ordinal = by_ordinal)
        assert result is expected, (count, by_ordinal)


def test_the_scheme_floors_are_nested_so_the_intersection_is_never_a_surprise():
    """`_dense_quant_supported` derives from an intersection, which is only equivalent to "every
    card is capable" while each arch's scheme set contains every lower arch's."""
    from core.inference import diffusion_transformer_quant as tq

    caps = sorted(set(tq._SCHEME_MIN_CAPABILITY.values()))
    sets = [{s for s, floor in tq._SCHEME_MIN_CAPABILITY.items() if cap >= floor} for cap in caps]
    for smaller, larger in zip(sets, sets[1:]):
        assert smaller <= larger, (smaller, larger)
    assert all(sets), "every arch tier must run at least one scheme"


def test_every_ladder_scheme_clears_its_own_floor():
    """The advertised floors and the auto ladder must not drift apart."""
    from core.inference import diffusion_transformer_quant as tq

    for floor, schemes in tq._AUTO_LADDER:
        for scheme in schemes:
            assert tq._SCHEME_MIN_CAPABILITY[scheme] <= floor, (scheme, floor)
    assert set(tq._SCHEME_MIN_CAPABILITY) == set(tq.TQ_SCHEMES)
