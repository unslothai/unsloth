# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


@pytest.fixture
def settle(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "orchestrator_vram_settle_under_test",
        Path(_BACKEND_DIR) / "core/inference/orchestrator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [100.0]
    samples: list = []
    probes = [0]
    module.time = types.SimpleNamespace(
        monotonic = lambda: clock[0],
        sleep = lambda s: clock.__setitem__(0, clock[0] + s),
        time = lambda: clock[0],
    )

    def free_vram_mib():
        probes[0] += 1
        return dict(samples.pop(0)) if samples else None

    monkeypatch.setattr(module.InferenceOrchestrator, "_free_vram_mib", staticmethod(free_vram_mib))

    def run(sequence, **kwargs):
        samples[:] = list(sequence)
        clock[0] = 100.0
        probes[0] = 0
        result = module.InferenceOrchestrator._wait_for_worker_vram_settle(1.0, **kwargs)
        return result, round(clock[0] - 100.0, 6), probes[0]

    return run


def test_release_before_first_poll_is_recognised_with_pre_kill_sample(settle):
    freed = [{0: 40_000}] * 20
    # Without the pre-kill sample the already-freed card looks stable but unreleased: full 2 s.
    assert settle(freed) == (True, 2.0, 8)
    # With it, the gain since the kill counts as the release; one stable sample confirms it.
    assert settle(freed, before_kill_free_mib = {0: 30_000}) == (True, 0.25, 2)


def test_pre_kill_sample_without_a_release_still_waits(settle):
    held = [{0: 30_000}] * 20
    assert settle(held, before_kill_free_mib = {0: 30_000}) == (True, 2.0, 8)
    # A gain under the tolerance is noise, not a release.
    assert settle([{0: 30_200}] * 20, before_kill_free_mib = {0: 30_000}) == (True, 2.0, 8)


def test_pre_kill_sample_waits_for_a_late_release_to_settle(settle):
    sequence = [{0: 31_000}, {0: 31_000}, {0: 39_000}, {0: 40_000}, {0: 40_000}]
    assert settle(sequence, before_kill_free_mib = {0: 30_000}) == (True, 0.25, 2)
    # Still moving after the first release: keeps polling until two samples agree.
    sequence = [{0: 35_000}, {0: 39_000}, {0: 45_000}, {0: 45_100}]
    assert settle(sequence, before_kill_free_mib = {0: 30_000}) == (True, 0.75, 4)


def test_pre_kill_sample_from_other_devices_is_ignored(settle):
    freed = [{0: 40_000}] * 20
    assert settle(freed, before_kill_free_mib = {1: 30_000}) == (True, 2.0, 8)
    assert settle(freed, before_kill_free_mib = {}) == (True, 2.0, 8)


def test_expected_free_target_is_unchanged_by_pre_kill_sample(settle):
    short = [{0: 10_000}] * 20
    expected = {0: 20.0}
    without = settle(short, expected_free_gb = expected)
    assert without == (False, 2.0, 8)
    assert settle(short, expected_free_gb = expected, before_kill_free_mib = {0: 1_000}) == without
    assert settle([{0: 30_000}], expected_free_gb = expected) == (True, 0.0, 1)


def test_unreadable_telemetry_keeps_its_answer(settle):
    assert settle([]) == (True, 0.0, 1)
    assert settle([], expected_free_gb = {0: 1.0}) == (False, 0.0, 1)
    assert settle([], before_kill_free_mib = {0: 1}) == (True, 0.0, 1)
