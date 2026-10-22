# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Re-detection flips CHAT_ONLY after an MLX repair, so the warm default snapshot must be rechecked."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.defaults as defaults_mod  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
from core.inference.orchestrator import InferenceOrchestrator  # noqa: E402


def _orchestrator(monkeypatch, models):
    monkeypatch.setattr(InferenceOrchestrator, "_fetch_top_models", lambda self: None)
    monkeypatch.setattr(defaults_mod, "get_default_models", lambda: list(models))
    return InferenceOrchestrator()


def test_defaults_refresh_when_hardware_is_redetected(monkeypatch):
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 1)
    orch = _orchestrator(monkeypatch, ["chat-only-gguf"])
    assert orch.default_models == ["chat-only-gguf"]

    monkeypatch.setattr(defaults_mod, "get_default_models", lambda: ["full-a", "full-b"])
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 2)

    assert orch.default_models == [
        "full-a",
        "full-b",
    ], "the orchestrator is still serving the pre-repair chat-only list"


def test_defaults_are_not_recomputed_without_a_redetect(monkeypatch):
    """The refresh is keyed on the generation, not on every read."""
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 5)
    calls = []

    def _counted():
        calls.append(1)
        return ["a"]

    monkeypatch.setattr(InferenceOrchestrator, "_fetch_top_models", lambda self: None)
    monkeypatch.setattr(defaults_mod, "get_default_models", _counted)
    orch = InferenceOrchestrator()

    orch.default_models
    orch.default_models

    assert len(calls) == 1, f"recomputed {len(calls)}x without a re-detection"


def test_detection_generation_advances_on_every_settled_detection():
    """The counter is what makes the staleness check possible."""
    before = hw.DETECTION_GENERATION
    hw.ensure_hardware_detected()
    hw.detect_hardware()
    assert hw.DETECTION_GENERATION > before


def test_a_forced_redetect_unpublishes_while_it_runs(monkeypatch):
    """Unpublish detection during a forced pass, or health reads the reset CHAT_ONLY state as settled."""
    seen = []

    def _mid_pass():
        seen.append(hw.DETECTION_COMPLETE.is_set())
        return hw.DeviceType.CPU

    hw.DETECTION_COMPLETE.set()
    monkeypatch.setattr(hw, "_detect_hardware_locked", _mid_pass)

    hw.detect_hardware()

    assert seen == [False], "detection was still published while the forced pass ran"
    assert hw.DETECTION_COMPLETE.is_set(), "the forced pass must republish when it settles"


def test_a_failed_forced_redetect_does_not_leave_detection_unpublished(monkeypatch):
    """A failed forced pass must restore the detected state, or health answers provisionally forever."""

    def _boom():
        raise RuntimeError("probe exploded mid-pass")

    hw.DETECTION_COMPLETE.set()
    monkeypatch.setattr(hw, "_detect_hardware_locked", _boom)

    try:
        hw.detect_hardware()
    except RuntimeError:
        pass

    assert hw.DETECTION_COMPLETE.is_set(), (
        "a failed forced re-detect left detection unpublished; health would stay "
        "provisional for the rest of the process"
    )
