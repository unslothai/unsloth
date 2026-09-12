# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Manual placement's pin of the --fit flag is a function-argument decision.

A user's log recorded `GPUs free: [], --fit: on` for a launch that spawned with
`--fit off`: the decision line interpolated `use_fit` BEFORE the Manual branch
turned it off, so every Manual-mode load logged the opposite of what it ran with
(#10821). load_model cannot be driven from a unit test -- the flags are decided
on one enormous function body -- so these pin the ORDER on the source.
"""

import inspect

import pytest

from core.inference import llama_cpp as mod


def _load_model_source() -> str:
    return inspect.getsource(mod.LlamaCppBackend.load_model)


def _first_index(source: str, needle: str) -> int:
    index = source.find(needle)
    assert index != -1, f"not found in load_model: {needle!r}"
    return index


# The one predicate Manual VIABILITY rests on: a fixed layer count. The launch
# branch and the log must consult the SAME expression, or they can disagree.
_MANUAL_PREDICATE = (
    'gpu_memory_mode == "manual" and gpu_layers >= 0'
)
# How the predicate is held so the log and the launch branch cannot ask two
# different questions.
_MANUAL_MARK = "_manual_placement"


class TestTheLogReportsTheFlagTheLaunchCarries:
    def test_manual_turns_the_fit_off_above_the_decision_line(self):
        src = _load_model_source()
        hoist = _first_index(src, f"if {_MANUAL_MARK}:")
        log = _first_index(src, "GPUs free: ")
        assert hoist < log, (
            "use_fit must be False for Manual placement BEFORE the decision line is "
            "logged, or the log reports --fit on for a launch that carries --fit off"
        )

    def test_the_hoist_reads_the_launch_branchs_own_condition(self):
        """Same predicate in both places, or they can drift into opposite answers."""
        src = _load_model_source()
        assignment = src.find(f"{_MANUAL_MARK} = {_MANUAL_PREDICATE}")
        assert assignment != -1, (
            "the pre-log verdict must consult the SAME predicate the launch branch "
            "applies below"
        )
        branch = src.find(f"if {_MANUAL_PREDICATE}:")
        assert branch != -1, "the Manual launch branch is gone"


class TestTheEmptyProbeIsLabeled:
    def test_an_empty_probe_can_name_manual(self):
        src = _load_model_source()
        log = _first_index(src, "GPUs free: ")
        window = src[log: log + 300]
        assert "manual placement" in window, (
            "an empty probe in Manual mode is a discarded-on-purpose list, not a "
            "failed enumeration; the line must say so"
        )


class TestManualSurvivesThePlacementFallback:
    """The placement try/except restores use_fit=True on any pricing failure. In
    Manual mode that fallback must not leak into the launch: the branch still owns
    the verdict before it emits --gpu-layers / --fit off."""

    def test_the_launch_branch_asserts_the_fit_off(self):
        src = _load_model_source()
        branch = src.find('if gpu_memory_mode == "manual" and gpu_layers >= 0:')
        flag = src.find('"--gpu-layers", str(gpu_layers)', branch)
        assert flag != -1, "Manual no longer emits --gpu-layers"
        window = src[branch:flag]
        assert "use_fit = False" in window, (
            "the placement except can only run after this point: without the "
            "re-assert here its --fit-on fallback would launch Manual with use_fit"
        )


class TestManualIsNotAPartialPlacement:
    """partial means "the fitter could not prove a fit". Manual never fits; the user
    placed the model. Recording partial there would smear Manual failure onto Auto."""

    def test_the_verdict_reads_the_post_hoist_use_fit(self):
        src = _load_model_source()
        hoist = _first_index(src, f"if {_MANUAL_MARK}:")
        verdict = _first_index(src, "_placement_verdict_partial = bool(_detected_gpus) and bool(use_fit)")
        assert hoist < verdict, (
            "the partial-placement verdict must read the Manual override, not the "
            "pre-hoist default it shadows"
        )


class TestManualPlusAutoLayersKeepsTheFitter:
    def test_only_the_fixed_count_branch_owns_the_verdict(self):
        """Manual + Auto layers hands memory to llama.cpp's fitter: that branch
        must keep its discard-and-fit shape and not reuse the placement mark."""
        src = _load_model_source()
        auto = src.find('if gpu_memory_mode == "manual" and gpu_layers < 0:')
        assert auto != -1, "Manual + Auto layers branch is gone"
        end = src.find('\n', auto)
        assert "_manual_placement" not in src[auto:end], (
            "the Auto-layers sibling must stay a fit launch, not a placement one"
        )
