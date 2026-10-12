# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server reads a long option's underscores as hyphens (`common/arg.cpp` replaces them
before the lookup), so `--cpu_moe` and `--preempt_ram` are the same options to it. Studio's own
readers have to agree, or exact mode is reported on beside a CPU placement and a caller's
parking budget is overridden or left switched on."""

from __future__ import annotations

from core.inference import llama_cpp as llama_mod
from core.inference import llama_exact as exact


class TestExactModeSeesTheUnderscoreSpelling:
    def test_cpu_placements_contradict_however_spelled(self):
        assert exact.contradicting_args(["--cpu_moe"])
        assert exact.contradicting_args(["--n_cpu_moe", "4"])
        assert exact.contradicting_args(["--override_tensor", "blk.*=CPU"])
        assert exact.contradicting_args(["--n-cpu-moe", "4"]) == exact.contradicting_args(
            ["--n_cpu_moe", "4"]
        )

    def test_a_default_spelling_is_still_not_a_contradiction(self):
        assert not exact.contradicting_args(["--n_cpu_moe", "0"])


class TestTheParkingBudgetIsReadHoweverSpelled:
    def test_a_named_budget(self):
        assert llama_mod._named_preempt_ram_mib(["--preempt_ram", "2048"], {}) == 2048
        assert llama_mod._named_preempt_ram_mib(["--preempt_ram=-1"], {}) == -1

    def test_a_named_zero_switches_parking_off(self):
        assert llama_mod._preempt_ram_disabled_in(["--preempt_ram", "0"], env = {}) is True
        assert llama_mod._preempt_ram_disabled_in(["--preempt_ram", "512"], env = {}) is False

    def test_studio_does_not_append_over_a_named_budget(self):
        assert (
            llama_mod._exact_parking_budget_mib(
                1024 * 1024 * 1024, args = ["--preempt_ram", "0"], env = {}
            )
            is None
        )

    def test_standing_down_zeroes_the_underscore_spelling(self, monkeypatch):
        monkeypatch.setattr(llama_mod, "_child_parking_stands_down", lambda supports: True)
        args = ["--preempt_ram", "4096"]
        env: dict = {}
        llama_mod._stand_down_child_parking(env, args, server_supports = True)
        assert args == ["--preempt_ram", "0"]
