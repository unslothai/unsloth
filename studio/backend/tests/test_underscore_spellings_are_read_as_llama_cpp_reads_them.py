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

    def test_cpu_ffn_layers_contradict_as_cpu_experts_do(self):
        # The dense twin of --n-cpu-moe: the first N layers' FFN weights run on CPU kernels.
        for args in (["--n-cpu-ffn", "4"], ["-ncffn", "4"], ["--n_cpu_ffn=4"]):
            assert exact.contradicting_args(args), args
        assert not exact.contradicting_args(["--n-cpu-ffn", "0"])
        assert not exact.contradicting_args(["-ncffn", "4", "--n-cpu-ffn", "0"])
        assert exact.contradicting_env({"LLAMA_ARG_N_CPU_FFN": "8"}) == ["LLAMA_ARG_N_CPU_FFN=8"]
        assert exact.contradicting_env({"LLAMA_ARG_N_CPU_FFN": "0"}) == []


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


def test_a_launch_line_count_overrides_the_inherited_ffn_variable():
    # llama.cpp reads the environment first and the argv after it: the line's zero wins.
    from core.inference import llama_exact as exact

    env = {"LLAMA_ARG_N_CPU_FFN": "4"}
    assert exact.contradicting_env(env) == ["LLAMA_ARG_N_CPU_FFN=4"]
    for args in (["--n-cpu-ffn", "0"], ["-ncffn", "0"], ["--n_cpu_ffn=0"]):
        assert exact.contradicting_env(env, args) == [], args
        assert exact.contradicting_args(args) == [], args
    # The expert count accumulates instead, so its variable still counts beside a line zero.
    assert exact.contradicting_env({"LLAMA_ARG_N_CPU_MOE": "4"}, ["--n-cpu-moe", "0"])
