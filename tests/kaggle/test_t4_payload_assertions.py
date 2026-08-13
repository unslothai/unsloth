# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only tests for what the T4 payloads actually ASSERT.

`test_t4_smoke_harness.py` covers the launcher, the gate and the shape of the
generated notebook. This file covers the other half: given a result dict, does
the payload call it a pass or a failure? Every one of these is a case where a
run that measured nothing, or measured the wrong thing, used to report green.

Nothing here needs a GPU, which is the whole point -- the pass/fail rule for a
leg that costs a Kaggle session has to be checkable without one.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

NAN = float("nan")
INF = float("inf")


# ------------------------------------------------------- determinism.py


def test_a_field_logged_by_only_one_run_is_a_difference():
    """grad_norm in cycle 0 and not in cycle 1 is nondeterminism, not a skip.

    ``check_reference`` already calls one-sided presence "a change in the
    SHAPE of what the trainer logged"; the exact comparator has to agree, or
    the strong assertion is weaker than the tolerance band beside it.
    """
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": 2.0}]
    b = [{"step": 1, "loss": 1.0}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["first_diff_step"] == 1


def test_a_field_absent_from_both_runs_is_still_not_a_difference():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0}]
    b = [{"step": 1, "loss": 1.0}]
    assert compare_metrics(a, b)["identical"] is True


def test_repeated_runs_must_agree_on_their_step_coordinates():
    """Same values in the same positions, against different step numbers."""
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}, {"step": 3, "loss": 0.25}]
    b = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}, {"step": 2, "loss": 0.25}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["step_mismatch"] == [{"index": 2, "a": 3, "b": 2}]


def test_matching_step_coordinates_are_not_reported_as_a_difference():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": NAN}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": NAN}]
    result = compare_metrics(a, b)
    assert result["identical"] is True
    assert result["step_mismatch"] == []


# -------------------------------------------------- run_t4_smoke.py: canary


def test_the_canary_must_be_the_whole_answer():
    from run_t4_smoke import CANARY, canary_failures

    run = {"run_index": 0, "generated": CANARY + "<|im_start|>and on it went"}
    failures = canary_failures(run, require = True)
    assert failures and "exactly" in failures[0]


def test_the_canary_tolerates_surrounding_whitespace_only():
    from run_t4_smoke import CANARY, canary_failures
    assert canary_failures({"run_index": 0, "generated": CANARY}, require = True) == []
    assert canary_failures({"run_index": 0, "generated": "\n" + CANARY + " \n"}, require = True) == []


def test_a_missing_canary_is_still_a_failure():
    from run_t4_smoke import canary_failures
    failures = canary_failures({"run_index": 1, "generated": "def my_function():"}, require = True)
    assert failures and "did not emit" in failures[0]


def test_the_canary_can_be_downgraded_to_a_warning():
    from run_t4_smoke import canary_failures
    assert canary_failures({"run_index": 1, "generated": "nope"}, require = False) == []


# -------------------------------------- run_t4_smoke.py: optimisation checks


def test_an_infinite_gradient_norm_is_not_an_applied_update():
    """fp16 overflow reports the norm as inf as readily as NaN.

    ``inf == inf``, so the NaN-only test counted every skipped step as an
    applied one and a run that trained nothing reported green.
    """
    from run_t4_smoke import optimisation_failures

    metrics = [{"step": s, "loss": 10.0 - s, "grad_norm": INF} for s in (1, 2, 3)]
    failures = optimisation_failures(metrics)
    assert failures and "no optimizer update was applied" in failures[0]


def test_one_finite_gradient_norm_is_enough():
    from run_t4_smoke import optimisation_failures
    metrics = [
        {"step": 1, "loss": 10.0, "grad_norm": NAN},
        {"step": 2, "loss": 5.0, "grad_norm": INF},
        {"step": 3, "loss": 1.0, "grad_norm": 11.2},
    ]
    assert optimisation_failures(metrics) == []


# ------------------------------------------- run_t4_smoke.py: saved adapter


def _adapter_state(**over) -> dict:
    state = {
        "dir": "/kaggle/working/smoke0/lora_run0",
        "files": ["adapter_config.json", "adapter_model.safetensors"],
        "weight_file": "adapter_model.safetensors",
        "config_readable": True,
        "tensors": 224,
        "parameters": 4325376,
        "non_finite_tensors": [],
        "nonzero_tensors": 224,
    }
    state.update(over)
    return state


def test_a_saved_adapter_that_reloads_with_weights_passes():
    from run_t4_smoke import saved_adapter_failures
    assert saved_adapter_failures(_adapter_state()) == []


def test_an_adapter_whose_weight_file_cannot_be_read_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(
        _adapter_state(tensors = None, error = "SafetensorError: header too small")
    )
    assert failures and "could not be read back" in failures[0]


def test_an_adapter_with_no_tensors_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(tensors = 0, parameters = 0))
    assert failures and "no tensors" in failures[0]


def test_an_adapter_with_non_finite_weights_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(non_finite_tensors = ["...lora_B.weight"]))
    assert failures and "non-finite" in failures[0]


def test_an_all_zero_adapter_is_a_failure():
    """lora_B starts at zero, so an all-zero file is an untrained adapter."""
    from run_t4_smoke import saved_adapter_failures

    failures = saved_adapter_failures(_adapter_state(nonzero_tensors = 0))
    assert failures and "every saved tensor is zero" in failures[0]


def test_a_missing_adapter_config_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(config_readable = False))
    assert failures and "adapter_config.json" in failures[0]


def test_the_adapter_check_reads_a_real_file_it_just_wrote(tmp_path):
    """End to end through the real writer, no GPU and no model."""
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import save_file

    from run_t4_smoke import saved_adapter_failures, verify_saved_adapter

    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(16, 8),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 16),
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    state = verify_saved_adapter(tmp_path)
    assert state["tensors"] == 2
    assert state["nonzero_tensors"] == 1
    assert state["config_readable"] is True
    assert saved_adapter_failures(state) == []

    save_file({"a.lora_B.weight": torch.zeros(4, 4)}, str(tmp_path / "adapter_model.safetensors"))
    failures = saved_adapter_failures(verify_saved_adapter(tmp_path))
    assert failures and "every saved tensor is zero" in failures[0]


# ------------------------------------------ run_t4_smoke.py: the reference


def _write_reference(
    path: Path,
    *,
    config: dict,
    model: str = "unsloth/Qwen2.5-0.5B-Instruct",
):
    path.write_text(
        json.dumps(
            {
                "config": config,
                "model": model,
                "environment": {},
                "metrics": [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)],
            }
        ),
        encoding = "utf-8",
    )


REFERENCE_CONFIG = {
    "max_steps": 3,
    "init_loss_scale": 0.0,
    "batch_size": 2,
    "grad_accum": 1,
    "max_seq_length": 512,
    "learning_rate": 0.001,
    "lora_r": 16,
    "lora_alpha": 32,
    "optim": "adamw_8bit",
    "gradient_checkpointing": "unsloth",
    "repeat": 2,
}


def test_a_reference_captured_with_another_learning_rate_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = dict(REFERENCE_CONFIG, learning_rate = 0.005),
    )
    assert verdict["status"] == "config_mismatch"
    assert verdict["config_differences"] == [
        {"key": "learning_rate", "reference": 0.001, "observed": 0.005}
    ]
    assert reference_failures(verdict, 0.10)


def test_a_reference_captured_for_another_model_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        model = "unsloth/Llama-3.2-1B-Instruct",
    )
    assert verdict["status"] == "config_mismatch"
    assert reference_failures(verdict, 0.10)


def test_the_repeat_count_does_not_invalidate_a_reference(tmp_path):
    """Each cycle is a fresh process, so how many were run is not the run."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed, ref, 0.10, 0.05, max_steps = 3, config = dict(REFERENCE_CONFIG, repeat = 5)
    )
    assert verdict["status"] == "ok"


def test_a_reference_that_predates_a_setting_does_not_refuse_on_it(tmp_path):
    """An older file simply does not carry the key; that is not a mismatch."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    older = {k: v for k, v in REFERENCE_CONFIG.items() if k != "gradient_checkpointing"}
    _write_reference(ref, config = older)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3, config = REFERENCE_CONFIG)
    assert verdict["status"] == "ok"
    assert verdict["config_unchecked"] == ["gradient_checkpointing"]


def test_the_reference_check_still_works_without_an_observed_config(tmp_path):
    """Backwards compatible: the step-count guard is what it always was."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    assert check_reference(observed, ref, 0.10, 0.05, max_steps = 3)["status"] == "ok"
    assert (
        check_reference(observed, ref, 0.10, 0.05, max_steps = 10)["status"] == "step_count_mismatch"
    )


def test_a_shifted_step_coordinate_is_refused_by_the_band_check(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": 1, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_mismatch"
    assert verdict["step_differences"] == [
        {"index": 1, "reference": 2, "observed": 1},
        {"index": 2, "reference": 3, "observed": 1},
    ]
    assert reference_failures(verdict, 0.10)


def test_the_model_revision_travels_with_the_reference(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    payload = json.loads(ref.read_text())
    payload["resolved_checkpoint"] = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"
    payload["resolved_revision"] = "10413c288cb9629acdf60b3e0229f3ba75efe413"
    ref.write_text(json.dumps(payload))
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]

    same = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        resolved_checkpoint = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        resolved_revision = "10413c288cb9629acdf60b3e0229f3ba75efe413",
    )
    assert same["status"] == "ok"

    moved = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        resolved_checkpoint = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        resolved_revision = "0000000000000000000000000000000000000000",
    )
    assert moved["status"] == "config_mismatch"
    assert reference_failures(moved, 0.10)


def test_a_reference_with_no_recorded_revision_does_not_refuse(tmp_path):
    """The committed file predates this; unknown is unknown, not a mismatch."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3, resolved_revision = "abc123")
    assert verdict["status"] == "ok"
    assert "resolved_revision" in verdict["config_unchecked"]


# -------------------------------------------- run_t4_smoke.py: main() paths


def _cycle(index: int, losses: list[float]) -> dict:
    return {
        "run_index": index,
        "metrics": [
            {"step": s, "loss": losses[s - 1], "grad_norm": 3.0} for s in range(1, len(losses) + 1)
        ],
        "generated": "__UNSLOTH__!!!",
        "canary_found": True,
        "adapter_files": ["adapter_config.json", "adapter_model.safetensors"],
        "saved_adapter": _adapter_state(),
        "determinism": {},
        "loss_scale": {},
        "timing_seconds": {},
        "peak_reserved_gb": 0.1,
    }


def _drive_main(monkeypatch, tmp_path, cycles: dict, argv: list[str]) -> tuple[int, dict]:
    import run_t4_smoke

    real_run = run_t4_smoke.subprocess.run

    def fake_run(cmd, *a, **kw):
        if "--cycle" not in cmd:
            return real_run(cmd, *a, **kw)
        index = int(cmd[cmd.index("--cycle") + 1])
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        outdir.mkdir(parents = True, exist_ok = True)
        payload = cycles.get(index)
        if payload is None:
            return types.SimpleNamespace(returncode = 1)
        (outdir / "cycle_report.json").write_text(json.dumps(payload), encoding = "utf-8")
        return types.SimpleNamespace(returncode = 0)

    monkeypatch.setattr(run_t4_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_t4_smoke.py", "--outdir", str(tmp_path)] + argv)
    code = run_t4_smoke.main()
    return code, json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))


def test_every_requested_repeat_is_compared_against_the_baseline(monkeypatch, tmp_path):
    """--repeat 3 compared cycles 0 and 1 and threw cycle 2 away."""
    cycles = {
        0: _cycle(0, [3.0, 2.0, 1.0]),
        1: _cycle(1, [3.0, 2.0, 1.0]),
        2: _cycle(2, [3.0, 2.0, 99.0]),
    }
    code, report = _drive_main(monkeypatch, tmp_path, cycles, ["--repeat", "3", "--max-steps", "3"])
    assert code == 1
    assert report["passed"] is False
    assert any("cycle 2" in f for f in report["failures"])
    repro = report["reproducibility"]
    assert repro["compared_cycles"] == ["1", "2"]
    assert repro["cycles"]["1"]["identical"] is True
    assert repro["cycles"]["2"]["identical"] is False
    # The keys report.py renders off this dict, unchanged in shape.
    assert repro["identical"] is False
    assert repro["first_diff_step"] == 3
    assert repro["max_abs_diff"]["loss"] == 98.0


def test_two_agreeing_repeats_still_pass(monkeypatch, tmp_path):
    cycles = {0: _cycle(0, [3.0, 2.0, 1.0]), 1: _cycle(1, [3.0, 2.0, 1.0])}
    code, report = _drive_main(monkeypatch, tmp_path, cycles, ["--repeat", "2", "--max-steps", "3"])
    assert code == 0
    assert report["passed"] is True
    assert report["reproducibility"]["identical"] is True
    assert report["reproducibility"]["cycles"]["1"]["identical"] is True


def test_the_summary_renderer_still_reads_a_multi_repeat_report(monkeypatch, tmp_path):
    """report.py renders `identical` / `first_diff_step` / `max_abs_diff`
    straight off `reproducibility`; a shape change there is a silent
    "**DIFFERED**" on every healthy run."""
    from report import render

    cycles = {0: _cycle(0, [3.0, 2.0, 1.0]), 1: _cycle(1, [3.0, 2.0, 1.0])}
    _, report_dict = _drive_main(
        monkeypatch, tmp_path, cycles, ["--repeat", "2", "--max-steps", "3"]
    )
    rendered = "\n".join(render(report_dict))
    assert "agreed **bitwise**" in rendered
    assert "DIFFERED" not in rendered


def test_a_failed_cycle_still_reports_its_environment(monkeypatch, tmp_path):
    """The one case where knowing which library set died matters most."""
    code, report = _drive_main(monkeypatch, tmp_path, {}, ["--repeat", "2", "--max-steps", "3"])
    assert code == 1
    assert report["passed"] is False
    assert report["environment"]["resolved"]
    assert report["config"]["max_steps"] == 3
    assert report["label"] == "t4-smoke"


# ------------------------------------------------------- run_gptoss_t4.py


def _gptoss_args(**over):
    base = {"max_steps": 3, "require_compile": True}
    base.update(over)
    return types.SimpleNamespace(**base)


def _gptoss_result(**over) -> dict:
    result = {
        "metrics": [{"step": s, "loss": 3.0 - s, "grad_norm": 1.5} for s in (1, 2, 3)],
        "compile": {"available": True, "unique_graphs": 32, "unique_graphs_delta": 30},
        "generated": "hello there",
        "precision": {
            "fp16": False,
            "bf16": False,
            "force_float32_env": "1",
            "custom_dtype_env": "down_projs;mlp.router",
        },
        "environment": {"bf16_supported": False, "gpu_name": "Tesla T4"},
    }
    result.update(over)
    return result


def test_a_healthy_gptoss_run_passes():
    from run_gptoss_t4 import failures_for
    assert failures_for(_gptoss_result(), _gptoss_args()) == []


def test_gptoss_fails_when_the_forced_float32_path_stopped_firing():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        precision = {
            "fp16": True,
            "bf16": False,
            "force_float32_env": None,
            "custom_dtype_env": None,
        }
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("float32" in f for f in failures)


def test_gptoss_does_not_demand_float32_on_a_card_with_bf16():
    """The force path exists for cards without bf16; elsewhere it is right
    that it does not fire, and a red there would be the harness's own bug."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        environment = {"bf16_supported": True, "gpu_name": "NVIDIA A100"},
        precision = {
            "fp16": False,
            "bf16": True,
            "force_float32_env": None,
            "custom_dtype_env": None,
        },
    )
    assert failures_for(result, _gptoss_args()) == []


def test_gptoss_fails_when_precision_was_never_recorded():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result()
    result.pop("precision")
    failures = failures_for(result, _gptoss_args())
    assert failures and any("precision" in f for f in failures)


def test_gptoss_measures_compilation_across_training_only():
    """Loading compiled 32 graphs and training compiled none."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        compile = {"available": True, "unique_graphs": 32, "unique_graphs_delta": 0}
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("eager" in f for f in failures)


def test_gptoss_compile_counters_report_a_delta():
    from run_gptoss_t4 import compile_counters

    before = compile_counters()
    after = compile_counters(before = before)
    assert after["unique_graphs_delta"] == 0
    fake = compile_counters(before = {"available": True, "unique_graphs": 5, "calls_captured": 9})
    assert fake["unique_graphs_delta"] == fake["unique_graphs"] - 5


def test_gptoss_fails_when_no_optimizer_update_was_applied():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        metrics = [{"step": s, "loss": 3.0 - s, "grad_norm": 0.0} for s in (1, 2, 3)]
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("optimizer update" in f for f in failures)


def test_gptoss_does_not_infer_a_verdict_from_an_unlogged_grad_norm():
    """A trainer that stops logging grad_norm says nothing either way."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(metrics = [{"step": s, "loss": 3.0 - s} for s in (1, 2, 3)])
    assert failures_for(result, _gptoss_args()) == []


# --------------------------------------------------------- run_grpo_t4.py


def _grpo_args(**over):
    base = {"max_steps": 3}
    base.update(over)
    return types.SimpleNamespace(**base)


def _grpo_result(**over) -> dict:
    result = {
        "log_history": [
            {"step": s, "loss": 0.0, "reward": 0.5, "reward_std": 0.2} for s in (1, 2, 3)
        ],
        "metrics": [{"step": s, "loss": 0.0, "grad_norm": 1.25} for s in (1, 2, 3)],
        "completions": [["four score and seven"], ["a completion with 7 in it"]],
        "fast_generate": "some generated text",
        "fast_generate_lora": {"requested": True, "applied": True},
    }
    result.update(over)
    return result


def test_a_healthy_grpo_run_passes():
    from run_grpo_t4 import failures_for
    assert failures_for(_grpo_result(), _grpo_args()) == []


def test_grpo_fails_when_a_step_logged_no_reward():
    from run_grpo_t4 import failures_for

    history = [
        {"step": 1, "loss": 0.0, "reward": 0.5, "reward_std": 0.2},
        {"step": 2, "loss": 0.0},
        {"step": 3, "loss": 0.0, "reward": 0.5, "reward_std": 0.2},
    ]
    failures = failures_for(_grpo_result(log_history = history), _grpo_args())
    assert failures and any("every step" in f for f in failures)


def test_grpo_ignores_the_summary_entry_that_carries_no_loss():
    """`train()` appends a train_runtime row; it is not a training step."""
    from run_grpo_t4 import failures_for

    history = [{"step": s, "loss": 0.0, "reward": 0.5, "reward_std": 0.2} for s in (1, 2, 3)]
    history.append({"step": 3})
    assert failures_for(_grpo_result(log_history = history), _grpo_args()) == []


def test_grpo_fails_when_every_gradient_norm_is_nan():
    from run_grpo_t4 import failures_for

    result = _grpo_result(metrics = [{"step": s, "loss": 0.0, "grad_norm": NAN} for s in (1, 2, 3)])
    failures = failures_for(result, _grpo_args())
    assert failures and any("optimizer update" in f for f in failures)


def test_grpo_fails_when_the_final_generation_skipped_the_trained_adapter():
    from run_grpo_t4 import failures_for

    result = _grpo_result(
        fast_generate_lora = {"requested": True, "applied": False, "error": "AttributeError: x"}
    )
    failures = failures_for(result, _grpo_args())
    assert failures and any("adapter" in f for f in failures)


def test_grpo_records_the_engine_it_built_before_a_later_failure(monkeypatch, tmp_path):
    """Construction succeeded and training raised: the report must say so."""
    import run_grpo_t4

    def exploding_train(args, report = None):
        if report is not None:
            report["engine_built"] = True
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(run_grpo_t4, "train", exploding_train)
    monkeypatch.setattr(run_grpo_t4, "make_libcuda_linkable", lambda: {"needed": False})
    monkeypatch.setattr(run_grpo_t4, "vllm_facts", lambda: {"version": "0.11.2"})
    monkeypatch.setattr(sys, "argv", ["run_grpo_t4.py", "--outdir", str(tmp_path)])
    code = run_grpo_t4.main()
    report = json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))
    assert code == 1
    assert report["engine_built"] is True
    assert any("illegal memory access" in f for f in report["failures"])


def test_grpo_still_reports_an_engine_that_never_built(monkeypatch, tmp_path):
    import run_grpo_t4

    def exploding_train(args, report = None):
        raise RuntimeError("EngineCore failed to start")

    monkeypatch.setattr(run_grpo_t4, "train", exploding_train)
    monkeypatch.setattr(run_grpo_t4, "make_libcuda_linkable", lambda: {"needed": False})
    monkeypatch.setattr(run_grpo_t4, "vllm_facts", lambda: {"version": "0.11.2"})
    monkeypatch.setattr(sys, "argv", ["run_grpo_t4.py", "--outdir", str(tmp_path)])
    assert run_grpo_t4.main() == 1
    report = json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))
    assert report["engine_built"] is False


# ------------------------------------------------------- references/README


def test_the_recapture_recipe_selects_the_control_report_by_label():
    """`reports[0]` is whichever kernel slug sorted first, not the control."""
    readme = (SMOKE_DIR / "references" / "README.md").read_text(encoding = "utf-8")
    assert 'reports"][0]' not in readme.replace(" ", "")
    assert '"control"' in readme
    recipe = readme[readme.index("## Capturing one") :]
    assert "label" in recipe


def test_reports_are_not_ordered_control_first(tmp_path):
    """The fact behind the item above, driven through the real extractor."""
    from launch import extract_reports

    def notebook(label: str) -> str:
        line = "T4_SMOKE_REPORT " + json.dumps(
            {"label": label, "model": "m", "metrics": [], "environment": {}, "config": {}}
        )
        return json.dumps({"cells": [{"outputs": [{"text": line + "\n"}]}]})

    for slug, legs in (
        ("unsloth-t4-ci-9f0e1a2b", ("control", "canary")),
        ("unsloth-t4-ci-0a1b2c3d", ("gptoss", "frontier")),
    ):
        kernel = tmp_path / slug
        kernel.mkdir(parents = True)
        for leg in legs:
            (kernel / f"t4_{leg}_output.ipynb").write_text(notebook(leg), encoding = "utf-8")

    labels = [r["label"] for r in extract_reports(tmp_path)]
    assert labels[0] != "control"
    assert sorted(labels) == ["canary", "control", "frontier", "gptoss"]
