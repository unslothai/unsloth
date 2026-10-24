# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fp16 stacks disagree, so the control asserts only: it ran, converged, same step count as arm."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

from naive_trl_compare import comparison_failures  # noqa: E402


def _trace(*losses):
    return {"metrics": [{"step": i + 1, "loss": v} for i, v in enumerate(losses)]}


def test_a_control_that_never_ran_is_a_failure_not_a_silence():
    """The finding this file exists for. A missing arm must not read as a pass:
    "no comparison" and "the comparison agreed" are opposite outcomes and only
    one of them is evidence."""
    assert comparison_failures(None, [{"loss": 1.0}])
    assert comparison_failures({"error": "OOM"}, [{"loss": 1.0}])
    assert "did not run" in comparison_failures({"error": "OOM"}, None)[0]


def test_a_control_that_loaded_and_trained_nothing_is_a_failure():
    broken = comparison_failures({"metrics": []}, [{"loss": 1.0}])
    assert len(broken) == 1 and "reported no steps" in broken[0]


def test_a_converging_control_passes():
    assert comparison_failures(_trace(9.0, 5.0, 2.0), [{"loss": 1.0}] * 3) == []


def test_a_flat_or_rising_control_fails():
    assert comparison_failures(_trace(2.0, 5.0, 9.0), [{"loss": 1.0}] * 3)
    assert comparison_failures(_trace(2.0, 2.0, 2.0), [{"loss": 1.0}] * 3)


def test_a_non_finite_loss_fails_and_short_circuits():
    broken = comparison_failures(_trace(9.0, float("nan"), 2.0), [{"loss": 1.0}] * 3)
    assert len(broken) == 1 and "non-finite" in broken[0]


def test_the_arms_must_have_run_the_same_number_of_steps():
    """A control that quietly ran fewer steps is printed beside a full unsloth
    trace as though the two were the same experiment."""
    broken = comparison_failures(_trace(9.0, 5.0, 2.0), [{"loss": 1.0}] * 10)
    assert broken and "different numbers of steps" in broken[0]


def test_the_arms_are_never_asserted_equal():
    """Mutation-proof against the obvious "improvement". Two wildly different
    converging traces must PASS, because asserting agreement is what would make
    this check red on every ordinary version bump."""
    assert (
        comparison_failures(
            _trace(10.3222, 6.0, 1.0), [{"loss": 6.4367}, {"loss": 3.0}, {"loss": 0.5}]
        )
        == []
    )


def test_the_control_module_never_imports_unsloth():
    """The control must never import unsloth, which patches transformers, trl and peft underneath it."""
    tree = ast.parse((PAYLOAD / "naive_trl_compare.py").read_text(encoding = "utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "unsloth" not in imported, sorted(imported)
    assert "unsloth_zoo" not in imported, sorted(imported)


def test_the_payload_runs_the_control_in_a_separate_process():
    """The control runs in a separate process after the cycles, so two 4bit models never share the T4."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert "naive_trl_compare.py" in src
    assert "if args.compare_naive_trl:" in src
    cycles_at = src.index("runs.append(json.loads(report_file.read_text")
    spawn_at = src.index('"naive_trl_compare.py"')
    assert cycles_at < spawn_at, "the control arm must be spawned after the cycles"


def test_the_control_arm_loads_the_repo_unsloth_resolved():
    """Both arms load the repo unsloth resolved, since the plain path quantises the original and OOMs."""
    src = (ROOT / "tests" / "kaggle" / "t4_smoke" / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'control_model = runs[0].get("resolved_checkpoint") or args.model' in src
    assert '("--model", control_model),' in src


def test_the_control_arm_uses_gradient_checkpointing():
    """The control needs gradient checkpointing too, or it is measured with the biggest memory lever off."""
    src = (PAYLOAD / "naive_trl_compare.py").read_text(encoding = "utf-8")
    assert "use_gradient_checkpointing = True" in src
    assert "gradient_checkpointing = True," in src
    # Non-reentrant, or a PEFT model's inputs carry no grad and backward fails.
    assert 'gradient_checkpointing_kwargs = {"use_reentrant": False}' in src


def test_a_load_time_oom_can_be_reported_rather_than_failed():
    """Load-time OOM on gemma-4-E2B-it is a card-and-checkpoint fact, so it is reported, not failed."""
    oom = {"error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 8.75 GiB"}
    assert comparison_failures(oom, [{"loss": 1.0}], allow_oom = True) == []


def test_an_oom_is_still_a_failure_when_the_leg_did_not_opt_in():
    oom = {"error": "OutOfMemoryError: CUDA out of memory"}
    assert comparison_failures(oom, [{"loss": 1.0}])


def test_an_oom_after_training_started_is_still_a_failure():
    """The narrowness is the point. An OOM DURING training is a finding about
    the run; only a failure to load is a fact about the card."""
    oom = {
        "error": "OutOfMemoryError: CUDA out of memory",
        "metrics": [{"step": 1, "loss": 3.0}],
    }
    assert comparison_failures(oom, [{"loss": 1.0}], allow_oom = True)


def test_a_non_oom_crash_is_never_excused():
    crash = {"error": "ImportError: no module named trl"}
    assert comparison_failures(crash, [{"loss": 1.0}], allow_oom = True)
