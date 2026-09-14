# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The plain-TRL control arm, and the ways a "comparison" proves nothing.

The rules under test are deliberately weak, and that is the point. Two library
stacks do not produce one fp16 trajectory -- `frontier` measured transformers
5.5.0 and 5.15.1 disagreeing at step 1 on identical weights, data and seed --
so a guard that asserts the arms AGREE would be red on ordinary drift. These
assert only what a comparison is entitled to: the control ran, it converged,
and it ran the same number of steps as the arm it is printed beside.
"""

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
    """Asserted from the SOURCE, not from a convention. Anything that has
    imported unsloth has had transformers, trl and peft patched underneath it
    and is no longer a control; the comparison would be unsloth against itself
    with extra steps, and it would look exactly like a real result."""
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
    """A control imported into the parent would be patched by whatever the
    parent imported. It must be spawned, and it must be spawned AFTER the
    cycles: two 4bit models resident at once on a 14.56GB T4 is how a
    comparison becomes an OOM blamed on the thing being compared."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert "naive_trl_compare.py" in src
    assert "if args.compare_naive_trl:" in src
    cycles_at = src.index("runs.append(json.loads(report_file.read_text")
    spawn_at = src.index('"naive_trl_compare.py"')
    assert cycles_at < spawn_at, "the control arm must be spawned after the cycles"


def test_the_control_arm_loads_the_repo_unsloth_resolved():
    """Not the name that was asked for, and the difference is an OOM.

    `load_in_4bit=True` sends unsloth through FLOAT_TO_INT_MAPPER to a
    pre-quantised `-unsloth-bnb-4bit` sibling. The plain path quantises the
    ORIGINAL on the fly and has to materialise the 16bit checkpoint first. On
    gemma-4-E2B-it that asked for 8.75 GiB on top of 7.25 GiB already resident
    and died (kernel unsloth-probe-latestcompile-r3-cb1125).

    Pointing both arms at the same weights is also the fairer comparison: the
    question is what the two training stacks do, not which repo each loader
    picks.
    """
    src = (ROOT / "tests" / "kaggle" / "t4_smoke" / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'control_model = runs[0].get("resolved_checkpoint") or args.model' in src
    assert '("--model", control_model),' in src


def test_the_control_arm_uses_gradient_checkpointing():
    """Leaving it off was unfair rather than neutral.

    The unsloth arm runs with `gradient_checkpointing="unsloth"`, so a control
    without it is measured with the single largest memory lever disabled on one
    side only. On gemma-4-E2B-it that is the difference between a comparison and
    an OOM: the control asked for 8.75 GiB on top of 8.96 GiB already resident,
    on a 14.56 GiB card (kernel unsloth-probe-latestcompile-r4-e67ef2).
    """
    src = (PAYLOAD / "naive_trl_compare.py").read_text(encoding = "utf-8")
    assert "use_gradient_checkpointing = True" in src
    assert "gradient_checkpointing = True," in src
    # Non-reentrant, or a PEFT model's inputs carry no grad and the backward
    # fails with "element 0 of tensors does not require grad".
    assert 'gradient_checkpointing_kwargs = {"use_reentrant": False}' in src


def test_a_load_time_oom_can_be_reported_rather_than_failed():
    """Measured: on gemma-4-E2B-it the plain arm asks for 8.75 GiB with 8.96 GiB
    already resident on a 14.56 GiB T4, at LOAD -- `metrics` is absent, so no
    step ever ran, and enabling gradient checkpointing changed the number not at
    all. That is a statement about the card and the checkpoint, not about either
    training stack."""
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
