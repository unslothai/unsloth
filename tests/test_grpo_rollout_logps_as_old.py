# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1: the rollout engine's logprobs are pi_old (verl
rollout_correction bypass_mode, vime --use-rollout-logprobs), so the generation step skips the
old-policy forward and marks the batch for unsloth_zoo's grpo_accumulated_loss.

The decision runs once per generation batch, before TRL's old-logprob gate, and is the only place
the env var is read: the loss acts on the per-row marker, so a toggle mid-run cannot split a batch
between two policies."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _decision():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_grpo_rollout_logps_as_old"
    ]
    assert len(nodes) == 1, "_unsloth_grpo_rollout_logps_as_old is missing"
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), ns)
    return ns["_unsloth_grpo_rollout_logps_as_old"]


def _patched_generate_and_score():
    spec = (
        importlib.util.find_spec("trl.trainer.grpo_trainer")
        if importlib.util.find_spec("trl")
        else None
    )
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    module_source = Path(spec.origin).read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(module_source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    source = "    " + ast.get_source_segment(module_source, node)
    if "sampling_per_token_logps" not in source:
        pytest.skip("this TRL has no rollout logprobs in _generate_and_score_completions")
    return grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)


def test_generation_skips_the_old_forward_and_marks_the_batch():
    patched = _patched_generate_and_score()
    ast.parse(textwrap.dedent(patched))
    decision = patched.index("_unsloth_rollout_old = _unsloth_grpo_rollout_logps_as_old(")
    skip = re.search(
        r"\n([ \t]*)if _unsloth_rollout_old:\n\1    old_per_token_logps = None\n"
        r"\1elif self\.args\.gradient_accumulation_steps % generate_every != 0",
        patched,
    )
    assert skip is not None and skip.start() > decision
    assert patched.index("self._get_per_token_logps_and_entropies(", skip.end()) > skip.end()
    assert patched.index("sampling_per_token_logps", 0) < decision
    marker = patched.index('output["unsloth_rollout_logps_as_old"] = torch.ones(')
    assert marker > patched.index('output["old_per_token_logps"] = old_per_token_logps')


def test_trl_gate_is_untouched_when_the_flag_is_off():
    patched = _patched_generate_and_score()
    match = re.search(
        r"elif (self\.args\.gradient_accumulation_steps % generate_every != 0[^:]*):\n", patched
    )
    assert match is not None
    if "or (" in match.group(1):
        assert "self.use_vllm and self.vllm_importance_sampling_correction" in match.group(1)


class _Zoo:
    def __init__(self, monkeypatch, supported):
        module = types.ModuleType("unsloth_zoo.rl_replacements")
        if supported:
            module.grpo_rollout_logps_as_old = lambda sampling, old = None: (sampling, None)
        monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", module)


def _trainer(loss_type = "dapo"):
    return SimpleNamespace(args = SimpleNamespace(loss_type = loss_type))


def _batch():
    sampling = -torch.rand(3, 5, dtype = torch.float32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    return sampling, mask


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", raising = False)
    _Zoo(monkeypatch, True)
    trainer = _trainer()
    sampling, mask = _batch()
    assert _decision()(trainer, sampling, mask) is False
    assert not hasattr(trainer, "_unsloth_rollout_logps_as_old_batches")
    assert not hasattr(trainer, "_unsloth_rollout_logps_as_old_fallbacks")


def test_engages_and_counts(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "1")
    _Zoo(monkeypatch, True)
    trainer = _trainer()
    sampling, mask = _batch()
    sampling[0, 4] = float("nan")
    tool_mask = torch.ones_like(mask)
    tool_mask[1, 2] = 0
    sampling[1, 2] = float("nan")
    decide = _decision()
    assert decide(trainer, sampling, mask, tool_mask) is True
    assert decide(trainer, sampling[2:], mask[2:]) is True
    assert trainer._unsloth_rollout_logps_as_old_batches == 2


@pytest.mark.parametrize(
    "case",
    ["no_rollout_logps", "nan_on_a_loss_token", "vespo", "old_zoo", "no_loss_type"],
)
def test_falls_back_to_the_trainer_old_forward(monkeypatch, case):
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "1")
    _Zoo(monkeypatch, case != "old_zoo")
    trainer = _trainer("vespo" if case == "vespo" else "dapo")
    if case == "no_loss_type":
        trainer = SimpleNamespace(args = SimpleNamespace())
    sampling, mask = _batch()
    if case == "no_rollout_logps":
        sampling = None
    if case == "nan_on_a_loss_token":
        sampling[1, 3] = float("nan")
    decide = _decision()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert decide(trainer, sampling, mask) is False
        assert decide(trainer, sampling, mask) is False
    assert trainer._unsloth_rollout_logps_as_old_fallbacks == 2
    assert not hasattr(trainer, "_unsloth_rollout_logps_as_old_batches")
    assert sum("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD" in str(w.message) for w in caught) == 1


def test_ranks_agree_on_the_fallback(monkeypatch):
    # One rank with a non-finite rollout logprob must pull every rank onto the trainer's own pi_old:
    # under FSDP / ZeRO-3 that forward is collective.
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "1")
    _Zoo(monkeypatch, True)
    sampling, mask = _batch()
    other_rank_finite = {"value": 0}

    class _Acc:
        num_processes = 2

        def gather(self, t):
            return torch.cat([t, torch.tensor([other_rank_finite["value"]], dtype = t.dtype)])

    trainer = SimpleNamespace(args = SimpleNamespace(loss_type = "dapo"), accelerator = _Acc())
    decide = _decision()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert decide(trainer, sampling, mask) is False
    other_rank_finite["value"] = 1
    assert decide(trainer, sampling, mask) is True
    assert trainer._unsloth_rollout_logps_as_old_fallbacks == 1
    assert trainer._unsloth_rollout_logps_as_old_batches == 1


_FAKE_TRL = """
    def _generate_and_score_completions(self, inputs):
        sampling_per_token_logps = inputs["sampling"]
        completion_mask = inputs["mask"]
        completion_ids = inputs["ids"]
        generate_every = self.args.steps_per_generation * self.num_iterations
        with torch.no_grad():
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps = self.score()
            else:
                old_per_token_logps = None
        output = {"completion_ids": completion_ids}
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        return output
"""


def test_marked_batch_keeps_its_rollout_logprobs_on_older_trl():
    # TRL 0.23.x only puts sampling_per_token_logps into the output with the IS correction on:
    # a marked batch must carry them anyway, or the loss gets the marker without the tensor.
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", _FAKE_TRL
    )
    ns = {
        "torch": torch,
        "_unsloth_grpo_rollout_logps_as_old": lambda *args: True,
        "_unsloth_reject_grpo_image_list": lambda *args: None,  # other rewrites' prologue
    }
    exec(textwrap.dedent(patched), ns)
    sampling, mask = _batch()
    calls = []
    trainer = SimpleNamespace(
        args = SimpleNamespace(steps_per_generation = 1, gradient_accumulation_steps = 1),
        num_iterations = 2,
        use_vllm = False,
        vllm_importance_sampling_correction = False,
        score = lambda: calls.append(1),
    )
    out = ns["_generate_and_score_completions"](
        trainer, {"sampling": sampling, "mask": mask, "ids": torch.zeros(3, 5)}
    )
    assert not calls and "old_per_token_logps" not in out
    assert out["unsloth_rollout_logps_as_old"].all()
    assert out["sampling_per_token_logps"] is sampling


def test_missing_rollout_logprobs_still_join_the_rank_reduction(monkeypatch):
    # A None rank must still join the reduction, or the other ranks deadlock.
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "1")
    _Zoo(monkeypatch, True)
    sampling, mask = _batch()
    gathered = []

    class _Acc:
        num_processes = 2

        def __init__(self, other):
            self.other = other

        def gather(self, t):
            gathered.append(int(t.item()))
            return torch.cat([t, torch.tensor([self.other], dtype = t.dtype)])

    decide = _decision()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        no_logps = SimpleNamespace(args = SimpleNamespace(loss_type = "dapo"), accelerator = _Acc(1))
        assert decide(no_logps, None, mask) is False
        has_logps = SimpleNamespace(args = SimpleNamespace(loss_type = "dapo"), accelerator = _Acc(0))
        assert decide(has_logps, sampling, mask) is False
    assert gathered == [0, 1]
    assert not hasattr(has_logps, "_unsloth_rollout_logps_as_old_batches")
