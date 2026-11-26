# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO: skip the old-policy forward under the vLLM IS correction when generation is aligned."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _helpers():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    names = (
        "_unsloth_grpo_old_logps_from_new",
        "_unsloth_grpo_is_metric_values",
    )
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), "a helper is missing"
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), ns)
    return ns


class _Zoo:
    def __init__(self, monkeypatch, supported):
        module = types.ModuleType("unsloth_zoo.rl_replacements")
        if supported:
            module.grpo_old_logps_from_new = lambda new: new.detach()
            module.grpo_policy_is_stochastic = (
                lambda model: any(
                    isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()
                )
                or float(getattr(getattr(model, "config", None), "attention_dropout", 0.0) or 0.0)
                > 0
            )
        monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", module)


def _trainer(
    ga = 4,
    spg = 4,
    num_iterations = 1,
    use_vllm = True,
    correction = True,
    model = None,
):
    return SimpleNamespace(
        args = SimpleNamespace(gradient_accumulation_steps = ga, steps_per_generation = spg),
        num_iterations = num_iterations,
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        model = model if model is not None else torch.nn.Linear(2, 2),
    )


@pytest.mark.parametrize(
    "ga, spg, num_iterations, aligned",
    [
        (4, 4, 1, True),  # TRL default: steps_per_generation = gradient_accumulation_steps
        (4, 2, 1, True),  # two generations per optimizer step, both on the same weights
        (4, 2, 2, True),  # both iterations inside one optimizer step
        (1, 1, 1, True),
        (4, 4, 2, False),  # second iteration after an optimizer step: pi_old is stale
        (2, 4, 1, False),  # one generation spans two optimizer steps
        (4, 3, 1, False),
        (1, 1, 2, False),
    ],
)
def test_skip_exactly_when_trl_calls_generation_aligned(
    monkeypatch, ga, spg, num_iterations, aligned
):
    monkeypatch.delenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", raising = False)
    _Zoo(monkeypatch, True)
    decide = _helpers()["_unsloth_grpo_old_logps_from_new"]
    trainer = _trainer(ga, spg, num_iterations)
    assert decide(trainer) is aligned
    # TRL's own test, verbatim: aligned exactly when it would leave old_per_token_logps None without IS.
    assert aligned == (ga % (spg * num_iterations) == 0)


@pytest.mark.parametrize(
    "case", ["no_vllm", "no_correction", "kill_switch", "old_zoo", "lora_dropout", "config_dropout"]
)
def test_keeps_the_forward_otherwise(monkeypatch, case):
    monkeypatch.setenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", "0" if case == "kill_switch" else "1")
    _Zoo(monkeypatch, case != "old_zoo")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    if case == "lora_dropout":
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.05))
    if case == "config_dropout":
        model.config = SimpleNamespace(attention_dropout = 0.1, classifier_dropout = 0.0)
    trainer = _trainer(
        use_vllm = case != "no_vllm",
        correction = case != "no_correction",
        model = model,
    )
    assert _helpers()["_unsloth_grpo_old_logps_from_new"](trainer) is False
    assert not hasattr(trainer, "_unsloth_old_logps_from_new_batches")


def test_zero_dropout_config_and_classifier_dropout_do_not_block(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", raising = False)
    _Zoo(monkeypatch, True)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0), torch.nn.Identity())
    model.config = SimpleNamespace(
        attention_dropout = 0.0,
        hidden_dropout = 0,
        classifier_dropout = 0.1,
        text_config = SimpleNamespace(attention_dropout = 0.0),
    )
    trainer = _trainer(model = model)
    decide = _helpers()["_unsloth_grpo_old_logps_from_new"]
    assert decide(trainer) is True and decide(trainer) is True
    assert trainer._unsloth_old_logps_from_new_batches == 2


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


def _patched(source = _FAKE_TRL):
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions
    return grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)


def _run_fake(
    monkeypatch,
    env = None,
    **trainer_kwargs,
):
    if env is None:
        monkeypatch.delenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", env)
    monkeypatch.delenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", raising = False)
    # Import the real modules before the stub zoo goes in, or they would bind to it.
    patched = _patched()
    _Zoo(monkeypatch, True)
    helpers = _helpers()
    ns = {
        "torch": torch,
        "_unsloth_grpo_rollout_logps_as_old": lambda *args: False,
        "_unsloth_grpo_old_logps_from_new": helpers["_unsloth_grpo_old_logps_from_new"],
        "_unsloth_grpo_old_logps_store": lambda trainer: False,
        "_unsloth_reject_grpo_image_list": lambda *args: None,
    }
    exec(textwrap.dedent(patched), ns)
    calls = []
    trainer = _trainer(**trainer_kwargs)
    trainer.score = lambda: calls.append(1) or torch.zeros(3, 5)
    out = ns["_generate_and_score_completions"](
        trainer, {"sampling": -torch.rand(3, 5), "mask": torch.ones(3, 5), "ids": torch.zeros(3, 5)}
    )
    return out, calls


def test_aligned_vllm_is_batch_skips_the_forward_and_is_marked(monkeypatch):
    out, calls = _run_fake(monkeypatch)
    assert calls == []
    assert "old_per_token_logps" not in out
    assert (
        out["unsloth_old_logps_from_new"].shape == (3,) and out["unsloth_old_logps_from_new"].all()
    )


@pytest.mark.parametrize(
    "env, kwargs, scored",
    [
        ("0", {}, True),  # kill switch: TRL's forward
        (None, dict(num_iterations = 2), True),  # misaligned: pi_old is stale, scored
        (None, dict(correction = False), False),  # IS off and aligned: TRL already skips
        (None, dict(correction = False, num_iterations = 2), True),
    ],
)
def test_other_batches_follow_trl(monkeypatch, env, kwargs, scored):
    out, calls = _run_fake(monkeypatch, env = env, **kwargs)
    assert bool(calls) is scored
    assert ("old_per_token_logps" in out) is scored
    assert "unsloth_old_logps_from_new" not in out


def test_installed_trl_gets_the_skip():
    spec = (
        importlib.util.find_spec("trl.trainer.grpo_trainer")
        if importlib.util.find_spec("trl")
        else None
    )
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    module_source = Path(spec.origin).read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(module_source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    source = "    " + ast.get_source_segment(module_source, node)
    if "vllm_importance_sampling_correction" not in source:
        pytest.skip("this TRL has no vLLM importance sampling correction")
    patched = _patched(source)
    ast.parse(textwrap.dedent(patched))
    skip = re.search(
        r"\n([ \t]*)_unsloth_old_from_new = \(not _unsloth_rollout_old\) and _unsloth_grpo_old_logps_from_new\(self\)\n"
        r"(?:\1#[^\n]*\n)*\1_unsloth_old_store = [^\n]*\n"
        r"\1if _unsloth_rollout_old or _unsloth_old_from_new or _unsloth_old_store:\n\1    old_per_token_logps = None\n"
        r"\1elif self\.args\.gradient_accumulation_steps % generate_every != 0",
        patched,
    )
    assert skip is not None
    assert patched.index("generate_every = ") < skip.start()
    assert patched.index("self._get_per_token_logps_and_entropies(", skip.end()) > skip.end()
    assert 'output["unsloth_old_logps_from_new"] = torch.ones(' in patched
    assert "if self.use_vllm and self.vllm_importance_sampling_correction:" not in patched


def test_compute_loss_forwards_the_marker():
    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    source = grpo_trainer_compute_loss("compute_loss", "def compute_loss(self):\n    pass\n")
    assert (
        '_old_logps_from_new = inputs.get("unsloth_old_logps_from_new", None) is not None' in source
    )
    assert '_grpo_accumulated_loss_kwargs["old_logps_from_new"] = True' in source
    assert "old_logps = per_token_logps.detach()" in source


def test_is_metrics_leave_out_unscored_rows():
    values = _helpers()["_unsloth_grpo_is_metric_values"]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    delta = torch.tensor([[0.1, 0.2, 0.0], [float("nan")] * 3, [0.3, 0.0, 0.0]])
    ratio = torch.tensor([[1.1, 0.9, 0.0], [float("nan")] * 3, [1.2, 0.0, 0.0]])
    d, r = values(delta, ratio, mask, False)
    torch.testing.assert_close(d, torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(r, torch.tensor([1.1, 0.9, 1.2]))
    d, r = values(delta, ratio, mask, True)
    torch.testing.assert_close(d, torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(r, torch.tensor([1.0, 1.2]))
    finite = torch.nan_to_num(delta)
    d, r = values(finite, torch.nan_to_num(ratio), mask, False)
    assert d.numel() == 6
