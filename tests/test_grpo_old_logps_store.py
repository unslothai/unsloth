# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO num_iterations > 1: pi_old comes from the first loss pass over each micro-batch, not a second
no-grad forward over the generation batch, and the later iterations reuse it."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
import textwrap
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _defs(names):
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), "a helper is missing"
    ns = {"torch": torch, "inspect": inspect}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), ns)
    return ns


def _gates():
    return _defs(("_unsloth_grpo_old_logps_from_new", "_unsloth_grpo_old_logps_store"))


def _zoo(
    monkeypatch,
    store = True,
    stochastic = False,
):
    module = types.ModuleType("unsloth_zoo.rl_replacements")
    module.grpo_old_logps_from_new = lambda new: new.detach()
    module.grpo_policy_is_stochastic = lambda model: stochastic
    if store:
        module.grpo_old_logps_store = lambda old, row_keep = None, full_rows = None: old.detach()
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", module)


def _trainer(
    ga = 4,
    spg = 4,
    num_iterations = 2,
    use_vllm = True,
    correction = True,
):
    return SimpleNamespace(
        args = SimpleNamespace(gradient_accumulation_steps = ga, steps_per_generation = spg),
        num_iterations = num_iterations,
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        model = torch.nn.Linear(2, 2),
    )


def _first_pass_inside_one_optimizer_step(
    ga,
    spg,
    num_iterations,
    generations = 6,
):
    """Brute force over TRL's schedule: every generation starts on an optimizer step and its first pass
    over every micro-batch ends before the next one, so that pass scores the rollout weights."""
    generate_every = spg * num_iterations
    for start in range(0, generations * generate_every, generate_every):
        if start % ga != 0:
            return False
        if (start + spg - 1) // ga != start // ga:
            return False
    return True


CASES = [
    (ga, spg, nit) for ga in (1, 2, 3, 4, 8) for spg in (1, 2, 3, 4, 8) for nit in (1, 2, 3, 4)
]


@pytest.mark.parametrize("ga, spg, num_iterations", CASES)
@pytest.mark.parametrize("vllm_is", [True, False])
def test_gate_is_exactly_the_aligned_multi_iteration_schedule(
    monkeypatch, ga, spg, num_iterations, vllm_is
):
    monkeypatch.delenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", raising = False)
    _zoo(monkeypatch)
    gates = _gates()
    trainer = _trainer(ga, spg, num_iterations, use_vllm = vllm_is, correction = vllm_is)
    store = gates["_unsloth_grpo_old_logps_store"](trainer)
    # TRL already leaves pi_old None when every iteration sits in one optimizer step.
    trl_skips = ga % (spg * num_iterations) == 0
    expected = (
        num_iterations > 1
        and not trl_skips
        and _first_pass_inside_one_optimizer_step(ga, spg, num_iterations)
    )
    assert store is expected
    # Never both: the single-iteration skip owns the aligned case.
    assert not (store and gates["_unsloth_grpo_old_logps_from_new"](trainer))


@pytest.mark.parametrize("case", ["kill_switch", "old_zoo", "stochastic"])
def test_gate_keeps_the_forward_otherwise(monkeypatch, case):
    monkeypatch.setenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", "0" if case == "kill_switch" else "1")
    _zoo(monkeypatch, store = case != "old_zoo", stochastic = case == "stochastic")
    trainer = _trainer()
    assert _gates()["_unsloth_grpo_old_logps_store"](trainer) is False
    assert not hasattr(trainer, "_unsloth_old_logps_store_batches")


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
    store = True,
    **trainer_kwargs,
):
    if env is None:
        monkeypatch.delenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_OLD_LOGPS_FROM_POLICY", env)
    monkeypatch.delenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", raising = False)
    patched = _patched()
    _zoo(monkeypatch, store = store)
    gates = _gates()
    ns = {
        "torch": torch,
        "_unsloth_grpo_rollout_logps_as_old": lambda *args: False,
        "_unsloth_reject_grpo_image_list": lambda *args: None,
        **gates,
    }
    exec(textwrap.dedent(patched), ns)
    calls = []
    trainer = _trainer(**trainer_kwargs)
    trainer.score = lambda: calls.append(1) or torch.zeros(3, 5)
    out = ns["_generate_and_score_completions"](
        trainer, {"sampling": -torch.rand(3, 5), "mask": torch.ones(3, 5), "ids": torch.zeros(3, 5)}
    )
    return out, calls


@pytest.mark.parametrize("vllm_is", [True, False])
def test_aligned_multi_iteration_batch_skips_the_forward_and_is_marked(monkeypatch, vllm_is):
    out, calls = _run_fake(monkeypatch, use_vllm = vllm_is, correction = vllm_is)
    assert calls == []
    assert "old_per_token_logps" not in out and "unsloth_old_logps_from_new" not in out
    assert out["unsloth_old_logps_store"].shape == (3,) and out["unsloth_old_logps_store"].all()


@pytest.mark.parametrize(
    "env, store, kwargs",
    [
        ("0", True, {}),  # kill switch
        (None, False, {}),  # zoo cannot hand the logps back
        (None, True, dict(ga = 2, spg = 4)),  # first pass spans an optimizer step
        (None, True, dict(ga = 4, spg = 2, num_iterations = 3)),  # generation starts mid step
    ],
)
def test_other_multi_iteration_batches_keep_trls_forward(monkeypatch, env, store, kwargs):
    out, calls = _run_fake(monkeypatch, env = env, store = store, **kwargs)
    assert calls == [1]
    assert "old_per_token_logps" in out and "unsloth_old_logps_store" not in out


def test_installed_trl_gets_the_store():
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
    if "generate_every" not in source:
        pytest.skip("this TRL has no steps_per_generation schedule")
    patched = _patched(source)
    ast.parse(textwrap.dedent(patched))
    assert re.search(
        r"_unsloth_old_store = \(not _unsloth_rollout_old\) and \(not _unsloth_old_from_new\) "
        r"and _unsloth_grpo_old_logps_store\(self\)\n[ \t]*if _unsloth_rollout_old or _unsloth_old_from_new "
        r"or _unsloth_old_store:\n[ \t]*old_per_token_logps = None\n",
        patched,
    )
    assert 'output["unsloth_old_logps_store"] = torch.ones(' in patched


LOSS_HELPERS = (
    "grpo_trainer_compute_loss",
    "_unsloth_grpo_accumulation_steps",
    "_unsloth_grpo_is_metric_values",
    "_unsloth_get_model_config",
    "_unsloth_text_configs",
    "_unsloth_get_final_logit_softcapping",
    "_unsloth_resolve_logit_scales",
    "_unsloth_grpo_nanmin",
    "_unsloth_grpo_nanmax",
    "_unsloth_grpo_flush_metrics",
    "_unsloth_grpo_defer_metric",
    "_unsloth_grpo_refuse_boundary_blind_packing",
    "_unsloth_grpo_refuse_rank_local_forwards",
)
B, P, C = 3, 2, 4


def _compute_loss(zoo_loss):
    ns = _defs(LOSS_HELPERS)
    source = ns["grpo_trainer_compute_loss"]("compute_loss", "")
    ns.update(
        grpo_accumulated_loss = zoo_loss,
        grpo_compute_loss_slow = None,
        detect_logit_transforms = None,
        _unsloth_grpo_vision_inputs = lambda source: {},
        _unsloth_fix_mm_token_type_ids = lambda *args, **kwargs: None,
        nanmin = torch.min,
        nanmax = torch.max,
    )
    exec(textwrap.dedent(source), ns)
    return ns["compute_loss"]


def _loss_trainer():
    return SimpleNamespace(
        args = SimpleNamespace(
            loss_type = "dapo",
            max_completion_length = C,
            delta = None,
            temperature = 1.0,
            unsloth_num_chunks = 1,
        ),
        accelerator = SimpleNamespace(num_processes = 1, gather = lambda tensor: tensor),
        beta = 0.0,
        epsilon_low = 0.2,
        epsilon_high = 0.2,
        loss_type = "dapo",
        use_vllm = False,
        model = SimpleNamespace(training = True, config = None),
        control = SimpleNamespace(should_evaluate = False),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
        _get_per_token_logps_and_entropies = lambda *args, **kwargs: (None, None),
        processing_class = None,
    )


def _buffered(marked = True):
    ids = torch.ones(B, P, dtype = torch.long)
    completion = torch.ones(B, C, dtype = torch.long)
    out = dict(
        prompt_ids = ids,
        prompt_mask = ids,
        completion_ids = completion,
        completion_mask = completion,
        advantages = torch.tensor([1.0, -1.0, 0.5]),
    )
    if marked:
        out["unsloth_old_logps_store"] = torch.ones(B, dtype = torch.bool)
    return out


class _Zoo:
    def __init__(self, stores = True):
        self.calls = []
        self.stores = stores

    def __call__(
        self,
        trainer = None,
        old_logps = None,
        **kwargs,
    ):
        new = -torch.rand(B, C, dtype = torch.float64)
        self.calls.append(dict(old = old_logps, new = new, kwargs = dict(kwargs)))
        if kwargs.get("store_old_logps") and self.stores:
            trainer._unsloth_grpo_stored_old_logps = new.detach()
        old = new.detach() if old_logps is None else old_logps
        coef_1 = torch.exp(new - old)
        ones = torch.ones(B, C)
        return (torch.tensor(1.0), torch.tensor(4.0), torch.tensor(0.0), None, None, coef_1, ones)


def test_first_pass_keeps_pi_old_and_later_passes_reuse_it():
    zoo = _Zoo()
    compute_loss = _compute_loss(zoo)
    trainer = _loss_trainer()
    inputs = _buffered()
    for _ in range(3):  # num_iterations = 3 over one buffered micro-batch
        compute_loss(trainer, trainer.model, inputs)
    first, second, third = zoo.calls
    assert first["old"] is None
    assert (
        first["kwargs"]["old_logps_from_new"] is True and first["kwargs"]["store_old_logps"] is True
    )
    # Iteration 1: pi_old is the scored policy itself, so the ratio is exactly 1.
    assert torch.equal(
        torch.exp(first["new"] - first["new"].detach()), torch.ones(B, C, dtype = torch.float64)
    )
    for later in (second, third):
        assert later["old"] is inputs["old_per_token_logps"]
        assert torch.equal(later["old"], first["new"])
        assert (
            "store_old_logps" not in later["kwargs"] and "old_logps_from_new" not in later["kwargs"]
        )
    assert not inputs["unsloth_old_logps_store"].any()
    assert trainer._unsloth_old_logps_stored == 1
    assert not hasattr(trainer, "_unsloth_grpo_stored_old_logps")


def test_unmarked_inputs_are_untouched():
    zoo = _Zoo()
    compute_loss = _compute_loss(zoo)
    trainer = _loss_trainer()
    inputs = _buffered(marked = False)
    inputs["old_per_token_logps"] = torch.zeros(B, C)
    compute_loss(trainer, trainer.model, inputs)
    (call,) = zoo.calls
    assert call["old"] is inputs["old_per_token_logps"]
    assert "store_old_logps" not in call["kwargs"] and "old_logps_from_new" not in call["kwargs"]
    assert not hasattr(trainer, "_unsloth_old_logps_stored")


def test_a_copied_batch_cannot_silently_reuse_ratio_one():
    compute_loss = _compute_loss(_Zoo())
    trainer = _loss_trainer()
    inputs = _buffered()
    compute_loss(trainer, trainer.model, dict(inputs))  # a shallow copy: the stored pi_old is lost
    with pytest.raises(RuntimeError, match = "lost the old-policy logprobs"):
        compute_loss(trainer, trainer.model, dict(inputs))


def test_a_zoo_that_does_not_store_is_refused():
    compute_loss = _compute_loss(_Zoo(stores = False))
    trainer = _loss_trainer()
    with pytest.raises(RuntimeError, match = "did not return the old-policy logprobs"):
        compute_loss(trainer, trainer.model, _buffered())
