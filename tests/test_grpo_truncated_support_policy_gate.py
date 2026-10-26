# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 must only score over truncated sets while every scored policy is the sampler.

Each scoring call rebuilds its own top_k / top_p / min_p kept set from the logits it sees and always keeps the
scored token. That set is the sampler's only while the scored weights are the rollout weights. A reference model
(beta != 0) or a stale old policy (TRL computes old logps when gradient_accumulation_steps is not a multiple of
steps_per_generation * num_iterations) keeps a different set, so an excluded token gets a denominator that
depends on which token is scored: the implied probabilities no longer sum to 1 and the PPO ratio / KL are wrong.
Runs the shipped binding block of ``_get_per_token_logps_and_entropies`` on a stub trainer.
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
_SUPPORT = {"support_top_k": 2, "support_top_p": 1.0, "support_min_p": 0.0}


def _binds(node, name):
    return any(
        isinstance(target, ast.Name) and target.id == name
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        for target in n.targets
    )


def _reads_or_binds(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _sources():
    text = _SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text)
    helpers = [
        ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_unsloth_grpo_support_mismatch"
    ]
    (function,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_per_token_logps_and_entropies"
    ]
    for node, field in ((n, f) for n in ast.walk(function) for f in ("body", "orelse")):
        stmts = getattr(node, field, None)
        if not isinstance(stmts, list):
            continue
        # Anchored on the try that binds _support_kwargs: an earlier try in the same body also
        # imports grpo_truncated_support_kwargs (the UNSLOTH_GRPO_LOGPROB_RECOMPUTE scorer pick).
        starts = [
            i
            for i, s in enumerate(stmts)
            if isinstance(s, ast.Try) and _binds(s, "_support_kwargs")
        ]
        if not starts:
            continue
        block = []
        for s in stmts[starts[0] :]:
            if not _reads_or_binds(s, "_support_kwargs"):
                break
            block.append(textwrap.dedent(ast.get_source_segment(text, s, padded = True)))
        return "\n".join(helpers), "\n".join(block)
    raise AssertionError("truncated-support block not found")


def _bind(monkeypatch, trainer):
    helpers, block = _sources()
    zoo = types.ModuleType("unsloth_zoo")
    zoo_rl = types.ModuleType("unsloth_zoo.rl_replacements")
    zoo_rl.grpo_truncated_support_kwargs = lambda t: dict(_SUPPORT)
    zoo.rl_replacements = zoo_rl
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", zoo_rl)

    def chunked_hidden_states_selective_log_softmax(*args, **kwargs):
        return kwargs

    def chunked_selective_log_softmax(*args, **kwargs):
        return kwargs

    ns = {
        "self": trainer,
        "chunked_hidden_states_selective_log_softmax": chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": chunked_selective_log_softmax,
    }
    exec(compile(helpers, str(_SOURCE_PATH), "exec"), ns)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        exec(compile(block, str(_SOURCE_PATH), "exec"), ns)
    return ns, [str(w.message) for w in caught]


def _trainer(
    beta = 0.0,
    num_iterations = 1,
    steps_per_generation = 4,
    gradient_accumulation_steps = 4,
):
    return SimpleNamespace(
        beta = beta,
        num_iterations = num_iterations,
        use_vllm = True,
        args = SimpleNamespace(
            steps_per_generation = steps_per_generation,
            gradient_accumulation_steps = gradient_accumulation_steps,
        ),
    )


@pytest.mark.parametrize(
    "trainer",
    [
        _trainer(),
        _trainer(steps_per_generation = 2, gradient_accumulation_steps = 4),
        _trainer(steps_per_generation = None),
    ],
    ids = ["on_policy", "generation_divides_accumulation", "default_steps_per_generation"],
)
def test_on_policy_without_reference_scores_over_the_kept_set(monkeypatch, trainer):
    ns, messages = _bind(monkeypatch, trainer)
    assert ns["chunked_selective_log_softmax"]() == _SUPPORT
    assert ns["chunked_hidden_states_selective_log_softmax"]() == _SUPPORT
    assert getattr(trainer, "_unsloth_truncated_support_scored", False) is True
    assert not messages


@pytest.mark.parametrize(
    "trainer, reason",
    [
        (_trainer(beta = 0.04), "beta"),
        (_trainer(num_iterations = 2), "num_iterations"),
        (_trainer(steps_per_generation = 8, gradient_accumulation_steps = 4), "steps_per_generation"),
    ],
    ids = ["reference_model", "multiple_iterations", "buffered_generation"],
)
def test_policies_with_their_own_kept_set_score_the_full_vocabulary(monkeypatch, trainer, reason):
    ns, messages = _bind(monkeypatch, trainer)
    # Left unbound: the zoo loss then refuses the support kwargs too, since old / ref logps exist and
    # the trainer is not marked, so every policy is normalised over the full vocabulary.
    assert ns["chunked_selective_log_softmax"]() == {}
    assert ns["chunked_hidden_states_selective_log_softmax"]() == {}
    assert not getattr(trainer, "_unsloth_truncated_support_scored", False)
    assert len(messages) == 1 and reason in messages[0]
    # The zoo's own refusal ("needs an unsloth build ...") would be misleading here: already explained.
    assert trainer._unsloth_truncated_support_refused is True
    _, messages = _bind(monkeypatch, trainer)
    assert not messages


def test_refusals_do_not_claim_the_variable_was_set(monkeypatch):
    # zoo's automatic gate means refusals fire with the variable unset: never say "ignoring ...=1".
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    _, messages = _bind(monkeypatch, _trainer(beta = 0.04))
    refuse = _refuse_helper(monkeypatch, _SUPPORT)
    trainer = _trainer()
    trainer._unsloth_vllm_sync_skip = True
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        refuse(trainer)
    messages += [str(w.message) for w in caught]
    assert len(messages) == 2
    for message in messages:
        assert "ignoring" not in message
        assert message.startswith(
            "Unsloth: not scoring over the rollouts' top_k / top_p / min_p kept set"
        )
        assert "automatic" in message


def _own_support_logps(logits, index, top_k):
    """The per-policy rule: each policy's own top-k set, plus the scored token."""
    kth = torch.topk(logits, top_k, dim = -1).values[:, -1:]
    keep = logits >= kth
    keep = keep.scatter(-1, index.unsqueeze(-1), True)
    kept = logits.masked_fill(~keep, float("-inf"))
    return torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(kept, dim = -1)


def test_own_support_scoring_is_not_a_distribution_when_supports_differ():
    """Why the gate exists: with a policy whose top-2 set excludes a sampled token, scoring every token
    under the per-policy rule implies probabilities summing to more than 1."""
    new = torch.tensor([[0.0, 1.0, 2.0]])
    implied = torch.cat([_own_support_logps(new, torch.tensor([i]), 2) for i in range(3)]).exp()
    assert implied.sum().item() > 1.08
    # Same policy as the sampler: the scored token is always inside the set, so it is a distribution.
    sampler = torch.tensor([[2.0, 1.0, 0.0]])
    on_policy = torch.cat(
        [_own_support_logps(sampler, torch.tensor([i]), 2) for i in range(2)]
    ).exp()
    assert on_policy.sum().item() == pytest.approx(1.0)


def test_skipped_vllm_sync_rejects_truncated_support():
    # LoRA / quantized without fast_inference: the sampler keeps the initial weights.
    text = _SOURCE_PATH.read_text(encoding = "utf-8")
    (node,) = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_grpo_support_mismatch"
    ]
    ns = {}
    exec(compile(ast.get_source_segment(text, node), str(_SOURCE_PATH), "exec"), ns)
    _unsloth_grpo_support_mismatch = ns["_unsloth_grpo_support_mismatch"]
    trainer = types.SimpleNamespace(
        beta = 0.0,
        num_iterations = 1,
        args = types.SimpleNamespace(gradient_accumulation_steps = 1, steps_per_generation = 1),
        _unsloth_vllm_sync_skip = True,
    )
    assert any("sync" in reason for reason in _unsloth_grpo_support_mismatch(trainer))
    trainer._unsloth_vllm_sync_skip = False
    assert _unsloth_grpo_support_mismatch(trainer) == []


def _refuse_helper(monkeypatch, support):
    text = _SOURCE_PATH.read_text(encoding = "utf-8")
    nodes = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_unsloth_grpo_support_mismatch", "_unsloth_grpo_refuse_unscored_support")
    ]
    zoo = types.ModuleType("unsloth_zoo")
    zoo_rl = types.ModuleType("unsloth_zoo.rl_replacements")
    zoo_rl.grpo_truncated_support_kwargs = lambda t: dict(support)
    zoo.rl_replacements = zoo_rl
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", zoo_rl)
    ns = {}
    for node in nodes:
        exec(compile(ast.get_source_segment(text, node), str(_SOURCE_PATH), "exec"), ns)
    return ns["_unsloth_grpo_refuse_unscored_support"]


def test_efficient_path_refuses_the_kept_set_for_an_unsynced_sampler(monkeypatch):
    # compute_efficient (beta = 0, vLLM) never reaches the scoring gate: the zoo loss scores the
    # current policy, so the refusal has to be recorded on the trainer before returning.
    refuse = _refuse_helper(monkeypatch, _SUPPORT)
    trainer = _trainer()
    trainer._unsloth_vllm_sync_skip = True
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        refuse(trainer)
        refuse(trainer)
    assert trainer._unsloth_truncated_support_refused is True
    assert len(caught) == 1 and "sync" in str(caught[0].message)
    # Synced sampler: the kept set is the sampler's, nothing is refused.
    trainer = _trainer()
    refuse(trainer)
    assert not getattr(trainer, "_unsloth_truncated_support_refused", False)
    # Opt-in off (no support kwargs): nothing to refuse.
    refuse = _refuse_helper(monkeypatch, {})
    trainer = _trainer()
    trainer._unsloth_vllm_sync_skip = True
    refuse(trainer)
    assert not getattr(trainer, "_unsloth_truncated_support_refused", False)
