# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in ScaleRL loss weights for GRPO: zero-variance group exclusion and prompt-level aggregation.

`_unsloth_grpo_sample_weight` runs inside TRL's `_generate_and_score_completions`, where the
group layout still exists, and returns per-completion weights normalised over the whole
generation batch. unsloth_zoo's `grpo_compute_loss` sums `loss * weight` per micro-batch with the
dapo process / accumulation rescale. Summed over micro-batches and averaged over processes (DDP)
this must equal the ScaleRL objective (arXiv 2510.13786):
    (1 / kept groups) * sum_g (1 / tokens in g) * sum_{i in g, t} l_it         (prompt level)
    (1 / tokens in kept groups) * sum_{i in kept groups, t} l_it               (exclusion only)
The helpers are lifted from the shipped source with `ast`, so nothing imports unsloth.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_unsloth_grpo_sample_weight_mode", "_unsloth_grpo_sample_weight")
    ]
    assert len(nodes) == 2
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), ns)
    return ns["_unsloth_grpo_sample_weight"]


sample_weight = _load()
G = 4


class _Acc:
    def __init__(self, everything):
        self.everything = everything

    def gather(self, tensor):
        return self.everything


def _trainer(loss_type, lengths_all):
    return SimpleNamespace(
        args = SimpleNamespace(loss_type = loss_type),
        num_generations = G,
        reward_weights = torch.tensor([1.0, 0.5]),
        accelerator = _Acc(lengths_all),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
    )


def _batch(seed, groups = 4, cols = 7):
    gen = torch.Generator().manual_seed(seed)
    rows = groups * G
    rpf = torch.randint(0, 3, (rows, 2), generator = gen).float()
    rpf[0:G] = torch.tensor([1.0, 2.0])  # group 0: zero variance
    rpf[2 * G:3 * G, 0] = 0.0
    rpf[2 * G:3 * G, 1] = torch.nan  # group 2: one reward func never applies, still zero variance
    rpf[3 * G, :] = torch.nan  # an unscorable completion in a varied group
    mask = (torch.rand(rows, cols, generator = gen) > 0.3).long()
    mask[5] = 0
    new = (-torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 2).requires_grad_(True)
    old = new.detach() + torch.randn(rows, cols, generator = gen, dtype = torch.float64) * 0.1
    adv = torch.randn(rows, generator = gen, dtype = torch.float64)
    return rpf, mask, new, old, adv


def _per_token(new, old, adv):
    coef = (new - old).exp()
    a = adv.unsqueeze(1)
    return -torch.min(coef * a, coef.clamp(0.8, 1.28) * a)


def _oracle(rpf, mask, new, old, adv, exclude, prompt):
    rewards = (rpf * torch.tensor([1.0, 0.5])).nansum(1)
    rewards[torch.isnan(rpf).all(1)] = torch.nan
    groups = rewards.view(-1, G)
    keep = []
    for g in groups:
        g = g[~torch.isnan(g)]
        keep.append(not exclude or len(g) < 2 or float(g.std()) > 1e-6)
    per_row = (_per_token(new, old, adv) * mask).sum(-1).view(-1, G)
    tokens = mask.sum(-1).double().view(-1, G).sum(-1)
    kept = [i for i, k in enumerate(keep) if k]
    if prompt:
        return sum(per_row[i].sum() / tokens[i].clamp(min = 1) for i in kept) / max(len(kept), 1)
    return sum(per_row[i].sum() for i in kept) / tokens[kept].sum().clamp(min = 1)


rr = pytest.importorskip("unsloth_zoo.rl_replacements")
HAS_ZOO = "sample_weight" in inspect.getsource(rr.grpo_compute_loss)


def _unsloth(rpf, mask, new, old, adv, loss_type, processes, spg, monkeypatch, exclude, prompt):
    monkeypatch.setenv("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "1" if exclude else "0")
    monkeypatch.setenv("UNSLOTH_GRPO_LOSS_AGGREGATION", "prompt" if prompt else "")
    rows = rpf.shape[0]
    local = rows // processes
    lengths_all = mask.sum(1).float()
    total = 0
    for p in range(processes):
        ps = slice(p * local, (p + 1) * local)
        trainer = _trainer(loss_type, lengths_all)
        w = sample_weight(trainer, "train", rpf, mask[ps], None, ps)
        assert w is not None and w.shape == (local,)
        # TRL shuffles the local batch, then splits it into steps_per_generation micro-batches.
        perm = torch.randperm(local, generator = torch.Generator().manual_seed(p))
        idx = torch.arange(rows)[ps][perm]
        micro = local // spg
        for m in range(spg):
            sel = idx[m * micro:(m + 1) * micro]
            loss = rr.grpo_compute_loss(
                None, new[sel], old[sel], None, torch.zeros_like(mask[sel]), mask[sel], 0.0, adv[sel],
                loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28,
                max_completion_length = mask.shape[1], num_items_in_batch = mask.sum(),
                current_gradient_accumulation_steps = spg, steps_per_generation = spg,
                num_processes = processes, importance_sampling_level = "token",
                sample_weight = w[perm][m * micro:(m + 1) * micro],
            )[0]
            total = total + loss
    return total / processes  # DDP averages the processes' gradients


@pytest.mark.skipif(not HAS_ZOO, reason = "unsloth_zoo without sample_weight support")
@pytest.mark.parametrize("exclude,prompt", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("processes,spg", [(1, 1), (1, 4), (2, 2), (4, 1)])
@pytest.mark.parametrize("seed", [0, 1])
def test_weights_reproduce_scalerl_objective(exclude, prompt, processes, spg, seed, monkeypatch):
    rpf, mask, new, old, adv = _batch(seed)
    got = _unsloth(rpf, mask, new, old, adv, "dapo", processes, spg, monkeypatch, exclude, prompt)
    want = _oracle(rpf, mask, new, old, adv, exclude, prompt)
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-9)
    g1, = torch.autograd.grad(got, new)
    g2, = torch.autograd.grad(want, new)
    torch.testing.assert_close(g1, g2, rtol = 1e-6, atol = 1e-9)


@pytest.mark.skipif(not HAS_ZOO, reason = "unsloth_zoo without sample_weight support")
def test_no_zero_variance_group_matches_dapo(monkeypatch):
    rpf, mask, new, old, adv = _batch(3)
    rpf[0:G, 0] = torch.arange(G).float()
    rpf[2 * G:3 * G, 0] = torch.arange(G).float()
    got = _unsloth(rpf, mask, new, old, adv, "dapo", 2, 2, monkeypatch, True, False)
    total = 0
    for sl in (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16)):
        total = total + rr.grpo_compute_loss(
            None, new[sl], old[sl], None, torch.zeros_like(mask[sl]), mask[sl], 0.0, adv[sl],
            loss_type = "dapo", epsilon_low = 0.2, epsilon_high = 0.28, num_items_in_batch = mask.sum(),
            current_gradient_accumulation_steps = 2, steps_per_generation = 2, num_processes = 2,
        )[0]
    torch.testing.assert_close(got, total / 2, rtol = 1e-6, atol = 1e-9)


def test_off_eval_and_unsupported_loss_types_return_none(monkeypatch):
    rpf, mask, *_ = _batch(0)
    lengths = mask.sum(1).float()
    monkeypatch.delenv("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_LOSS_AGGREGATION", raising = False)
    assert sample_weight(_trainer("dapo", lengths), "train", rpf, mask, None, slice(0, 16)) is None
    monkeypatch.setenv("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "1")
    assert sample_weight(_trainer("dapo", lengths), "eval", rpf, mask, None, slice(0, 16)) is None
    for loss_type, prompt in (("grpo", ""), ("dr_grpo", ""), ("sapo", ""), ("luspo", "prompt")):
        monkeypatch.setenv("UNSLOTH_GRPO_LOSS_AGGREGATION", prompt)
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            assert sample_weight(_trainer(loss_type, lengths), "train", rpf, mask, None, slice(0, 16)) is None
        assert caught
    monkeypatch.setenv("UNSLOTH_GRPO_LOSS_AGGREGATION", "sequence")
    with pytest.raises(ValueError):
        sample_weight(_trainer("dapo", lengths), "train", rpf, mask, None, slice(0, 16))


def test_tool_mask_and_metric(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_LOSS_AGGREGATION", "")
    rpf, mask, *_ = _batch(1)
    tool = (torch.arange(mask.shape[1]) % 2 == 0).long().expand_as(mask)
    trainer = _trainer("dapo", (mask * tool).sum(1).float())
    w = sample_weight(trainer, "train", rpf, mask, tool, slice(0, 16))
    kept = torch.tensor([0.0] * 4 + [1.0] * 4 + [0.0] * 4 + [1.0] * 4)
    torch.testing.assert_close(w, kept / ((mask * tool).sum(1).float() * kept).sum())
    assert trainer._metrics["train"]["frac_reward_zero_std_excluded"] == [0.5]


def test_trl_generation_anchors_present():
    trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
    # From the file: the imported method can be a runtime wrapper.
    text = Path(trl_grpo.__file__).read_text(encoding = "utf-8")
    node = next(
        n for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    src = ast.get_source_segment(text, node, padded = True)
    assert src.count('self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())') == 1
    assert src.count("\n        return output") == 1
    for name in ("rewards_per_func", "process_slice", "completion_mask"):
        assert name in src
