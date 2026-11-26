# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL <= 1.14 DPO token accuracy uses the half-logit argmax (ask zoo only if it accepts it); TRL 1.15 uses fp32."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RL_REPLACEMENTS = ROOT / "unsloth" / "models" / "rl_replacements.py"
RL = ROOT / "unsloth" / "models" / "rl.py"


def _stats_namespace(fake):
    tree = ast.parse(RL_REPLACEMENTS.read_text(encoding = "utf-8"))
    wanted = ("_unsloth_top1_half_kwargs", "_unsloth_dpo_loss_stats")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {"torch": torch, "_unsloth_fused_linear_logprobs": fake}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_REPLACEMENTS), "exec"), ns)
    assert "_unsloth_dpo_loss_stats" in ns
    return ns


def _fake(accepts):
    calls = []

    def run(hidden, weight, labels, **kw):
        calls.append(kw)
        n = labels.shape
        if kw.get("return_is_top1"):
            return torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.zeros(n, dtype = torch.bool)
        return torch.zeros(n)

    if accepts:

        def fused_linear_logprobs(
            hidden,
            weight,
            targets,
            *,
            is_top1_half_logits = False,
            **kw,
        ):
            return run(hidden, weight, targets, is_top1_half_logits = is_top1_half_logits, **kw)

    else:

        def fused_linear_logprobs(hidden, weight, targets, **kw):
            return run(hidden, weight, targets, **kw)

    return fused_linear_logprobs, calls


def _call(ns, ref):
    head = torch.nn.Linear(4, 8, bias = False)
    chunked = (None, head, (0.0, 0.0, 0.0))
    hidden = torch.zeros(2, 3, 4)
    labels = torch.zeros(2, 3, dtype = torch.long)
    ns["_unsloth_dpo_loss_stats"](
        object(), chunked, hidden, labels, torch.ones(2, 3, dtype = torch.bool), ref = ref
    )


def test_compute_loss_path_requests_half_logit_top1():
    fn, calls = _fake(accepts = True)
    ns = _stats_namespace(fn)
    _call(ns, ref = False)
    _call(ns, ref = True)
    assert calls[0].get("is_top1_half_logits") is True and calls[0]["return_is_top1"] is True
    assert "is_top1_half_logits" not in calls[1] or calls[1]["is_top1_half_logits"] is False


def test_compute_loss_path_older_zoo_keeps_working():
    fn, calls = _fake(accepts = False)
    ns = _stats_namespace(fn)
    _call(ns, ref = False)
    assert "is_top1_half_logits" not in calls[0]


def test_trl115_chunked_route_keeps_fp32_top1():
    tree = ast.parse(RL.read_text(encoding = "utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "fused_linear_logprobs"
        and any(k.arg == "return_is_top1" for k in n.keywords)
    ]
    assert calls, "rl.py no longer calls fused_linear_logprobs(return_is_top1 = ...)"
    for call in calls:
        assert all(k.arg != "is_top1_half_logits" for k in call.keywords)
        assert all(
            "is_top1_half_logits" not in ast.unparse(k.value)
            for k in call.keywords
            if k.arg is None
        )
