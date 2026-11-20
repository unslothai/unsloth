# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO old / ref logps are scored under torch.no_grad(), as TRL does, never inference_mode.

On a GPU the compiled packed policy forward under inference_mode is not bit-equal to the loss's
grad-mode forward, so with num_iterations > 1 the first PPO ratio was exp(~0.01 nats) per token
instead of exactly 1 (Qwen3-0.6B, L4). Under torch.no_grad() both passes agree bitwise."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as _packed  # noqa: E402

_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


class _ModeRecordingModel(_packed._Model):
    def forward(self, *args, **kwargs):
        self.modes = getattr(self, "modes", [])
        self.modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return super().forward(*args, **kwargs)


def _scoring_function():
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    factory = _packed._named_function(tree, "grpo_trainer__get_per_token_logps_and_entropies")
    return _packed._named_function(factory, "_get_per_token_logps_and_entropies")


def _call_name(node):
    parts = []
    func = node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def test_packed_scoring_forwards_never_run_under_inference_mode():
    # The real inference-mode manager: the block must not consult it at all.
    model = _ModeRecordingModel(hidden_states = True)
    namespace, model, _ids, _mlp = _packed._run_packed_block(
        model = model,
        _get_inference_mode_context_manager = lambda _model: torch.inference_mode(),
    )
    assert namespace["_pk_use"] is True
    assert namespace["_pk_ref"] is not None, "the per-row verify forwards never ran"
    assert len(model.modes) == 1 + namespace["total_rows"]
    assert all(not inference for inference, _ in model.modes), model.modes
    assert all(not grad for _, grad in model.modes), model.modes
    assert not torch.is_inference(namespace["_pk_result"])


def test_no_scoring_site_enters_inference_mode():
    inner = _scoring_function()
    calls = {_call_name(node) for node in ast.walk(inner) if isinstance(node, ast.Call)}
    assert "_get_inference_mode_context_manager" not in calls
    assert "torch.inference_mode" not in calls


def test_every_model_forward_sits_under_no_grad():
    inner = _scoring_function()
    forwards = 0
    for node in ast.walk(inner):
        if not isinstance(node, ast.With):
            continue
        managers = {
            _call_name(item.context_expr)
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
        }
        inside = {
            _call_name(call)
            for stmt in node.body
            for call in ast.walk(stmt)
            if isinstance(call, ast.Call)
        }
        if "unwrapped_model" in inside and "torch.amp.autocast" in managers | inside:
            if "torch.no_grad" in managers:
                forwards += 1
    # PrefixGrouper, packed, packed verify and the padded loop.
    assert forwards == 4, forwards


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
