# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO log-probs must use the output head of the model being scored.

With full fine-tuning and beta > 0, TRL scores a separate ``ref_model``; pairing its hidden
states with the trained policy's head corrupts the reference log-probs and the KL term.
"""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _lm_head_assignment():
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    (function,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_get_per_token_logps_and_entropies"
    ]
    (assign,) = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "lm_head" for t in node.targets)
    ]
    return compile(textwrap.dedent(ast.get_source_segment(source, assign)), "<lm_head>", "exec")


def _causal_lm(seed):
    torch.manual_seed(seed)
    head = torch.nn.Linear(8, 16, bias = False)
    return SimpleNamespace(get_output_embeddings = lambda: head)


def test_reference_model_scored_with_its_own_head():
    policy, ref = _causal_lm(0), _causal_lm(1)
    namespace = {"self": SimpleNamespace(model = policy), "unwrapped_model": ref}
    exec(_lm_head_assignment(), namespace)
    assert namespace["lm_head"] is ref.get_output_embeddings().weight
