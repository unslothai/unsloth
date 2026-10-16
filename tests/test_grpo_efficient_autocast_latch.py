# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""With no scoring forward (vLLM, beta = 0) the efficient call must still latch this trainer's precision."""

import ast
import os
import textwrap
import types
from pathlib import Path

import torch

REPL_SRC = (
    Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
).read_text(encoding = "utf-8")


def _function(name, parent = None):
    tree = ast.parse(REPL_SRC)
    scope = tree.body
    if parent is not None:
        scope = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == parent
        ).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(REPL_SRC, node))


def _namespace():
    ns = {"torch": torch, "os": os}
    exec(_function("_unsloth_grpo_autocast"), ns)
    exec(
        _function(
            "_get_per_token_logps_and_entropies", "grpo_trainer__get_per_token_logps_and_entropies"
        ),
        ns,
    )
    return ns


def test_efficient_call_latches_this_trainers_precision(monkeypatch):
    ns = _namespace()
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = None,
    )
    out = ns["_get_per_token_logps_and_entropies"](
        trainer, None, None, None, 4, compute_efficient = True
    )
    assert out == (None, None)
    assert trainer._autocast_enabled is True
    assert trainer._autocast_dtype is torch.bfloat16


def test_efficient_call_keeps_an_existing_latch(monkeypatch):
    ns = _namespace()
    trainer = types.SimpleNamespace(
        args = None, model = None, _autocast_enabled = False, _autocast_dtype = torch.float16
    )
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is False
    assert trainer._autocast_dtype is torch.float16
