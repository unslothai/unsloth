# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Sharded FSDP2 forwards all-gather, so GRPO ranks must run identical forwards (else 2xT4 hangs)."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
SOURCE = SOURCE_PATH.read_text(encoding = "utf-8")
HELPER = "_unsloth_grpo_refuse_rank_local_forwards"
FLAGS = (
    "_unsloth_seq_packing_nograd_ok",
    "_unsloth_seq_packing_grad_ok",
    "_unsloth_prefix_grouper_nograd_disabled",
)


def _helper():
    tree = ast.parse(SOURCE)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == HELPER)
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace[HELPER]


def _zoo(monkeypatch, collective):
    module = types.ModuleType("unsloth_zoo.rl_replacements")
    if collective is not None:
        module.grpo_forwards_are_collective = lambda model: collective
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", module)


def _trainer():
    return SimpleNamespace(
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m)
    )


def test_collective_forwards_turn_packing_and_prefix_grouper_off(monkeypatch):
    _zoo(monkeypatch, True)
    model = torch.nn.Linear(2, 2)
    assert _helper()(_trainer(), model) is True
    assert (model._unsloth_seq_packing_nograd_ok, model._unsloth_seq_packing_grad_ok) == (
        False,
        False,
    )
    assert model._unsloth_prefix_grouper_nograd_disabled is True


@pytest.mark.parametrize("collective", [False, None])
def test_other_layouts_and_an_older_zoo_keep_them(monkeypatch, collective):
    _zoo(monkeypatch, collective)
    model = torch.nn.Linear(2, 2)
    assert _helper()(_trainer(), model) is False
    assert not any(flag in model.__dict__ for flag in FLAGS)


def _scoring_source():
    outer = next(
        n
        for n in ast.parse(SOURCE).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    inner = next(
        n
        for n in ast.walk(outer)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_per_token_logps_and_entropies"
    )
    return ast.get_source_segment(SOURCE, inner)


def test_scoring_refuses_before_its_first_forward_and_agrees_on_the_mini_batch():
    source = _scoring_source()
    refuse = source.index(f"_grpo_collective_forwards = {HELPER}(self, model)")
    assert refuse < source.index("unwrapped_model(")
    tuned = source.index("B = max(1, -(-total_rows // B))")
    agree = source.index("B = grpo_agree_mini_batch(B, input_ids.device)")
    assert refuse < tuned < agree < source.index("B = self.args.unsloth_grpo_mini_batch")
    assert "if _grpo_collective_forwards:" in source[tuned:agree]


def test_the_loss_refuses_too_and_the_helper_reaches_the_generated_trainer():
    # Under vLLM / beta = 0 the scoring pass is skipped, but the zoo loss still packs.
    loss = SOURCE[SOURCE.index("def grpo_trainer_compute_loss") :]
    assert f"        {HELPER}(self, model)\n" in loss
    assert f'RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource({HELPER}))' in SOURCE
