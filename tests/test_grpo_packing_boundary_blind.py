# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO's packed forwards stay off for models that mix tokens across packed rows.

Sequence packing and PrefixGrouper score a batch as one flattened stream with per-row position_ids. DeepSeek V4's
compressors, Zaya's CCA and RecurrentGemma's RG-LRU ignore those, so SFT keeps padding-free off for them
(packing_boundary_blocker). Their leak is under the packed paths' 0.7 nat self-verify, so GRPO has to refuse them
up front as well."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
HELPER = "_unsloth_grpo_refuse_boundary_blind_packing"
FLAGS = (
    "_unsloth_seq_packing_nograd_ok",
    "_unsloth_seq_packing_grad_ok",
    "_unsloth_prefix_grouper_nograd_disabled",
    "_unsloth_prefix_grouper_disabled",
)


@pytest.fixture(autouse = True)
def packing_module(monkeypatch):
    # The helper imports unsloth.utils.packing; load that file alone (importing unsloth needs a GPU).
    import importlib.util
    import sys
    import types

    path = SOURCE_PATH.parents[1] / "utils" / "packing.py"
    spec = importlib.util.spec_from_file_location("unsloth.utils.packing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("unsloth", "unsloth.utils"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "unsloth.utils.packing", module)


def _helper():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == HELPER)
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace[HELPER]


def _blind_module(name):
    return type(name, (torch.nn.Module,), {})()


class _Model(torch.nn.Module):
    def __init__(
        self,
        inner,
        model_type = "llama",
    ):
        super().__init__()
        self.layer = inner
        self.config = SimpleNamespace(model_type = model_type)


def _trainer():
    return SimpleNamespace(
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m)
    )


@pytest.mark.parametrize(
    "model",
    [
        _Model(_blind_module("RecurrentGemmaRglru")),
        _Model(_blind_module("ZayaCCAProjection")),
        _Model(_blind_module("DeepseekV4CSACompressor")),
        _Model(torch.nn.Linear(2, 2), model_type = "deepseek_v4"),
    ],
)
def test_every_packed_path_is_refused_once(model):
    trainer = _trainer()
    _helper()(trainer, model)
    assert model.__dict__["_unsloth_seq_packing_nograd_ok"] is False
    assert model.__dict__["_unsloth_seq_packing_grad_ok"] is False
    assert model.__dict__["_unsloth_prefix_grouper_nograd_disabled"] is True
    assert model.__dict__["_unsloth_prefix_grouper_disabled"] is True
    # Decided once per trainer: a later call does not re-walk the model.
    model.__dict__["_unsloth_seq_packing_grad_ok"] = True
    _helper()(trainer, model)
    assert model.__dict__["_unsloth_seq_packing_grad_ok"] is True


def test_an_ordinary_model_keeps_packing():
    model = _Model(torch.nn.Linear(2, 2))
    _helper()(_trainer(), model)
    assert not any(flag in model.__dict__ for flag in FLAGS)


def test_the_generated_trainer_defines_and_calls_the_check():
    src = SOURCE_PATH.read_text(encoding = "utf-8")
    assert f'RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource({HELPER}))' in src
    tree = ast.parse(src)
    for outer in ("grpo_trainer__get_per_token_logps_and_entropies", "grpo_trainer_compute_loss"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == outer)
        assert f"{HELPER}(self, model)" in ast.unparse(node), outer
