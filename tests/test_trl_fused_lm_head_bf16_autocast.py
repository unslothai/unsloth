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

"""bfloat16 full finetuning keeps norms in float32 under a bfloat16-autocast forward; TRL >= 1.15's fused head
(DPO / GRPO, fused_lm_head = True) calls the backbone directly. The patched add_fused_lm_head must keep the autocast,
else the first float32 norm output meets a bfloat16 projection ("expected mat1 and mat2 to have the same dtype").
CPU only: real TRL add_fused_lm_head, real unsloth_zoo autocast wrap, rl.py helpers loaded from source."""

from __future__ import annotations

import ast
import copy
import functools
import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {"torch": torch, "functools": functools, "sys": sys, "types": types}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name.startswith(
                (
                    "_spans_several",
                    "_first_tensor",
                    "_ApplyOnInput",
                    "patch_trl_fused_lm_head_devices",
                    "_bf16_autocast_fused",
                    "_keep_bf16",
                )
            ):
                exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
        elif isinstance(node, ast.Assign) and "_bf16_autocast_fused_forward" in ast.unparse(node):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    return ns


@pytest.fixture
def trl_utils(monkeypatch):
    tu = pytest.importorskip("trl.trainer.utils")
    if not hasattr(tu, "add_fused_lm_head"):
        pytest.skip("TRL < 1.15 has no fused LM head")
    # Restore TRL's module bindings after the patch rebinds them.
    for name, mod in list(sys.modules.items()):
        if name.startswith("trl.") and mod is not None and hasattr(mod, "add_fused_lm_head"):
            monkeypatch.setattr(mod, "add_fused_lm_head", mod.add_fused_lm_head)
    return tu


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    for name, module in model.named_modules():
        if "norm" in type(module).__name__.lower():
            module.float()  # what bfloat16 full finetuning does with norms
    _wrap_forward_in_bf16_autocast(model, torch.bfloat16)
    return model


def test_fused_forward_keeps_bf16_autocast(trl_utils):
    _load()["patch_trl_fused_lm_head_devices"]()
    model = _model()
    ids = torch.randint(0, 64, (2, 8))
    model(input_ids = ids)  # the class-level autocast wrap alone works
    trl_utils.add_fused_lm_head(model)
    out = model(input_ids = ids, labels = ids, fused_lm_head = True)
    assert out.log_probs.shape == (2, 7)
    out.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model(input_ids = ids).logits.shape == (2, 8, 64)  # unfused branch unchanged

    # A deepcopy (reference model) calls its own forward, not the original's.
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.lm_head.weight.zero_()
    ref_out = ref(input_ids = ids, labels = ids, fused_lm_head = True)
    assert torch.allclose(
        ref_out.log_probs.float(),
        torch.full_like(ref_out.log_probs.float(), -torch.log(torch.tensor(64.0))),
        atol = 1e-2,
    )
    assert not torch.allclose(
        model(input_ids = ids, labels = ids, fused_lm_head = True).log_probs, ref_out.log_probs
    )

    # Re-applying the patch does not stack wrappers.
    trl_utils.add_fused_lm_head(model)
    model(input_ids = ids, labels = ids, fused_lm_head = True)


def test_models_without_the_autocast_wrap_keep_trl_forward(trl_utils):
    from transformers import LlamaConfig, LlamaForCausalLM

    _load()["patch_trl_fused_lm_head_devices"]()
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size = 64,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            num_key_value_heads = 2,
        )
    )
    trl_utils.add_fused_lm_head(model)
    assert model.forward.__func__.__code__.co_name == "_fused_forward"
