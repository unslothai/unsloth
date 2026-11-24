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

    # Re-applying the patch: both branches still run (the unfused one used to recurse through a shared delegate).
    trl_utils.add_fused_lm_head(model)
    assert model(input_ids = ids, labels = ids, fused_lm_head = True).log_probs.shape == (2, 7)
    assert model(input_ids = ids).logits.shape == (2, 8, 64)
    trl_utils.add_fused_lm_head(ref)
    assert ref(input_ids = ids).logits.shape == (2, 8, 64)


def test_fused_forward_keeps_the_forward_signature(trl_utils):
    # generate() validates model kwargs against inspect.signature(model.forward) and only trims prefill logits
    # when it lists logits_to_keep.
    import inspect

    _load()["patch_trl_fused_lm_head_devices"]()
    model = _model()
    trl_utils.add_fused_lm_head(model)
    params = inspect.signature(model.forward).parameters
    for name in ("input_ids", "position_ids", "logits_to_keep"):
        assert name in params
    ids = torch.randint(0, 64, (1, 4))
    out = model.generate(
        ids,
        attention_mask = torch.ones_like(ids),
        position_ids = torch.arange(4)[None],
        max_new_tokens = 2,
        min_new_tokens = 2,
        do_sample = False,
    )
    assert out.shape == (1, 6)


def test_fused_forward_after_the_autocast_subclass_is_removed(trl_utils):
    # Re-preparing in float32 drops the autocast subclass and its marker; the instance forward must still run.
    _load()["patch_trl_fused_lm_head_devices"]()
    model = _model()
    trl_utils.add_fused_lm_head(model)
    model.float()
    model.__class__ = type(model).__mro__[1]
    model._unsloth_bf16_autocast_wrapped = False
    ids = torch.randint(0, 64, (2, 8))
    assert model(input_ids = ids, labels = ids, fused_lm_head = True).log_probs.shape == (2, 7)
    assert model(input_ids = ids).logits.shape == (2, 8, 64)


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
