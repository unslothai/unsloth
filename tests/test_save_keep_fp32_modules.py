# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A 16-bit save of a non-PEFT model must keep what transformers loads in fp32
(`_keep_in_fp32_modules_strict`, e.g. the MoE router's `e_score_correction_bias`): a bf16 copy
moves the expert-selection scores by up to 0.03 and changes which experts fire. Sliced with
`ast`: importing `unsloth.save` needs a GPU."""

import ast
import re
from pathlib import Path

import pytest
import torch

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"


@pytest.fixture(scope = "module")
def helpers():
    tree = ast.parse(SAVE_PY.read_text(encoding = "utf-8"))
    wanted = {"_cast_state_dict_16bit"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {"re": re, "torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<save>", "exec"), namespace)
    return namespace


def _tiny_glm_lite():
    pytest.importorskip("transformers.models.glm4_moe_lite")  # transformers >= 5.3
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "glm4_moe_lite",
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        n_routed_experts = 4,
        n_shared_experts = 1,
        num_experts_per_tok = 2,
        first_k_dense_replace = 1,
        kv_lora_rank = 16,
        q_lora_rank = 16,
        qk_rope_head_dim = 8,
        qk_nope_head_dim = 8,
        v_head_dim = 8,
        n_group = 1,
        topk_group = 1,
    )
    return AutoModelForCausalLM.from_config(config, dtype = torch.bfloat16)


@pytest.mark.parametrize("target", [torch.bfloat16, torch.float16])
def test_router_bias_stays_fp32(helpers, target):
    model = _tiny_glm_lite()
    state_dict = model.state_dict()
    biases = [k for k in state_dict if k.endswith("e_score_correction_bias")]
    assert biases and all(state_dict[k].dtype == torch.float32 for k in biases)
    with torch.no_grad():
        for k in biases:
            state_dict[k].copy_(torch.linspace(-1, 1, state_dict[k].numel()) / 3)
    out = helpers["_cast_state_dict_16bit"](model, state_dict, target)
    for k in biases:
        assert out[k].dtype == torch.float32 and torch.equal(out[k], state_dict[k])
    others = [k for k, v in out.items() if v.is_floating_point() and k not in biases]
    assert others and all(out[k].dtype == target for k in others)


@pytest.mark.parametrize("target", [torch.bfloat16, torch.float16])
def test_strict_list_from_a_submodule_is_honoured(helpers, target):
    """Same rule on every transformers version: a strict name declared on a nested submodule."""

    class Router(torch.nn.Module):
        _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8, dtype = torch.bfloat16))
            self.register_buffer("e_score_correction_bias", torch.linspace(-1, 1, 4) / 3)

    model = torch.nn.Sequential(torch.nn.Linear(8, 8).to(torch.bfloat16), Router())
    state_dict = model.state_dict()
    out = helpers["_cast_state_dict_16bit"](model, state_dict, target)
    assert out["1.e_score_correction_bias"].dtype == torch.float32
    assert torch.equal(out["1.e_score_correction_bias"], state_dict["1.e_score_correction_bias"])
    assert out["1.weight"].dtype == target and out["0.weight"].dtype == target


def test_models_without_fp32_modules_cast_everything(helpers):
    module = torch.nn.Linear(4, 4)
    out = helpers["_cast_state_dict_16bit"](module, module.state_dict(), torch.bfloat16)
    assert all(v.dtype == torch.bfloat16 for v in out.values())


def test_non_strict_list_only_for_fp16(helpers):
    class Model(torch.nn.Module):
        _keep_in_fp32_modules = ["wo"]

        def __init__(self):
            super().__init__()
            self.wo = torch.nn.Linear(2, 2)
            self.wi = torch.nn.Linear(2, 2)
            self.word_embeddings = torch.nn.Embedding(3, 2)

    model = Model()
    bf16 = helpers["_cast_state_dict_16bit"](model, model.state_dict(), torch.bfloat16)
    fp16 = helpers["_cast_state_dict_16bit"](model, model.state_dict(), torch.float16)
    assert bf16["wo.weight"].dtype == torch.bfloat16
    assert fp16["wo.weight"].dtype == torch.float32 and fp16["wi.weight"].dtype == torch.float16
    # A name is a whole path component: "wo" does not keep "word_embeddings" in fp32.
    assert fp16["word_embeddings.weight"].dtype == torch.float16


def test_generic_save_casts_through_the_helper():
    tree = ast.parse(SAVE_PY.read_text(encoding = "utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unsloth_generic_save"
    )
    calls = {
        c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "_cast_state_dict_16bit" in calls
    decorators = [getattr(d, "id", getattr(d, "attr", None)) for d in fn.decorator_list]
    assert decorators == ["_normalize_tied_weights_keys_for_save", "inference_mode"]
