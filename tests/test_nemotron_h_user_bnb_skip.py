# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A caller's BitsAndBytesConfig must keep Nemotron-H's mixer.out_proj unquantized: the fused Mamba-2
kernel runs F.linear on out_proj.weight, which crashes on a packed Params4bit."""

import ast
import copy
import importlib.util
import os

import pytest


def _bnb(**kw):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = "nf4", **kw)


def test_user_bnb_config_gets_nemotron_h_skip_modules():
    from unsloth.models.vision import _with_architecture_skip_modules
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    user = _bnb()
    before = copy.deepcopy(user.llm_int8_skip_modules)
    out = _with_architecture_skip_modules(user, ["nemotron_h"])
    assert "out_proj" in out.llm_int8_skip_modules
    assert set(SKIP_QUANTIZATION_MODULES) <= set(out.llm_int8_skip_modules)
    assert user.llm_int8_skip_modules == before, "the caller's object must not be mutated"

    explicit = _bnb(llm_int8_skip_modules = ["lm_head"])
    out = _with_architecture_skip_modules(explicit, ["nemotron_h"])
    assert out.llm_int8_skip_modules == ["lm_head", "out_proj"]

    already = _bnb(llm_int8_skip_modules = ["lm_head", "out_proj"])
    assert _with_architecture_skip_modules(already, ["nemotron_h"]) is already

    as_dict = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "llm_int8_skip_modules": ["lm_head"],
    }
    out = _with_architecture_skip_modules(as_dict, ["nemotron_h"])
    assert out["llm_int8_skip_modules"] == ["lm_head", "out_proj"] and as_dict[
        "llm_int8_skip_modules"
    ] == ["lm_head"]

    # transformers reads a dict with a load flag as bitsandbytes even without quant_method.
    shorthand = {"load_in_4bit": True}
    out = _with_architecture_skip_modules(shorthand, ["nemotron_h"])
    assert "out_proj" in out["llm_int8_skip_modules"] and "llm_int8_skip_modules" not in shorthand


def test_other_architectures_and_quantizers_are_untouched():
    from unsloth.models.vision import _with_architecture_skip_modules

    user = _bnb()
    assert _with_architecture_skip_modules(user, ["llama"]) is user
    assert _with_architecture_skip_modules(None, ["nemotron_h"]) is None
    fp8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert _with_architecture_skip_modules(fp8, ["nemotron_h"]) is fp8


def test_from_pretrained_merges_before_the_planner_and_the_load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "unsloth",
        "models",
        "vision.py",
    )
    tree = ast.parse(open(path, encoding = "utf-8").read())
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "user_quantization_config" for t in n.targets)
    ]
    assert assigns and all(
        "_with_architecture_skip_modules" in ast.unparse(n.value) for n in assigns
    )


TINY = "trl-internal-testing/tiny-NemotronHForCausalLM-3.5-lightning"


@pytest.mark.skipif(
    not (importlib.util.find_spec("mamba_ssm") and importlib.util.find_spec("bitsandbytes")),
    reason = "needs mamba_ssm (fused Mamba-2 path) and bitsandbytes",
)
def test_tiny_nemotron_h_trains_with_a_user_bnb_config():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    modeling = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    if not getattr(modeling.NemotronHPreTrainedModel, "supports_gradient_checkpointing", False):
        pytest.skip("this transformers' NemotronH does not support gradient checkpointing")
    from unsloth import FastModel
    from transformers import BitsAndBytesConfig

    qc = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    model, tok = FastModel.from_pretrained(
        TINY, max_seq_length = 256, dtype = torch.bfloat16, quantization_config = qc
    )
    mixers = [m for n, m in model.named_modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
    assert mixers
    for m in mixers:
        assert type(m.out_proj).__name__ != "Linear4bit" and m.out_proj.weight.dim() == 2
    assert any(
        type(m).__name__ == "Linear4bit" for m in model.modules()
    ), "the rest must still be 4bit"
    model = FastModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
    )
    model.train()
    ids = torch.randint(10, 1000, (2, 64), device = "cuda")
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n or "lora_A" in n
    )
