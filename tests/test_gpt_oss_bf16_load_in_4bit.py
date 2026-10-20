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

"""gpt-oss fused BF16 checkpoint + load_in_4bit must not go through zoo's bnb split-expert classes (random experts)."""

import os
from types import SimpleNamespace

import pytest
import torch

import unsloth  # noqa: F401  (patches before transformers)
from unsloth.models import loader_utils


def gpt_oss_checkpoint_has_fused_experts(config):
    return loader_utils.gpt_oss_checkpoint_has_fused_experts(config)


def sync_unsloth_model_name_bnb_flags(*args, **kwargs):
    return loader_utils.sync_unsloth_model_name_bnb_flags(*args, **kwargs)

BNB_QC = {"quant_method": "bitsandbytes", "load_in_4bit": True}
MXFP4_QC = {"quant_method": "mxfp4"}


def test_fused_checkpoint_detection():
    assert gpt_oss_checkpoint_has_fused_experts(None) is None
    assert gpt_oss_checkpoint_has_fused_experts(SimpleNamespace()) is True
    assert gpt_oss_checkpoint_has_fused_experts(SimpleNamespace(quantization_config = None)) is True
    assert gpt_oss_checkpoint_has_fused_experts(SimpleNamespace(quantization_config = BNB_QC)) is False
    assert gpt_oss_checkpoint_has_fused_experts(SimpleNamespace(quantization_config = MXFP4_QC)) is False


@pytest.mark.parametrize(
    "config, expected",
    [
        (SimpleNamespace(), "gpt_oss,"),  # fused BF16: no bnb split classes
        (SimpleNamespace(quantization_config = BNB_QC), "gpt_oss,_load_in_4bit_"),
        (None, "gpt_oss,_load_in_4bit_"),  # unknown config: unchanged behaviour
    ],
)
def test_sync_token(monkeypatch, config, expected):
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0")
    sync_unsloth_model_name_bnb_flags(True, False, config = config)
    assert os.environ["UNSLOTH_MODEL_NAME"] == expected


def test_sync_refuses_when_split_classes_already_installed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "1")
    with pytest.raises(RuntimeError, match = "randomly initialised"):
        sync_unsloth_model_name_bnb_flags(True, False, config = SimpleNamespace())


def test_sync_ignores_other_models(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "llama,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "1")
    sync_unsloth_model_name_bnb_flags(True, False, config = SimpleNamespace())
    assert os.environ["UNSLOTH_MODEL_NAME"] == "llama,_load_in_4bit_"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU for bitsandbytes")
def test_bf16_checkpoint_loads_real_weights_in_4bit(tmp_path):
    pytest.importorskip("bitsandbytes")
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = "trl-internal-testing/tiny-GptOssForCausalLM"
    ckpt = str(tmp_path / "tiny_gpt_oss_bf16")
    ref = AutoModelForCausalLM.from_pretrained(src, dtype = torch.bfloat16)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "experts" in name or "router" in name:
                param.normal_(0.0, 0.05)
    ref.save_pretrained(ckpt)
    AutoTokenizer.from_pretrained(src).save_pretrained(ckpt)
    del ref
    sd = load_file(os.path.join(ckpt, "model.safetensors"))

    from unsloth import FastLanguageModel
    import bitsandbytes.functional as bnbF

    model, _ = FastLanguageModel.from_pretrained(
        ckpt, max_seq_length = 64, load_in_4bit = True, dtype = torch.bfloat16
    )
    mlp = model.model.layers[0].mlp
    router = mlp.router
    router_w = router.linear.weight if hasattr(router, "linear") else router.weight
    torch.testing.assert_close(
        router_w.detach().float().cpu(), sd["model.layers.0.mlp.router.weight"].float()
    )

    experts = mlp.experts
    assert hasattr(experts, "gate_up_proj"), type(experts)
    w = experts.gate_up_proj
    assert type(w).__name__ == "Params4bit" and w.quant_state is not None
    deq = bnbF.dequantize_4bit(w.data, w.quant_state).float().cpu()
    ck = sd["model.layers.0.mlp.experts.gate_up_proj"].float()
    if deq.shape != ck.shape:
        deq = deq.transpose(-1, -2)
    assert float((deq - ck).abs().max()) < 0.25 * float(ck.abs().max())

    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "gate_up_proj", "down_proj"],
    )
    FastLanguageModel.for_training(model)
    ids = torch.randint(0, 200, (1, 16), device = "cuda")
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    expert_grads = [
        p.grad
        for n, p in model.named_parameters()
        if "experts" in n and "lora_B" in n and p.grad is not None
    ]
    assert expert_grads and any(float(g.abs().sum()) > 0 for g in expert_grads)
