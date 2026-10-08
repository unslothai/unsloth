# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full finetuning a pre-quantized bitsandbytes checkpoint dequantizes it instead of crashing (#2613)."""

import importlib.util
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth.models.loader_utils import _dequantize_bitsandbytes_for_full_finetuning


class _Deserialize:
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer


class _FakeModel(torch.nn.Module):
    def __init__(self, quant_method, conversions):
        super().__init__()
        self.hf_quantizer = SimpleNamespace(
            quantization_config = SimpleNamespace(quant_method = quant_method)
        )
        self._weight_conversions = conversions(self.hf_quantizer)
        self.dequantized_with = None

    def dequantize(self, dtype = None):
        self.dequantized_with = dtype
        return self


def test_bitsandbytes_is_dequantized_and_its_load_converter_dropped(capsys):
    kept = SimpleNamespace(operations = [object()])
    model = _FakeModel(
        "bitsandbytes",
        lambda q: [SimpleNamespace(operations = [_Deserialize(q)]), kept],
    )
    assert _dequantize_bitsandbytes_for_full_finetuning(model, torch.bfloat16, "local/dir") is True
    assert model.dequantized_with is torch.bfloat16
    assert model._weight_conversions == [kept]
    assert "local/dir" in capsys.readouterr().out


def test_other_quant_methods_are_left_alone():
    model = _FakeModel("fp8", lambda q: [SimpleNamespace(operations = [_Deserialize(q)])])
    assert _dequantize_bitsandbytes_for_full_finetuning(model, torch.bfloat16) is False
    assert model.dequantized_with is None and len(model._weight_conversions) == 1
    assert _dequantize_bitsandbytes_for_full_finetuning(torch.nn.Linear(2, 2)) is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("bitsandbytes") is None,
    reason = "needs CUDA and bitsandbytes",
)
def test_full_finetune_a_local_bnb_4bit_folder(tmp_path):
    script = textwrap.dedent(
        """
        import sys, torch
        from unsloth import FastLanguageModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        src, quantized, saved = sys.argv[1:4]
        AutoModelForCausalLM.from_pretrained(
            src, quantization_config = BitsAndBytesConfig(load_in_4bit = True), device_map = "cuda:0"
        ).save_pretrained(quantized)
        AutoTokenizer.from_pretrained(src).save_pretrained(quantized)
        model, tokenizer = FastLanguageModel.from_pretrained(
            quantized, max_seq_length = 64, load_in_4bit = False, full_finetuning = True
        )
        assert all(p.is_floating_point() and p.requires_grad for p in model.parameters())
        ids = torch.tensor([[1, 2, 3, 4]], device = model.device)
        model(input_ids = ids, labels = ids).loss.backward()
        model.save_pretrained(saved)
        print("FULL_FINETUNE_OK")
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            str(tmp_path / "bnb"),
            str(tmp_path / "saved"),
        ],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert "FULL_FINETUNE_OK" in result.stdout, result.stdout[-3000:] + result.stderr[-3000:]
    config = (tmp_path / "saved" / "config.json").read_text(encoding = "utf-8")
    assert "quantization_config" not in config
