# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch

pytest.importorskip("torchao")


def test_fast_language_model_applies_qat_on_the_new_model_path(monkeypatch):
    import unsloth.models.vision as vision
    from torchao.quantization.qat import FakeQuantizedLinear
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from unsloth import FastLanguageModel

    monkeypatch.setattr(vision, "clean_gpu_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    if not hasattr(torch.cpu, "empty_cache"):
        monkeypatch.setattr(torch.cpu, "empty_cache", lambda: None, raising = False)
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "1")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size = 128,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 32,
        )
    )
    model.max_seq_length = 64
    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        target_modules = ["q_proj", "v_proj"],
        use_gradient_checkpointing = False,
        qat_scheme = "int8-int4",
    )
    assert any(isinstance(m, FakeQuantizedLinear) for m in model.modules())
