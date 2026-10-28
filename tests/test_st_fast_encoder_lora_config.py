# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastSentenceTransformer.get_peft_model must hand every LoRA argument to LoraConfig on the fast encoder path."""

import math

import pytest

pytest.importorskip("sentence_transformers")

from unsloth import FastSentenceTransformer  # noqa: E402


@pytest.fixture
def fast_encoder(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "red", "square"]),
        encoding = "utf-8",
    )
    config = BertConfig(
        vocab_size = 8,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 32,
        max_position_embeddings = 64,
    )
    model_dir = tmp_path / "model"
    BertModel(config).save_pretrained(model_dir)
    BertTokenizerFast(vocab_file = str(vocab)).save_pretrained(model_dir)
    model = SentenceTransformer(
        modules = [Transformer(str(model_dir)), Pooling(16)],
        device = "cpu",
    )
    model._unsloth_fast_encoder = True
    return model


def test_layers_rslora_and_modules_to_save_reach_the_adapter(fast_encoder):
    r, alpha = 4, 8
    model = FastSentenceTransformer.get_peft_model(
        fast_encoder,
        r = r,
        lora_alpha = alpha,
        target_modules = ["query", "key", "value"],
        layers_to_transform = [1],
        use_rslora = True,
        modules_to_save = ["pooler"],
        use_gradient_checkpointing = False,
    )
    peft_model = model[0].auto_model
    config = peft_model.peft_config["default"]
    assert config.use_rslora is True
    assert config.layers_to_transform == [1]
    assert set(config.modules_to_save) == {"pooler"}

    names = [name for name, _ in peft_model.named_parameters() if ".lora_A." in name]
    assert names and all(".layer.1." in name for name in names)
    scalings = [
        module.scaling["default"] for module in peft_model.modules() if hasattr(module, "lora_A")
    ]
    assert scalings and all(s == pytest.approx(alpha / math.sqrt(r)) for s in scalings)
    assert any(
        "modules_to_save" in name and param.requires_grad
        for name, param in peft_model.named_parameters()
    )


def test_float16_modules_to_save_copy_trains_in_float32(fast_encoder):
    import torch

    fast_encoder.half()
    model = FastSentenceTransformer.get_peft_model(
        fast_encoder,
        r = 4,
        lora_alpha = 8,
        target_modules = ["query", "key", "value"],
        modules_to_save = ["pooler"],
        use_gradient_checkpointing = False,
    )
    copies = [
        param
        for name, param in model[0].auto_model.named_parameters()
        if ".modules_to_save." in name
    ]
    assert copies and all(p.dtype == torch.float32 and p.requires_grad for p in copies)
