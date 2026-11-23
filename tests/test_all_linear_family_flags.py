# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""target_modules = "all-linear" on models without a multimodal remap hook: T5 / BART turn it into their own leaf names
and DeepSeek-V4.1 style remap hooks keep the caller's attention / MLP choice."""

import pytest
import torch


@pytest.fixture
def vision(monkeypatch):
    import unsloth  # noqa: F401
    import unsloth.models.vision as vision

    monkeypatch.setattr(vision, "clean_gpu_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    if not hasattr(torch.cpu, "empty_cache"):
        monkeypatch.setattr(torch.cpu, "empty_cache", lambda: None, raising = False)
    return vision


def _seq2seq(kind):
    import transformers

    torch.manual_seed(0)
    if kind == "t5":
        config = transformers.T5Config(
            vocab_size = 128, d_model = 64, d_ff = 128, num_layers = 2, num_heads = 4, d_kv = 16
        )
        return transformers.T5ForConditionalGeneration(config), {"q", "k", "v", "o", "wi", "wo"}
    config = transformers.BartConfig(
        vocab_size = 128,
        d_model = 64,
        encoder_layers = 2,
        decoder_layers = 2,
        encoder_attention_heads = 4,
        decoder_attention_heads = 4,
        encoder_ffn_dim = 128,
        decoder_ffn_dim = 128,
    )
    return transformers.BartForConditionalGeneration(config), {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    }


def _wrapped(model):
    return {
        name.rsplit(".lora_A", 1)[0] for name, _ in model.named_parameters() if ".lora_A." in name
    }


@pytest.mark.parametrize("kind", ["t5", "bart"])
def test_all_linear_wraps_every_seq2seq_linear(vision, kind):
    # all-linear used to set finetune_audio_layers before the seq2seq branch turned it into leaf names, so the leaves
    # went through get_peft_regex: T5 (q / k / v / o / wi / wo) raised "No layers to finetune", BART lost fc1 / fc2.
    model, leaves = _seq2seq(kind)
    model.max_seq_length = 64
    expected = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in leaves
    }
    out = vision.FastBaseModel.get_peft_model(
        model, r = 4, target_modules = "all-linear", use_gradient_checkpointing = False
    )
    wrapped = {name.split("base_model.model.", 1)[-1] for name in _wrapped(out)}
    assert wrapped == expected, sorted(expected - wrapped)[:6]


@pytest.mark.parametrize(
    "flags, families",
    [
        (dict(finetune_mlp_modules = False), {"attention": True, "mlp": False}),
        (dict(finetune_attention_modules = False), {"attention": False, "mlp": True}),
        (dict(), {"attention": True, "mlp": True}),
    ],
)
def test_all_linear_keeps_family_flags_for_a_remap_without_vision(vision, flags, families):
    # A remap hook with no `vision` parameter (native DeepSeek-V4.1) is handed the caller's families, as before.
    import transformers

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    model = transformers.LlamaForCausalLM(config)
    model.max_seq_length = 64
    seen = {}

    def unsloth_lora_target_modules(
        self,
        target_modules,
        attention = True,
        mlp = True,
    ):
        seen.update(attention = attention, mlp = mlp)
        return [
            name
            for name, module in self.named_modules()
            if isinstance(module, torch.nn.Linear)
            and ((attention and ".self_attn." in name) or (mlp and ".mlp." in name))
        ]

    class Remapped(type(model)):
        pass

    Remapped.unsloth_lora_target_modules = unsloth_lora_target_modules
    model.__class__ = Remapped
    out = vision.FastBaseModel.get_peft_model(
        model, r = 4, target_modules = "all-linear", use_gradient_checkpointing = False, **flags
    )
    assert seen == families
    wrapped = _wrapped(out)
    assert any(".self_attn." in name for name in wrapped) == families["attention"]
    assert any(".mlp." in name for name in wrapped) == families["mlp"]
