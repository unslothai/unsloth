# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPT-OSS GGUF: expert LoRA must merge to 16-bit (MXFP4 drops it); attention-only keeps MXFP4."""

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")


class _Stop(Exception):
    pass


def _tiny_gpt_oss_peft(target_parameters):
    from peft import LoraConfig, get_peft_model
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        num_local_experts = 2,
        num_experts_per_tok = 1,
        layer_types = ["full_attention"],
    )
    model = GptOssForCausalLM(config)
    lora = LoraConfig(
        r = 2, lora_alpha = 4, target_modules = ["q_proj"], target_parameters = target_parameters or None
    )
    return get_peft_model(model, lora)


def _routed_save_method(monkeypatch, tmp_path, model):
    import unsloth.tokenizer_utils as tokenizer_utils

    seen = {}

    def _record(**kwargs):
        seen["save_method"] = kwargs.get("save_method")
        raise _Stop()

    for name, value in {
        "raise_if_merging_mistral_format_view": lambda *a, **k: None,
        "raise_if_merging_inkling2": lambda *a, **k: None,
        "_assert_export_target_is_not_base_with_lora_layers": lambda *a, **k: None,
        "_gguf_remote_code_approved": lambda *a, **k: True,
        "_preflight_gguf_disk": lambda **k: (k["save_directory"], True),
        "fix_tokenizer_bos_token": lambda tokenizer: (False, None),
        "_resolve_imatrix_file": lambda *a, **k: None,
        "unsloth_generic_save": _record,
    }.items():
        monkeypatch.setattr(save_mod, name, value)
    monkeypatch.setattr(tokenizer_utils, "_tokenizer_objects", lambda tokenizer: [])
    with pytest.raises(RuntimeError):
        save_mod.unsloth_save_pretrained_gguf(
            model, str(tmp_path / "out"), tokenizer = object(), quantization_method = "bf16"
        )
    return seen.get("save_method")


def test_expert_lora_is_merged_to_16bit(monkeypatch, tmp_path):
    model = _tiny_gpt_oss_peft(["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
    assert _routed_save_method(monkeypatch, tmp_path, model) == "merged_16bit"


def test_attention_only_lora_keeps_mxfp4(monkeypatch, tmp_path):
    model = _tiny_gpt_oss_peft([])
    assert _routed_save_method(monkeypatch, tmp_path, model) == "mxfp4"
