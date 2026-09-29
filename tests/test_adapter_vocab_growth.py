# SPDX-License-Identifier: AGPL-3.0-only
"""A LoRA adapter trained after adding tokens reloads onto its smaller base (#1215)."""

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")


def _save_adapter(
    path,
    rows,
    hidden = 8,
    with_lm_head = True,
):
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, hidden),
        "base_model.model.model.embed_tokens.weight": torch.zeros(rows, hidden),
    }
    if with_lm_head:
        tensors["base_model.model.lm_head.weight"] = torch.zeros(rows, hidden)
    path.mkdir(parents = True, exist_ok = True)
    safetensors_torch.save_file(tensors, str(path / "adapter_model.safetensors"))
    return str(path)


def _tiny_model(vocab = 32, hidden = 8):
    config = transformers.LlamaConfig(
        vocab_size = vocab,
        hidden_size = hidden,
        intermediate_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    return transformers.LlamaForCausalLM(config)


def test_vocab_rows_read_from_saved_embeddings(tmp_path):
    from unsloth.models.loader import _adapter_vocab_rows

    assert _adapter_vocab_rows(_save_adapter(tmp_path / "a", 35)) == 35
    lora_embed = tmp_path / "b"
    lora_embed.mkdir()
    safetensors_torch.save_file(
        {"base_model.model.model.embed_tokens.lora_embedding_A": torch.zeros(4, 40)},
        str(lora_embed / "adapter_model.safetensors"),
    )
    assert _adapter_vocab_rows(str(lora_embed)) == 40
    plain = tmp_path / "c"
    plain.mkdir()
    safetensors_torch.save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8)},
        str(plain / "adapter_model.safetensors"),
    )
    assert _adapter_vocab_rows(str(plain)) is None
    assert _adapter_vocab_rows(str(tmp_path / "missing")) is None


def test_base_grows_to_fit_adapter(tmp_path):
    from unsloth.models.loader import _grow_vocab_for_adapter

    model = _tiny_model()
    assert _grow_vocab_for_adapter(model, _save_adapter(tmp_path / "a", 35))
    assert model.get_input_embeddings().weight.shape[0] == 35
    assert model.get_output_embeddings().weight.shape[0] == 35


def test_short_lm_head_alone_is_grown(tmp_path):
    # patch_model_and_tokenizer grows only the input embedding to len(tokenizer).
    from unsloth.models.loader import _grow_vocab_for_adapter

    model = _tiny_model()
    embed = model.get_input_embeddings()
    model.set_input_embeddings(torch.nn.Embedding(35, embed.weight.shape[1]))
    assert _grow_vocab_for_adapter(model, _save_adapter(tmp_path / "a", 35))
    assert model.get_output_embeddings().weight.shape[0] == 35


def test_never_shrinks_or_touches_matching_base(tmp_path):
    from unsloth.models.loader import _grow_vocab_for_adapter

    model = _tiny_model(vocab = 40)
    weight = model.get_input_embeddings().weight
    assert not _grow_vocab_for_adapter(model, _save_adapter(tmp_path / "small", 35))
    assert not _grow_vocab_for_adapter(model, _save_adapter(tmp_path / "same", 40))
    assert model.get_input_embeddings().weight is weight
