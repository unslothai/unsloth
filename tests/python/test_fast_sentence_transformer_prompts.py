# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

import json

import pytest
import torch

from real_accelerator import has_real_accelerator


def test_decoder_embedding_model_keeps_its_prompts(tmp_path):
    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("the FastModel route requires CUDA")
    pytest.importorskip("sentence_transformers")
    from unsloth import FastSentenceTransformer
    from transformers import Gemma3TextConfig, Gemma3TextModel, PreTrainedTokenizerFast
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    config = Gemma3TextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 48,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        max_position_embeddings = 32,
        sliding_window = 4,
        layer_types = ["sliding_attention", "full_attention"],
        use_bidirectional_attention = True,
        use_cache = False,
        pad_token_id = 0,
    )
    checkpoint = tmp_path / "base"
    Gemma3TextModel(config).save_pretrained(checkpoint)
    vocab = {f"word{i}": i for i in range(64)}
    del vocab["word0"], vocab["word1"]
    vocab.update({"[PAD]": 0, "[UNK]": 1})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        unk_token = "[UNK]",
        model_max_length = 32,
    ).save_pretrained(checkpoint)
    prompts = {"query": "task: search result | query: ", "document": "title: none | text: "}
    SentenceTransformer(
        modules = [Transformer(str(checkpoint)), Pooling(32)],
        device = "cpu",
        prompts = prompts,
        default_prompt_name = "query",
        similarity_fn_name = "dot",
    ).save_pretrained(tmp_path / "sentence")

    model = FastSentenceTransformer.from_pretrained(
        str(tmp_path / "sentence"),
        load_in_4bit = False,
        full_finetuning = True,
        max_seq_length = 32,
    )
    assert {k: model.prompts[k] for k in prompts} == prompts
    assert model.default_prompt_name == "query"
    assert model.similarity_fn_name == "dot"

    model.save_pretrained(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "config_sentence_transformers.json").read_text())
    assert {k: saved["prompts"][k] for k in prompts} == prompts
    assert saved["default_prompt_name"] == "query"
    assert saved["similarity_fn_name"] == "dot"
