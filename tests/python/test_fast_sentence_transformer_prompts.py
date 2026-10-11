# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

import json

import pytest
import torch

from real_accelerator import has_real_accelerator


PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}


def _save_decoder_sentence_model(tmp_path):
    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("the FastModel route requires CUDA")
    pytest.importorskip("sentence_transformers")
    import unsloth  # noqa: F401
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
    SentenceTransformer(
        modules = [Transformer(str(checkpoint)), Pooling(32)],
        device = "cpu",
        prompts = PROMPTS,
        default_prompt_name = "query",
        similarity_fn_name = "dot",
    ).save_pretrained(tmp_path / "sentence")
    return str(tmp_path / "sentence")


def test_decoder_embedding_model_keeps_its_prompts(tmp_path):
    path = _save_decoder_sentence_model(tmp_path)
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        path,
        load_in_4bit = False,
        full_finetuning = True,
        max_seq_length = 32,
    )
    assert {k: model.prompts[k] for k in PROMPTS} == PROMPTS
    assert model.default_prompt_name == "query"
    assert model.similarity_fn_name == "dot"

    model.save_pretrained(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "config_sentence_transformers.json").read_text())
    assert {k: saved["prompts"][k] for k in PROMPTS} == PROMPTS
    assert saved["default_prompt_name"] == "query"
    assert saved["similarity_fn_name"] == "dot"


def test_hub_error_reading_sentence_config_does_not_fail_the_load(tmp_path, monkeypatch):
    path = _save_decoder_sentence_model(tmp_path)
    import sentence_transformers.util
    from unsloth import FastSentenceTransformer

    real_load_file_path = sentence_transformers.util.load_file_path

    def load_file_path(model_name_or_path, filename, *args, **kwargs):
        if filename == "config_sentence_transformers.json":
            raise OSError("429 Too Many Requests")
        return real_load_file_path(model_name_or_path, filename, *args, **kwargs)

    monkeypatch.setattr(sentence_transformers.util, "load_file_path", load_file_path)
    model = FastSentenceTransformer.from_pretrained(
        path,
        load_in_4bit = False,
        full_finetuning = True,
        max_seq_length = 32,
    )
    assert model.prompts.get("query", "") == ""


def test_local_files_only_reaches_the_sentence_config_lookup(tmp_path, monkeypatch):
    path = _save_decoder_sentence_model(tmp_path)
    import sentence_transformers.util
    from unsloth import FastSentenceTransformer

    real_load_file_path = sentence_transformers.util.load_file_path
    seen = []

    def load_file_path(model_name_or_path, filename, *args, **kwargs):
        if filename == "config_sentence_transformers.json":
            seen.append(kwargs.get("local_files_only"))
        return real_load_file_path(model_name_or_path, filename, *args, **kwargs)

    monkeypatch.setattr(sentence_transformers.util, "load_file_path", load_file_path)
    model = FastSentenceTransformer.from_pretrained(
        path,
        load_in_4bit = False,
        full_finetuning = True,
        max_seq_length = 32,
        local_files_only = True,
    )
    assert seen == [True]
    assert {k: model.prompts[k] for k in PROMPTS} == PROMPTS
