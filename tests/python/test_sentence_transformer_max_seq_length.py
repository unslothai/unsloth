# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

import pytest
import torch

from real_accelerator import has_real_accelerator

LONG_TEXT = " ".join(f"word{i % 60 + 3}" for i in range(200))


def _save_encoder_sentence_model(
    tmp_path,
    family,
    position_limit = 32,
):
    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("FastSentenceTransformer needs CUDA")
    pytest.importorskip("sentence_transformers")
    import unsloth  # noqa: F401
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from transformers import RobertaConfig, RobertaModel

    pad = 0 if family == "bert" else 1
    config_class, model_class = {
        "bert": (BertConfig, BertModel),
        "roberta": (RobertaConfig, RobertaModel),
    }[family]
    config = config_class(
        vocab_size = 64,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 24,
        max_position_embeddings = position_limit + (0 if family == "bert" else pad + 1),
        pad_token_id = pad,
    )
    checkpoint = tmp_path / "base"
    model_class(config).save_pretrained(checkpoint)
    vocab = {f"word{i}": i for i in range(64) if i not in (pad, 2)}
    vocab.update({"[PAD]": pad, "[UNK]": 2})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer, pad_token = "[PAD]", unk_token = "[UNK]"
    ).save_pretrained(checkpoint)
    SentenceTransformer(
        modules = [Transformer(str(checkpoint), max_seq_length = 8), Pooling(16)],
        device = "cpu",
    ).save_pretrained(str(tmp_path / "sentence"))
    return str(tmp_path / "sentence")


def _load(path, **kwargs):
    from unsloth import FastSentenceTransformer
    return FastSentenceTransformer.from_pretrained(path, full_finetuning = True, **kwargs)


@pytest.mark.parametrize("for_inference", [False, True])
@pytest.mark.parametrize("family", ["bert", "roberta"])
def test_requested_max_seq_length_replaces_the_saved_one(tmp_path, family, for_inference):
    path = _save_encoder_sentence_model(tmp_path, family)
    model = _load(path, max_seq_length = 20, for_inference = for_inference)
    assert model.max_seq_length == 20
    assert model.tokenize([LONG_TEXT])["input_ids"].shape[1] == 20

    from sentence_transformers import SentenceTransformer

    model.save_pretrained(str(tmp_path / "saved"))
    assert SentenceTransformer(str(tmp_path / "saved"), device = "cpu").max_seq_length == 20


@pytest.mark.parametrize("for_inference", [False, True])
@pytest.mark.parametrize("family", ["bert", "roberta"])
def test_max_seq_length_is_capped_at_the_model_positions(tmp_path, family, for_inference):
    path = _save_encoder_sentence_model(tmp_path, family)
    model = _load(path, max_seq_length = 4096, for_inference = for_inference)
    assert model.max_seq_length == 32
    assert model.encode([LONG_TEXT]).shape == (1, 16)


def test_saved_max_seq_length_stays_when_none_is_requested(tmp_path):
    path = _save_encoder_sentence_model(tmp_path, "bert")
    assert _load(path).max_seq_length == 8


def test_router_uses_the_smallest_encoder_position_limit(tmp_path):
    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("FastSentenceTransformer needs CUDA")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Router

    query_model = SentenceTransformer(
        _save_encoder_sentence_model(tmp_path / "query", "bert", position_limit = 24),
        device = "cpu",
    )
    document_model = SentenceTransformer(
        _save_encoder_sentence_model(tmp_path / "document", "roberta"),
        device = "cpu",
    )
    router = Router.for_query_document(
        query_modules = list(query_model.children()),
        document_modules = list(document_model.children()),
    )
    path = tmp_path / "router"
    SentenceTransformer(modules = [router], device = "cpu").save_pretrained(str(path))

    model = _load(str(path), max_seq_length = 4096, for_inference = True)
    assert model.max_seq_length == 24
    assert {route[0].max_seq_length for route in model[0].sub_modules.values()} == {24}


def test_compile_disable_keeps_fast_encoder_and_skips_compile(tmp_path, monkeypatch):
    """UNSLOTH_COMPILE_DISABLE=1 keeps the fast encoder path (SDPA, unpadding) and skips only its torch.compile."""
    path = _save_encoder_sentence_model(tmp_path, "bert")
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    model = _load(path)
    assert getattr(model, "_unsloth_fast_encoder", False) is True

    from unsloth import FastSentenceTransformer

    inner = model[0].auto_model
    assert FastSentenceTransformer._apply_torch_compile(model) is model
    assert model[0].auto_model is inner
    assert "_orig_mod" not in model.__dict__


@pytest.mark.skipif(not has_real_accelerator(), reason = "bitsandbytes 8-bit loading needs a GPU")
def test_compile_disable_keeps_8bit_encoders_on_the_loader_that_forwards_it(tmp_path, monkeypatch):
    """The fast encoder path builds no 8-bit config; with UNSLOTH_COMPILE_DISABLE=1 an 8-bit request stays on the
    fallback loader (as before that switch kept the fast path) and loads quantized."""
    bnb = pytest.importorskip("bitsandbytes")
    path = _save_encoder_sentence_model(tmp_path, "bert")
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        path, load_in_8bit = True, load_in_16bit = False, full_finetuning = False
    )
    assert getattr(model, "_unsloth_fast_encoder", False) is False
    assert any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
