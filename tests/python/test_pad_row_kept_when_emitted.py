# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pad-row zeroing at load must spare EOS ids and tokens the tokenizer adds itself (#4984)."""

import json

import pytest

import unsloth.tokenizer_utils as tu


class _Tok:
    def __init__(
        self,
        prefix = (),
        suffix = (),
        fail = False,
    ):
        self.prefix, self.suffix, self.fail = list(prefix), list(suffix), fail

    def __call__(
        self,
        text,
        add_special_tokens = True,
    ):
        if self.fail:
            raise ValueError("needs images")
        ids = [65]
        if add_special_tokens:
            ids = self.prefix + ids + self.suffix
        return {"input_ids": ids}


class _Proc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        raise AssertionError("the inner tokenizer must be asked, not the processor")


def test_appended_eos_is_reported():
    assert tu._special_ids_added_on_encode(_Tok(suffix = [151643])) == {151643}


def test_prepended_bos_and_appended_eos():
    assert tu._special_ids_added_on_encode(_Tok(prefix = [2], suffix = [1])) == {1, 2}


def test_plain_tokenizer_adds_nothing():
    assert tu._special_ids_added_on_encode(_Tok()) == frozenset()


def test_processor_asks_inner_tokenizer():
    assert tu._special_ids_added_on_encode(_Proc(_Tok(suffix = [7]))) == {7}


def test_batched_nested_ids():
    class _Batched(_Tok):
        def __call__(
            self,
            text,
            add_special_tokens = True,
        ):
            return {"input_ids": [super().__call__(text, add_special_tokens)["input_ids"]]}

    assert tu._special_ids_added_on_encode(_Batched(prefix = [2])) == {2}


def test_failing_tokenizer_keeps_old_behaviour():
    assert tu._special_ids_added_on_encode(_Tok(fail = True)) == frozenset()


def _tokenizer_from_json(post_processor, special):
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    vocab = {"A": 0, **{tok: i + 1 for i, tok in enumerate(special)}}
    spec = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": i,
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for t, i in vocab.items()
            if t != "A"
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": post_processor,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "A"},
    }
    return PreTrainedTokenizerFast(tokenizer_object = tokenizers.Tokenizer.from_str(json.dumps(spec)))


def test_real_template_processing_appending_pad():
    # The harrier-oss-v1-0.6b post_processor: single = $A <|endoftext|>.
    tok = _tokenizer_from_json(
        {
            "type": "TemplateProcessing",
            "single": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
            ],
            "special_tokens": {
                "<|endoftext|>": {"id": "<|endoftext|>", "ids": [1], "tokens": ["<|endoftext|>"]}
            },
        },
        ["<|endoftext|>"],
    )
    tok.pad_token = "<|endoftext|>"
    assert tok.pad_token_id in tu._special_ids_added_on_encode(tok)


def test_real_tokenizer_without_post_processor():
    tok = _tokenizer_from_json(None, ["<pad>"])
    tok.pad_token = "<pad>"
    assert tok.pad_token_id not in tu._special_ids_added_on_encode(tok)


class _PadTok(_Tok):
    def __init__(
        self,
        pad_token_id,
        eos_token_id,
        suffix = (),
    ):
        super().__init__(suffix = suffix)
        self.pad_token_id, self.eos_token_id = pad_token_id, eos_token_id


def _model(
    padding_idx,
    tied = False,
    config_eos = None,
    generation_eos = None,
):
    import types

    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(8, 4, padding_idx = padding_idx)
    with torch.no_grad():
        model.embed.weight.fill_(1.0)
    model.lm_head = torch.nn.Linear(4, 8, bias = False)
    if tied:
        model.lm_head.weight = model.embed.weight
    model.config = types.SimpleNamespace(eos_token_id = config_eos)
    model.generation_config = types.SimpleNamespace(eos_token_id = generation_eos)
    return model


def _zeroed(model):
    return [i for i in range(8) if not model.embed.weight[i].any()]


def test_plain_pad_row_is_zeroed():
    model = _model(5)
    tu.zero_pad_embedding_rows(model, _PadTok(5, 2))
    assert _zeroed(model) == [5]


def test_pad_equal_to_eos_is_never_zeroed():
    model = _model(2)
    tu.zero_pad_embedding_rows(model, _PadTok(2, 2))
    assert _zeroed(model) == []


@pytest.mark.parametrize("where", ["config", "generation"])
def test_pad_in_model_eos_list_is_never_zeroed(where):
    # Llama 3 style: several EOS ids in config / generation_config, the tokenizer names one.
    kwargs = {"config_eos": [2, 5]} if where == "config" else {"generation_eos": [2, 5]}
    model = _model(5, **kwargs)
    tu.zero_pad_embedding_rows(model, _PadTok(5, 2))
    assert _zeroed(model) == []


def test_config_padding_idx_on_eos_row_is_not_zeroed():
    # Phi-3.5-mini: config pad_token_id 32000 is EOS, the tokenizer pads with 32009.
    model = _model(2)
    tu.zero_pad_embedding_rows(model, _PadTok(6, 2))
    assert _zeroed(model) == []


def test_emitted_pad_row_is_kept():
    model = _model(5)
    tu.zero_pad_embedding_rows(model, _PadTok(5, 2, suffix = [5]))
    assert _zeroed(model) == []


def test_tied_embedding_is_not_zeroed():
    model = _model(5, tied = True)
    tu.zero_pad_embedding_rows(model, _PadTok(5, 2))
    assert _zeroed(model) == []


def test_no_tokenizer_or_pad_is_a_no_op():
    model = _model(5)
    tu.zero_pad_embedding_rows(model, None)
    tu.zero_pad_embedding_rows(model, _PadTok(None, 2))
    assert _zeroed(model) == []
