# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The loaders zero weight[padding_idx], but not for a token the tokenizer adds to every input.

microsoft/harrier-oss-v1-0.6b sets pad_token_id = <|endoftext|> (151643), and its tokenizer.json
post_processor appends that same token to every text; last-token pooling then reads its row.
Zeroing it turned every FastSentenceTransformer embedding into a different vector (cosine ~0.5
vs stock sentence-transformers). See unslothai/unsloth#4984.
"""

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
