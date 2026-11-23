# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gemma chat text loses its template <bos> only when the tokenizer puts one back."""

import pytest
from datasets import Dataset

from utils.datasets import format_and_template_dataset
from utils.datasets.online_tokenization import resolve_add_special_tokens

_BOS_ID = 2
_ROWS = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]


class _GemmaTokenizer:
    chat_template = "{{ bos_token }}{{ messages }}"
    bos_token = "<bos>"
    bos_token_id = _BOS_ID
    eos_token = "<eos>"
    adds_bos = True

    def apply_chat_template(self, conversation, **_kwargs):
        return "<bos>" + "".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)

    def __call__(
        self,
        text,
        add_special_tokens = True,
        **_kwargs,
    ):
        # A literal <bos> in the text is the special token, as in the real tokenizers.
        input_ids = []
        if text.startswith("<bos>"):
            input_ids.append(_BOS_ID)
            text = text.removeprefix("<bos>")
        input_ids += [7] * len(text)
        if add_special_tokens and self.adds_bos:
            input_ids.insert(0, _BOS_ID)
        return {"input_ids": input_ids}


class _Gemma4Tokenizer(_GemmaTokenizer):
    adds_bos = False


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def apply_chat_template(self, conversation, **kwargs):
        return self.tokenizer.apply_chat_template(conversation, **kwargs)


def _format(tokenizer, model_name = "stub/gemma-stub-it"):
    result = format_and_template_dataset(
        Dataset.from_list(_ROWS), model_name = model_name, tokenizer = tokenizer
    )
    assert result["success"], result["errors"]
    return list(result["dataset"]["text"])


def _trained_ids(tokenizer, text):
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    return inner(text, add_special_tokens = resolve_add_special_tokens(tokenizer, text))["input_ids"]


def test_bos_is_stripped_when_the_tokenizer_adds_it_back():
    assert _format(_GemmaTokenizer()) == ["<user>q<assistant>a"]


def test_bos_is_kept_when_the_tokenizer_does_not_add_one():
    assert _format(_Gemma4Tokenizer()) == ["<bos><user>q<assistant>a"]


def test_a_processor_is_judged_by_its_inner_tokenizer():
    assert _format(_Processor(_GemmaTokenizer())) == ["<user>q<assistant>a"]
    assert _format(_Processor(_Gemma4Tokenizer())) == ["<bos><user>q<assistant>a"]


def test_other_model_names_keep_the_template_text_as_rendered():
    assert _format(_GemmaTokenizer(), model_name = "stub/other-it") == ["<bos><user>q<assistant>a"]


@pytest.mark.parametrize(
    "tokenizer",
    [
        _GemmaTokenizer(),
        _Gemma4Tokenizer(),
        _Processor(_GemmaTokenizer()),
        _Processor(_Gemma4Tokenizer()),
    ],
    ids = ["adds-bos", "no-bos", "processor-adds-bos", "processor-no-bos"],
)
def test_the_tokenized_row_starts_with_exactly_one_bos(tokenizer):
    (text,) = _format(tokenizer)

    input_ids = _trained_ids(tokenizer, text)

    assert input_ids[0] == _BOS_ID
    assert input_ids.count(_BOS_ID) == 1
