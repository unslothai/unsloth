# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest

tokenizers = pytest.importorskip("tokenizers")
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import check_dataset, get_chat_template

TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] not in ['user', 'assistant'] %}{{ raise_exception('bad role') }}{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
)
MARKER = "<|im_start|>assistant\n"


def _tokenizer(template = TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<|im_start|>", "<|im_end|>", "hi", "hello", "there"] + [f"w{i}" for i in range(40)]
    tk = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}, unk_token = "[UNK]")
    )
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object = tk, unk_token = "[UNK]", eos_token = "<|im_end|>")
    tok.chat_template = template
    tok.name_or_path = "tiny/chat-tokenizer"
    return tok


def _chat(user, assistant):
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    }


LONG = " ".join(f"w{i}" for i in range(30))
ROWS = [
    _chat("hi", "hello"),
    _chat("hi there", "hello there"),
    _chat("hi", "hello"),
    _chat("hi", LONG),
    _chat("hi there", LONG),
    _chat("   ", ""),
    "not a row",
    {"prompt": "hi"},
    {"messages": "hi hello"},
    {"messages": [{"role": "user"}]},
    {"messages": [{"role": "tool", "content": "hi"}]},
]


def test_counts_malformed_empty_and_over_length_rows(capsys):
    report = check_dataset(ROWS, _tokenizer(), max_seq_length = 8)

    assert report["rows"] == 11
    assert report["malformed"] == 5
    assert report["empty"] == 1
    assert report["over_max_seq_length"] == 2
    assert report["max_seq_length"] == 8
    assert sorted(report["lengths"]) == [4, 6, 6, 8, 35, 36]
    assert report["response_marker_found"] == 6

    out = capsys.readouterr().out
    assert "Rows: 11  |  Malformed: 5  |  Empty: 1" in out
    assert "row 6: row is not a dict" in out
    assert "row 7: no messages, conversations or text column" in out
    assert "row 10: chat template failed: bad role" in out
    assert "Token lengths: min 4 | median 8 | p95 35 | max 36" in out
    assert "Over max_seq_length (8): 2 (18.2%)" in out
    assert (
        "Template: tiny/chat-tokenizer (tokenizer's own) | Response marker found in: 6 / 11" in out
    )


def test_hf_dataset_and_conversations_column():
    datasets = pytest.importorskip("datasets")
    rows = [{"conversations": r["messages"]} for r in ROWS[:6]]
    report = check_dataset(datasets.Dataset.from_list(rows), _tokenizer(), max_seq_length = 8)

    assert report["rows"] == 6
    assert report["malformed"] == 0
    assert report["empty"] == 1
    assert report["over_max_seq_length"] == 2
    assert report["response_marker_found"] == 6


def test_text_column_with_explicit_marker():
    tok = _tokenizer(template = None)
    rows = [
        {"text": "hi there ### Response: hello"},
        {"text": "hi ### Response: " + LONG},
        {"text": "hi hello"},
        {"text": "   "},
        {"text": 42},
    ]
    report = check_dataset(rows, tok, max_seq_length = 8, response_part = "### Response:")

    assert report["malformed"] == 1
    assert report["empty"] == 1
    assert report["over_max_seq_length"] == 1
    assert report["response_marker_found"] == 2
    assert sorted(report["lengths"]) == [0, 2, 5, 33]


def test_no_chat_template_makes_message_rows_malformed(capsys):
    report = check_dataset(ROWS[:2], _tokenizer(template = None), max_seq_length = 8)

    assert report["malformed"] == 2
    assert "Template: none" in capsys.readouterr().out


def test_get_chat_template_names_the_template(capsys):
    tok = get_chat_template(
        _tokenizer(), chat_template = "chatml", map_eos_token = False, patch_saving = False
    )
    report = check_dataset(ROWS[:3], tok, max_seq_length = 8)

    assert report["template"] == "chatml"
    assert report["response_marker_found"] == 3
    assert "Template: chatml | Response marker found in: 3 / 3" in capsys.readouterr().out


def test_sharegpt_from_value_rows_are_not_malformed():
    expected = check_dataset([_chat("hi", "hello")], _tokenizer(), max_seq_length = 8)["lengths"]
    rows = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": "  "}, {"from": "gpt", "value": ""}]},
    ]
    report = check_dataset(rows, _tokenizer(), max_seq_length = 8)
    assert report["malformed"] == 0
    assert report["empty"] == 1
    assert report["lengths"][:1] == expected

    tok = get_chat_template(
        _tokenizer(),
        chat_template = "chatml",
        mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token = False,
        patch_saving = False,
    )
    report = check_dataset(rows, tok, max_seq_length = 8)
    assert report["malformed"] == 0
    assert report["empty"] == 1


def test_row_tools_are_rendered_into_length():
    import json

    template = (
        "{% if tools %}{% for t in tools %}{{ t['name'] }} {% endfor %}{% endif %}" + TEMPLATE
    )
    row = _chat("hi", "hello")
    plain = check_dataset([row], _tokenizer(template), max_seq_length = 8)["lengths"][0]
    tools = [{"name": f"w{i}"} for i in range(10)]
    as_list = check_dataset([{**row, "tools": tools}], _tokenizer(template), max_seq_length = 8)
    as_str = check_dataset(
        [{**row, "tools": json.dumps(tools)}], _tokenizer(template), max_seq_length = 8
    )
    assert as_list["lengths"][0] == as_str["lengths"][0] == plain + 10
    assert as_list["over_max_seq_length"] == 1


class _ImagesFirstProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(
        self,
        images = None,
        text = None,
        **kwargs,
    ):
        if images is not None:
            raise ValueError("Incorrect image source")
        return self.tokenizer(text, **kwargs)


def test_vision_processor_measures_with_inner_tokenizer():
    tok = _tokenizer()
    rows = ROWS[:5]
    report = check_dataset(rows, _ImagesFirstProcessor(tok), max_seq_length = 8)
    assert report["lengths"] == check_dataset(rows, tok, max_seq_length = 8)["lengths"]
