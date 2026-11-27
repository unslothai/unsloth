# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest

tokenizers = pytest.importorskip("tokenizers")
from transformers import PreTrainedTokenizerFast, ProcessorMixin

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
    assert (
        "row 7: no messages, conversations, text, prompt/completion or instruction/output column"
        in out
    )
    assert "row 10: chat template failed: bad role" in out
    assert "Token lengths: min 4 | median 7 | p95 35 | max 36" in out
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


class _ImagesFirstProcessor(ProcessorMixin):
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


def test_tokenizer_with_backend_attribute_is_not_unwrapped():
    tok = _tokenizer()
    expected = check_dataset(ROWS[:5], tok, max_seq_length = 8)["lengths"]
    tok.tokenizer = object()
    assert check_dataset(ROWS[:5], tok, max_seq_length = 8)["lengths"] == expected


def _lengths(rows, tok):
    return check_dataset(rows, tok, max_seq_length = 8)["lengths"]


def test_alpaca_rows():
    tok = _tokenizer()
    rows = [
        {"instruction": "hi", "input": "there", "output": "hello"},
        {"instruction": "hi", "input": "", "output": LONG},
        {"instruction": "hi", "output": "hello"},
    ]
    report = check_dataset(rows, tok, max_seq_length = 8)

    assert report["malformed"] == 0
    assert report["over_max_seq_length"] == 1
    assert report["response_marker_found"] == 3
    assert report["lengths"] == _lengths(
        [_chat("hi\n\nthere", "hello"), _chat("hi", LONG), _chat("hi", "hello")], tok
    )


def test_prompt_completion_rows():
    tok = _tokenizer()
    rows = [
        {"prompt": "hi there", "completion": " hello"},
        {"prompt": "hi", "completion": " " + LONG},
        {"prompt": "", "completion": " "},
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "completion": [{"role": "assistant", "content": "hello"}],
        },
    ]
    report = check_dataset(rows, tok, max_seq_length = 8)

    assert report["malformed"] == 0
    assert report["empty"] == 1
    assert report["over_max_seq_length"] == 1
    assert report["response_marker_found"] == 1
    # SFT appends EOS to a string completion: "hi there hello<|im_end|>" is 4 tokens.
    assert report["lengths"] == [4, 32, 1] + _lengths([_chat("hi", "hello")], tok)


def test_unusable_prompt_completion_and_alpaca_rows_stay_malformed(capsys):
    rows = [
        {"prompt": "hi", "completion": [{"role": "assistant", "content": "hello"}]},
        {"instruction": "hi", "output": 42},
        {"instruction": "hi", "input": [], "output": "hello"},
    ]
    report = check_dataset(rows, _tokenizer(), max_seq_length = 8)

    assert report["malformed"] == 3
    assert report["lengths"] == []
    out = capsys.readouterr().out
    assert "row 0: prompt and completion are not both strings or both message lists" in out
    assert "row 1: instruction, input or output is not a string" in out


def test_median_of_an_even_number_of_rows(capsys):
    check_dataset([{"text": "hi"}, {"text": "hi hello"}], _tokenizer(), max_seq_length = 8)

    assert "Token lengths: min 1 | median 1.5 | p95 1 | max 2" in capsys.readouterr().out


class TokenizerWrapper:
    # Same forwarding as mlx_lm.tokenizer_utils.TokenizerWrapper, which defines no __call__.
    def __init__(self, tok):
        self._tokenizer = tok

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return self.__getattribute__(attr)
        return getattr(self._tokenizer, attr)


def test_mlx_tokenizer_wrapper_is_unwrapped():
    tok = get_chat_template(
        _tokenizer(), chat_template = "chatml", map_eos_token = False, patch_saving = False
    )
    report = check_dataset(ROWS[:3], TokenizerWrapper(tok), max_seq_length = 8)
    assert report["lengths"] == _lengths(ROWS[:3], tok)
    assert report["template"] == "chatml"


def test_replaced_template_is_not_reported_by_old_name():
    tok = get_chat_template(
        _tokenizer(), chat_template = "chatml", map_eos_token = False, patch_saving = False
    )
    tok.chat_template = TEMPLATE
    assert check_dataset(ROWS[:1], tok, max_seq_length = 8)["template"] != "chatml"


def test_preformatted_text_gets_no_second_bos():
    from tokenizers import processors

    tok = _tokenizer(template = None)
    tok.add_special_tokens({"bos_token": "<s>"})
    tok._tokenizer.post_processor = processors.TemplateProcessing(
        single = "<s> $A", special_tokens = [("<s>", tok.convert_tokens_to_ids("<s>"))]
    )
    assert _lengths([{"text": "<s> hi hello"}, {"text": "hi hello"}], tok) == [3, 3]


def test_media_rows_are_counted_but_not_measured(capsys):
    template = (
        "{% for m in messages %}{% if m['content'] is string %}{{ m['content'] }}"
        "{% else %}{% for c in m['content'] %}{{ c.get('text', '<image>') }} {% endfor %}"
        "{% endif %}{% endfor %}"
    )
    image_row = {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
    }
    tok = _tokenizer(template)
    report = check_dataset([image_row, _chat("hi", "hello")], tok, max_seq_length = 8)
    assert report["media"] == 1
    assert report["lengths"] == _lengths([_chat("hi", "hello")], tok)
    assert "Not measured: 1 rows with images, video or audio" in capsys.readouterr().out


def test_text_column_wins_over_conversations():
    rows = [{"conversations": _chat("hi", "hello")["messages"], "text": LONG}]
    assert _lengths(rows, _tokenizer()) == _lengths([{"text": LONG}], _tokenizer())


def test_text_column_wins_over_messages():
    rows = [{"messages": _chat("hi", "hello")["messages"], "text": LONG}]
    report = check_dataset(rows, _tokenizer(), max_seq_length = 8)
    assert report["lengths"] == _lengths([{"text": LONG}], _tokenizer())
    assert report["over_max_seq_length"] == 1


def test_prompt_completion_wins_over_text_and_messages():
    rows = [
        {"prompt": "hi", "completion": " hello", "text": LONG},
        {"prompt": "hi", "completion": " hello", **_chat("hi", LONG)},
    ]
    assert _lengths(rows, _tokenizer()) == [3, 3]


def test_processor_prompt_completion_gets_inner_eos():
    tok = _tokenizer()
    rows = [{"prompt": "hi", "completion": " hello"}]
    assert _lengths(rows, _ImagesFirstProcessor(tok)) == _lengths(rows, tok) == [3]


def test_assistant_tool_call_turn_without_content():
    template = (
        "{% for m in messages %}{{ m['role'] }} "
        "{% if m.get('tool_calls') %}{{ m['tool_calls'][0]['name'] }}{% else %}{{ m['content'] }}{% endif %}"
        " {% endfor %}"
    )
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"name": "w1"}]},
        ]
    }
    report = check_dataset([row], _tokenizer(template), max_seq_length = 8)
    assert report["malformed"] == 0
    assert report["empty"] == 0


def test_sharegpt_bad_role_is_malformed_and_extra_fields_render():
    template = "{% for m in messages %}{{ m['content'] }} {{ m.get('thinking', '') }} {% endfor %}"
    rows = [
        {"conversations": [{"from": ["human"], "value": "hi"}]},
        {"conversations": [{"from": "gpt", "value": "hi", "thinking": LONG}]},
    ]
    report = check_dataset(rows, _tokenizer(template), max_seq_length = 8)
    assert report["rows"] == 2
    assert report["over_max_seq_length"] == 1


def test_plain_text_rows_with_images_are_not_measured():
    rows = [
        {"text": "hi hello", "images": ["a.png"]},
        {"prompt": "hi", "completion": " hello", "image": "a.png"},
        {"text": "hi hello"},
    ]
    report = check_dataset(rows, _tokenizer(), max_seq_length = 8)
    assert report["media"] == 2
    assert report["lengths"] == [2]


def test_lengths_keep_row_order_across_batches():
    rows = [{"text": " ".join(["hi"] * (i % 7 + 1))} for i in range(2500)]
    rows[1500] = {"messages": _chat("hi", "hello")["messages"]}
    tok = _tokenizer()
    expected = [i % 7 + 1 for i in range(2500)]
    expected[1500] = _lengths(rows[1500:1501], tok)[0]
    assert check_dataset(rows, tok, max_seq_length = 8)["lengths"] == expected


def test_structured_blank_content_is_empty_and_sharegpt_media_is_flagged():
    template = "{% for m in messages %}{{ m['role'] }} {% endfor %}"
    blank = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": " "}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]
    }
    image = {"messages": [{"role": "user", "content": [{"type": "image"}]}]}
    report = check_dataset([blank, image], _tokenizer(template), max_seq_length = 8)
    assert report["empty"] == 1
    assert report["media"] == 1

    tok = get_chat_template(
        _tokenizer(),
        chat_template = "chatml",
        mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token = False,
        patch_saving = False,
    )
    tok.chat_template = tok.chat_template.replace("message['content']", "''")
    row = {"conversations": [{"from": "human", "value": [{"type": "image"}]}]}
    assert check_dataset([row], tok, max_seq_length = 8)["media"] == 1


def test_array_media_fields_and_openai_media_parts():
    np = pytest.importorskip("numpy")
    template = "{% for m in messages %}{{ m['role'] }} {% endfor %}"
    rows = [
        {"text": "hi", "image": np.zeros((2, 2))},
        {"text": "hi", "image": np.zeros((0,))},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "a"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]},
    ]
    report = check_dataset(rows, _tokenizer(template), max_seq_length = 8)
    assert report["rows"] == 4
    assert report["media"] == 3
