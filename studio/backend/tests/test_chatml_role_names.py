# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import format_and_template_dataset


class _Tokenizer:
    chat_template = "chatml"
    eos_token = "</s>"

    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
            if m["role"] in ("system", "user", "assistant", "tool")
        )


def _conversation(system, user, assistant):
    return [
        {"role": system, "content": "Be brief."},
        {"role": user, "content": "What is 2+2?"},
        {"role": assistant, "content": "4"},
    ]


def _render(rows, column, format_type):
    result = format_and_template_dataset(
        Dataset.from_dict({column: rows}),
        model_name = "Qwen3ForCausalLM",
        tokenizer = _Tokenizer(),
        format_type = format_type,
        batch_size = 1,
        num_proc = 1,
    )
    assert result["success"] is True
    return list(result["dataset"]["text"])


@pytest.mark.parametrize("column", ["messages", "conversations"])
@pytest.mark.parametrize("format_type", ["auto", "chatml"])
@pytest.mark.parametrize(
    "roles", [("System", "User", "Assistant"), (" system ", "USER ", " Assistant")]
)
def test_chatml_role_names_reach_the_template_in_lowercase(column, format_type, roles):
    expected = _render([_conversation("system", "user", "assistant")], column, format_type)

    assert _render([_conversation(*roles)], column, format_type) == expected
    assert "<|im_start|>assistant\n4" in expected[0]


@pytest.mark.parametrize("mapping", [None, {"prompt": "user", "completion": "assistant"}])
def test_mapped_message_columns_reach_the_template_in_lowercase(mapping):
    def render(user, assistant):
        result = format_and_template_dataset(
            Dataset.from_dict(
                {
                    "prompt": [[{"role": user, "content": "What is 2+2?"}]],
                    "completion": [[{"role": assistant, "content": "4"}]],
                }
            ),
            model_name = "Qwen3ForCausalLM",
            tokenizer = _Tokenizer(),
            format_type = "chatml" if mapping else "auto",
            custom_format_mapping = mapping,
            batch_size = 1,
            num_proc = 1,
        )
        assert result["success"] is True
        return list(result["dataset"]["text"])

    expected = render("user", "assistant")
    assert render("User", " Assistant ") == expected
    assert "<|im_start|>assistant\n4" in expected[0]


def test_lowercase_roles_skip_the_extra_pass():
    from utils.datasets.format_conversion import normalize_chatml_roles

    lower = Dataset.from_dict({"messages": [_conversation("system", "user", "assistant")]})
    assert normalize_chatml_roles(lower, "messages") is lower
    upper = Dataset.from_dict({"messages": [_conversation("System", "User", "Assistant")]})
    assert [m["role"] for m in normalize_chatml_roles(upper, "messages")["messages"][0]] == [
        "system",
        "user",
        "assistant",
    ]
