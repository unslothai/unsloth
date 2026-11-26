# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from datasets import Dataset

from models.training import TrainingStartRequest
from utils.datasets import format_and_template_dataset
from utils.datasets.chat_templates import get_training_chat_template


CUSTOM_TEMPLATE = (
    "{% for message in messages %}"
    "<{{ message['role'] }}>{{ message['content'] }}"
    "{% endfor %}{{ eos_token }}"
)


class _Tokenizer:
    eos_token = "<eos>"
    chat_template = None

    def apply_chat_template(self, conversation, **kwargs):
        if not self.chat_template:
            raise ValueError("chat_template is not set")
        if self.chat_template == CUSTOM_TEMPLATE:
            return (
                "".join(f"<{message['role']}>{message['content']}" for message in conversation)
                + self.eos_token
            )
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in conversation
        )


def _conversation_dataset():
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ]
        }
    )


def test_custom_chat_template_formats_chat_rows_and_is_saved():
    tokenizer = _Tokenizer()

    result = format_and_template_dataset(
        _conversation_dataset(),
        model_name = "unsloth/Qwen3.5-0.8B-Base",
        tokenizer = tokenizer,
        num_proc = 1,
        chat_template = CUSTOM_TEMPLATE,
    )

    assert result["success"], result["errors"]
    assert result["dataset"]["text"] == ["<user>Hi<assistant>Hello<eos>"]
    assert tokenizer.chat_template == CUSTOM_TEMPLATE


def test_custom_chat_template_wins_over_the_checkpoint_template():
    tokenizer = _Tokenizer()
    tokenizer.chat_template = "checkpoint-template"

    resolved = get_training_chat_template(
        tokenizer,
        "unsloth/Qwen3.5-0.8B-Base",
        "chatml_messages",
        CUSTOM_TEMPLATE,
    )

    assert resolved.chat_template == CUSTOM_TEMPLATE


def test_training_start_request_accepts_a_custom_chat_template():
    request = TrainingStartRequest(
        model_name = "unsloth/Qwen3.5-0.8B-Base",
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        chat_template = CUSTOM_TEMPLATE,
    )

    assert request.chat_template == CUSTOM_TEMPLATE


def test_blank_chat_template_is_treated_as_absent():
    request = TrainingStartRequest(
        model_name = "unsloth/Qwen3.5-0.8B-Base",
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        chat_template = "  \n",
    )

    assert request.chat_template is None
