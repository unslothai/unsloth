# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import jinja2
import pytest
from pydantic import ValidationError

from core.training import rl_format
from core.training.rl_format import (
    REASONING_START,
    SYSTEM_PROMPT,
    apply_reasoning_template,
    build_warmup_dataset,
)
from models.training import TrainingStartRequest


class FakeTokenizer:
    eos_token = "</s>"
    chat_template = "original"

    def apply_chat_template(self, messages, tokenize = False, add_generation_prompt = False):
        return jinja2.Template(self.chat_template).render(
            messages = messages, eos_token = self.eos_token, add_generation_prompt = add_generation_prompt
        )

    def __call__(self, text, add_special_tokens = False):
        return {"input_ids": text.split()}


def _tok():
    tok = FakeTokenizer()
    apply_reasoning_template(tok)
    return tok


def test_template_matches_the_notebook_layout():
    tok = _tok()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], add_generation_prompt = True
    )
    assert text == SYSTEM_PROMPT + "</s>" + "What is 2+2?" + REASONING_START


def test_a_row_system_prompt_replaces_the_default():
    tok = _tok()
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert text == "Be brief.</s>qa</s>"


def test_processor_and_inner_tokenizer_both_get_the_template():
    class Processor:
        chat_template = "x"

        def __init__(self):
            self.tokenizer = FakeTokenizer()

    proc = Processor()
    apply_reasoning_template(proc)
    assert proc.chat_template == proc.tokenizer.chat_template == rl_format.CHAT_TEMPLATE


ROWS = [
    {"problem": "1+1?", "generated_solution": "<think>add</think>", "expected_answer": "2"},
    {"problem": "name it", "generated_solution": "x", "expected_answer": "a cat"},
    {"problem": "big", "generated_solution": "word " * 500, "expected_answer": "3"},
    {"problem": "2+2?", "generated_solution": "four", "expected_answer": "4"},
]


def test_warmup_rows_are_numeric_short_and_formatted():
    ds = build_warmup_dataset(_tok(), max_seq_length = 400, rows = 10, dataset = ROWS)
    assert len(ds) == 2
    first = ds[0]["text"]
    assert "<think>" not in first
    assert first.endswith(
        f"1+1?{REASONING_START}add<end_working_out><SOLUTION>2</SOLUTION></s>"
    )


def test_warmup_rows_are_capped():
    assert len(build_warmup_dataset(_tok(), 400, rows = 1, dataset = ROWS)) == 1


def test_nothing_fits_is_a_clear_error():
    with pytest.raises(ValueError, match = "max sequence length"):
        build_warmup_dataset(_tok(), max_seq_length = 4, rows = 10, dataset = ROWS)


def _request(**kw):
    base = dict(
        model_name = "unsloth/Qwen3-4B-Base",
        training_type = "LoRA/QLoRA",
        hf_dataset = "openai/gsm8k",
        format_type = "auto",
        objective = "grpo",
        grpo_rewards = [{"name": "r"}],
    )
    return TrainingStartRequest(**{**base, **kw})


def test_warmup_needs_the_reasoning_format():
    with pytest.raises(ValidationError, match = "reasoning format"):
        _request(grpo_format_warmup_steps = 50)
    ok = _request(grpo_format_warmup_steps = 50, grpo_reasoning_format = True, max_seq_length = 2048)
    assert ok.grpo_format_warmup_steps == 50


def test_warmup_needs_room_for_its_examples():
    with pytest.raises(ValidationError, match = "2048 or more"):
        _request(grpo_format_warmup_steps = 50, grpo_reasoning_format = True, max_seq_length = 1024)


def test_warmup_steps_are_bounded():
    with pytest.raises(ValidationError):
        _request(grpo_format_warmup_steps = 5000, grpo_reasoning_format = True)


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("add them<end_working_out><SOLUTION>1,234</SOLUTION>", [3.0, 3.0, 3.5]),
        ("add them<end_working_out><SOLUTION>12</SOLUTION>", [3.0, 0.0, -1.5]),
        ("no tags, 1234", [0.0, 0.0, -2.5]),
    ],
)
def test_bundled_solution_rewards_read_the_reasoning_format(reply, expected):
    from core.training.rewards import get_reward, preview_scores

    specs = [get_reward(n) for n in ("solution-format", "solution-exact", "solution-numeric")]
    scores, _ = preview_scores(specs, reply, reference = "So the total is #### 1234")
    assert [s["score"] for s in scores] == expected
