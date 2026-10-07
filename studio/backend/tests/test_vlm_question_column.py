# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset
from PIL import Image

from utils.datasets import format_and_template_dataset, llm_assist

HELPER_SENTENCE = "Solve the math problem shown in the image."


@pytest.fixture(autouse = True)
def _helper_writes_one_sentence(monkeypatch):
    monkeypatch.setattr(
        llm_assist,
        "llm_generate_vlm_instruction",
        lambda **kwargs: {"instruction": HELPER_SENTENCE, "confidence": 0.85},
    )


def _format(rows, mapping = None):
    images = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(2)]
    ds = Dataset.from_dict({"image": images, **rows})
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "org/ds",
        custom_format_mapping = mapping,
    )
    assert info["success"], info["errors"]
    return info["dataset"]


def _turns(sample):
    user, assistant = sample["messages"]
    user_text = next(part["text"] for part in user["content"] if part["type"] == "text")
    return user_text, assistant["content"][0]["text"]


@pytest.mark.parametrize(
    "question_col, answer_col, mapping",
    [
        ("problem", "solution", True),
        ("problem", "answer", False),
        ("Question", "Answer", False),
        ("inputs", "outputs", True),
        ("input", "output", False),
    ],
)
def test_each_row_keeps_its_own_question(question_col, answer_col, mapping):
    questions = ["Find x in the triangle.", "How many red cubes are left?"]
    answers = ["x = 42", "3"]
    out = _format(
        {question_col: questions, answer_col: answers},
        {"image": "image", answer_col: "text"} if mapping else None,
    )

    assert [_turns(sample) for sample in out] == list(zip(questions, answers))


def test_answer_column_is_not_reused_as_the_question():
    prompts = ["a red square", "a dark red square"]
    out = _format({"prompt": prompts}, {"image": "image", "prompt": "text"})

    assert [_turns(sample) for sample in out] == [(HELPER_SENTENCE, p) for p in prompts]


def test_exact_question_column_wins_over_a_case_variant():
    questions = ["Find x in the triangle.", "How many red cubes are left?"]
    answers = ["x = 42", "3"]
    out = _format(
        {"Question": ["upper one", "upper two"], "question": questions, "answer": answers}
    )

    assert [_turns(sample) for sample in out] == list(zip(questions, answers))
