# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Role detection on columns whose names contain a shorter role word: "text" inside
"context" used to win the user slot, sending the passage to the user turn and the
question to the system prompt."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402

_HEURISTICS = [
    pytest.param(format_detection.detect_custom_format_heuristic, id = "utils"),
    pytest.param(dataset_format.detect_custom_format_heuristic, id = "hub"),
]

_LONG = "x" * 600
_MID = "y" * 120

_CASES = [
    (
        {"context": _LONG, "question": _MID, "answer": _MID},
        {"question": "user", "context": "system", "answer": "assistant"},
    ),
    (
        {"id": "1", "title": "t", "context": _LONG, "question": _MID, "answers": _MID},
        {"question": "user", "context": "system", "answers": "assistant"},
    ),
    (
        {"instruction": _MID, "context": _LONG, "response": _MID, "category": "qa"},
        {"instruction": "user", "context": "system", "response": "assistant"},
    ),
    (
        {"retrieved_contexts": _LONG, "question": _MID, "ground_truth_answer": _MID},
        {"question": "user", "retrieved_contexts": "system", "ground_truth_answer": "assistant"},
    ),
    (
        {"context": _LONG, "response": _MID},
        {"context": "user", "response": "assistant"},
    ),
    (
        {"instruction": _MID, "input": _MID, "output": _MID},
        {"instruction": "user", "input": "system", "output": "assistant"},
    ),
    (
        {"user_input": _MID, "assistant_output": _MID},
        {"user_input": "user", "assistant_output": "assistant"},
    ),
    (
        {"input_text": _MID, "target_text": _MID},
        {"input_text": "user", "target_text": "assistant"},
    ),
    (
        {"userInput": _MID, "assistantResponse": _MID},
        {"userInput": "user", "assistantResponse": "assistant"},
    ),
    (
        {"inputs": _MID, "targets": _MID},
        {"inputs": "user", "targets": "assistant"},
    ),
    (
        {"fulltext": _LONG, "answer": _MID},
        {"fulltext": "user", "answer": "assistant"},
    ),
    (
        {"subtask": _MID, "answer": _MID},
        {"subtask": "user", "answer": "assistant"},
    ),
    (
        {"task": _MID, "input": _MID, "output": _MID},
        {"input": "user", "task": "system", "output": "assistant"},
    ),
    (
        {"context": "Background.", "prompt": "Summarize.", "question": "Why?", "answer": _MID},
        {"context": "system", "prompt": "user", "answer": "assistant"},
    ),
    (
        {"context": _LONG, "response_text": _MID},
        {"context": "user", "response_text": "assistant"},
    ),
]


# An assistant column must never be drafted into the user turn just because the shadow
# rule emptied the user candidates: the context column is the better user turn.
_ASSISTANT_LEFTOVER_CASES = [
    (
        {"context": _LONG, "answer": _MID, "explanation": _MID},
        {"answer": "assistant", "context": "user"},
    ),
    (
        {"context": _LONG, "answer": _MID, "output": _MID},
        {"answer": "assistant", "context": "user"},
    ),
    (
        {"context": _LONG, "response": _MID, "target": _MID},
        {"response": "assistant", "context": "user"},
    ),
]

# A column that is only ever a system word stays unmapped, so the caller keeps asking
# for a manual mapping instead of silently training the system prompt as the user turn.
_NO_USER_COLUMN_ROWS = [
    {"system": _MID, "output": _MID},
    {"system_prompt": _MID, "output": _MID},
    {"system_prompt": _MID, "context_id": "c1", "output": _MID},
    {"persona": _MID, "reply": _MID},
    {"role": _MID, "response": _MID},
    {"template": _MID, "output": _MID},
    {"question_type": "causal", "answer": _MID},
    {"answer": _MID, "explanation": _MID},
]


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row, expected", _CASES)
def test_context_column_is_not_matched_as_text(heuristic, row, expected):
    assert heuristic([row]) == expected


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row, expected", _ASSISTANT_LEFTOVER_CASES)
def test_assistant_column_is_not_promoted_to_the_user_turn(heuristic, row, expected):
    assert heuristic([row]) == expected


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row", _NO_USER_COLUMN_ROWS)
def test_system_only_column_is_not_promoted_to_the_user_turn(heuristic, row):
    assert heuristic([row]) is None


_ANSWER_LEFTOVER_CASES = [
    (
        {"problem": _MID, "generated_solution": _LONG, "expected_answer": "14"},
        {"problem": "user", "generated_solution": "assistant"},
    ),
    (
        {"question": _MID, "solution": _LONG, "final_answer": "42"},
        {"question": "user", "solution": "assistant"},
    ),
    (
        {"instruction": _MID, "response_base": _LONG, "response": _MID},
        {"instruction": "user", "response_base": "assistant"},
    ),
    (
        {"input": _MID, "output": _LONG, "target": "42"},
        {"input": "user", "output": "assistant"},
    ),
]


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row, expected", _ANSWER_LEFTOVER_CASES)
def test_answer_column_is_not_mapped_to_the_system_prompt(heuristic, row, expected):
    assert heuristic([row]) == expected


_LEFTOVER_CASES = [
    (
        {"type": "GSM_SV", "query": _MID, "original_question": _MID, "response": _LONG},
        {"query": "user", "response": "assistant"},
    ),
    (
        {"id": "1", "input": _MID, "output": _LONG, "source": "cf", "license": "cc-by-4.0"},
        {"input": "user", "output": "assistant"},
    ),
    (
        {"prompt": _MID, "response": _LONG, "helpfulness": 3, "correctness": 3},
        {"prompt": "user", "response": "assistant"},
    ),
    (
        {"question": _MID, "distractor3": "viruses", "correct_answer": _MID, "support": _LONG},
        {"question": "user", "correct_answer": "assistant"},
    ),
    (
        {"Question": _MID, "Complex_CoT": _LONG, "Response": _MID},
        {"Question": "user", "Response": "assistant"},
    ),
    (
        {"task_id": "HumanEval/0", "prompt": _LONG, "canonical_solution": _MID, "test": _LONG},
        {"prompt": "user", "canonical_solution": "assistant"},
    ),
    (
        {"Instruction": "Translate this.", "Input": "It is sunny.", "Output": "Il fait beau."},
        {"Instruction": "user", "Input": "system", "Output": "assistant"},
    ),
    (
        {"instruction": "Translate this.", "input": "It is sunny.", "response": "Il fait beau."},
        {"instruction": "user", "input": "system", "response": "assistant"},
    ),
    (
        {"question": _MID, "answer": "true", "passage": _LONG},
        {"question": "user", "answer": "assistant", "passage": "system"},
    ),
]


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row, expected", _LEFTOVER_CASES)
def test_leftover_column_is_not_mapped_to_the_system_prompt(heuristic, row, expected):
    assert heuristic([row]) == expected


_CONTEXT_CASES = [
    (
        {"id": "f.1", "system_prompt": _LONG, "question": _MID, "response": _LONG},
        {"question": "user", "system_prompt": "system", "response": "assistant"},
    ),
    (
        {"task_id": "t1", "system": "Be brief.", "prompt": _MID, "response": _MID},
        {"prompt": "user", "system": "system", "response": "assistant"},
    ),
    (
        {"index": 0, "question": _MID, "text": _LONG, "answer": "Yes"},
        {"question": "user", "text": "system", "answer": "assistant"},
    ),
    (
        {"article": _LONG, "question": _MID, "options": ["a", "b"], "answer": 1},
        {"article": "system", "question": "user", "answer": "assistant"},
    ),
    (
        {"task_name": "task001_quoref", "definition": _LONG, "inputs": _MID, "targets": _MID},
        {"definition": "system", "inputs": "user", "targets": "assistant"},
    ),
    (
        {"prompt": _MID, "response": _LONG, "input_tokens": 12, "passage_id": "p1"},
        {"prompt": "user", "response": "assistant"},
    ),
    (
        {"question": _MID, "answer": _MID, "input_ids": [1, 2], "background": _LONG},
        {"question": "user", "answer": "assistant", "background": "system"},
    ),
    (
        {"text": _LONG, "question_type": "causal", "answer": "yes"},
        {"text": "user", "answer": "assistant"},
    ),
]


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row, expected", _CONTEXT_CASES)
def test_system_and_context_columns_keep_the_system_prompt(heuristic, row, expected):
    assert heuristic([row]) == expected


@pytest.mark.parametrize("heuristic", _HEURISTICS)
def test_identifier_column_is_not_the_user_turn(heuristic):
    assert heuristic([{"task_id": "HumanEval/0", "canonical_solution": _MID}]) is None


_SYSTEM_METADATA_CASES = [
    {"question": _MID, "answer": _MID, "system_id": "s1"},
    {"question": _MID, "answer": _MID, "context_id": "c1"},
    {"question": _MID, "answer": _MID, "systemId": "s1"},
    {"question": _MID, "answer": _MID, "contextId": "c1"},
    {"question": _MID, "answer": _MID, "systemID": "s1"},
    {"question": _MID, "answer": _MID, "contextID": "c1"},
    {"question": _MID, "answer": _MID, "context_length": 4096},
]


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize("row", _SYSTEM_METADATA_CASES)
def test_system_metadata_is_not_mapped_to_the_system_prompt(heuristic, row):
    assert heuristic([row]) == {"question": "user", "answer": "assistant"}
