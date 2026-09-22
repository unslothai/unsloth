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
        {"question": "user", "context": "system", "title": "system", "answers": "assistant"},
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
        {"answer": "assistant", "context": "user", "explanation": "system"},
    ),
    (
        {"context": _LONG, "answer": _MID, "output": _MID},
        {"answer": "assistant", "context": "user", "output": "system"},
    ),
    (
        {"context": _LONG, "response": _MID, "target": _MID},
        {"response": "assistant", "context": "user", "target": "system"},
    ),
]

# A column that is only ever a system word stays unmapped, so the caller keeps asking
# for a manual mapping instead of silently training the system prompt as the user turn.
_NO_USER_COLUMN_ROWS = [
    {"system": _MID, "output": _MID},
    {"persona": _MID, "reply": _MID},
    {"role": _MID, "response": _MID},
    {"template": _MID, "output": _MID},
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
