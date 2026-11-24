# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. See /studio/LICENSE.AGPL-3.0

"""`standardize_chat_format` must not swap role/content keys on a distinct-count tie."""

import sys
from pathlib import Path

from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.format_conversion import standardize_chat_format  # noqa: E402


def _canned_log(rows = 50):
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"speaker": "user", "utterance": "how do I reset my password"},
                    {"speaker": "agent", "utterance": "go to settings, then security"},
                ]
            }
            for _ in range(rows)
        ]
    )


def test_tied_cardinality_keeps_role_column():
    result = standardize_chat_format(_canned_log(), num_proc = 1)
    for message in result[0]["messages"]:
        assert message["role"] in ("user", "agent"), message
        assert message["content"] in (
            "how do I reset my password",
            "go to settings, then security",
        ), message
    assert result[0]["messages"][0]["role"] == "user"


def test_tied_cardinality_sharegpt_ordering():
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"body": "q?", "who": "human"},
                    {"body": "a.", "who": "gpt"},
                ]
            }
            for _ in range(50)
        ]
    )
    result = standardize_chat_format(dataset, num_proc = 1)
    roles = [m["role"] for m in result[0]["conversations"]]
    assert roles == ["user", "assistant"], roles


def test_cardinality_still_applies_without_alias_evidence():
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"side": "alpha", "line": f"line-{i}"},
                    {"side": "beta", "line": f"line-{i}-b"},
                ]
            }
            for i in range(20)
        ]
    )
    result = standardize_chat_format(dataset, num_proc = 1)
    roles = {m["role"] for m in result[0]["conversations"]}
    assert roles == {"alpha", "beta"}, roles


def test_single_alias_like_message_does_not_flip_custom_roles():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"speaker": "customer", "text": f"q{i}"},
                    {"speaker": "agent", "text": "Output" if i == 0 else f"a{i}"},
                ]
            }
            for i in range(30)
        ]
    )
    result = standardize_chat_format(dataset, num_proc = 1)
    assert [m["role"] for m in result[0]["messages"]] == ["customer", "agent"]
    assert result[0]["messages"][1]["content"] == "Output"


def test_alias_lines_in_text_do_not_outvote_lower_cardinality_roles():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"speaker": "customer", "text": "Human"},
                    {"speaker": "agent", "text": "Assistant"},
                    {"speaker": "customer", "text": "first prompt"},
                    {"speaker": "agent", "text": "second prompt"},
                ]
            }
        ]
    )
    result = standardize_chat_format(dataset, num_proc = 1)
    roles = [m["role"] for m in result[0]["messages"]]
    assert roles == ["customer", "agent", "customer", "agent"], roles
