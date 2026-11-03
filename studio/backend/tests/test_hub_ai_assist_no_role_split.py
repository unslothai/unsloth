# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AI Assist on datasets with no prompt/response split (#15901)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.schemas.datasets import AiAssistMappingRequest
from hub.services.datasets import formatting as dataset_formatting
from hub.utils import llm_assist

PASS1 = json.dumps(
    {
        "dataset_type": "other",
        "is_conversational": False,
        "needs_conversion": True,
        "description": "Prose passages.",
        "task_description": "Continued pretraining.",
    }
)
SYSTEM_PROMPT = "You answer questions about sloths accurately."


class _FakeBackend:
    replies: list[str] = []
    calls = 0
    loads = 0

    def load_model(self, intent):
        type(self).loads += 1
        return True

    def unload_model(self):
        pass

    def generate_chat_completion(self, **kwargs):
        type(self).calls += 1
        yield type(self).replies.pop(0)


@pytest.fixture
def backend(monkeypatch):
    module = types.ModuleType("core.inference.llama_cpp")
    module.LlamaCppBackend = _FakeBackend
    module.GgufLoadIntent = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "core.inference.llama_cpp", module)
    monkeypatch.delenv("UNSLOTH_HELPER_MODEL_DISABLE", raising = False)
    _FakeBackend.replies, _FakeBackend.calls, _FakeBackend.loads = [], 0, 0
    return _FakeBackend


@pytest.fixture
def warnings(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        llm_assist.logger, "warning", lambda msg, *args: seen.append(msg % args), raising = False
    )
    return seen


def _roles(roles, notes = "plain prose, nothing to answer"):
    return json.dumps({"column_roles": roles, "label_mapping": None, "notes": notes})


def _advise(columns):
    samples = [{col: "Sloths move slowly through the canopy." for col in columns}]
    return llm_assist.llm_conversion_advisor(columns, samples, dataset_name = "local.jsonl")


def test_single_column_skips_the_helper_model(backend):
    result = _advise(["content"])

    assert result == {"success": False, "warning": llm_assist.NO_ROLE_SPLIT_WARNING}
    assert backend.loads == 0


@pytest.mark.parametrize(
    "roles", [{"text": "assistant", "id": "skip"}, {"text": "skip", "id": "skip"}]
)
def test_text_plus_metadata_reports_no_split_and_logs_notes(backend, warnings, roles):
    backend.replies = [PASS1, _roles(roles)]

    result = _advise(["text", "id"])

    assert result == {"success": False, "warning": llm_assist.NO_ROLE_SPLIT_WARNING}
    assert backend.calls == 2
    assert any("plain prose, nothing to answer" in line for line in warnings)


def test_unparseable_pass2_is_logged(backend, warnings):
    backend.replies = [PASS1, "I am not sure"]

    assert (
        llm_assist._run_multi_pass_advisor(
            columns = ["question", "answer"],
            samples = [],
            dataset_name = None,
            dataset_card = None,
            dataset_metadata = None,
            model_name = None,
            model_type = None,
        )
        is None
    )
    assert any("I am not sure" in line for line in warnings)


def test_valid_split_still_maps(backend):
    backend.replies = [
        PASS1,
        _roles({"question": "user", "answer": "assistant", "ghost": "user"}),
        SYSTEM_PROMPT,
    ]

    result = _advise(["question", "answer"])

    assert result["success"] is True
    assert result["suggested_mapping"] == {"question": "user", "answer": "assistant"}
    assert result["system_prompt"] == SYSTEM_PROMPT


def test_rejected_split_falls_back_to_name_heuristics(backend):
    backend.replies = [PASS1, _roles({"question": "assistant", "answer": "skip"})]

    result = _advise(["question", "answer"])

    assert result["success"] is True
    assert result["suggested_mapping"] == {"question": "user", "answer": "assistant"}
    assert "Review the suggested mapping" in result["warning"]


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["text", "id"], None),
        (["content", "url", "source"], None),
        (["question_id", "question"], None),
        (["id", "question", "answer"], {"question": "user", "answer": "assistant"}),
        (["prompt", "chosen", "rejected"], {"prompt": "user", "rejected": "assistant"}),
        (["context", "summary"], {"context": "user", "summary": "assistant"}),
        (["a", "b"], {"a": "user", "b": "assistant"}),
    ],
)
def test_heuristic_never_assigns_metadata_columns(columns, expected):
    assert llm_assist._heuristic_mapping(columns) == expected


def test_service_returns_the_advisor_warning(monkeypatch):
    fake = types.ModuleType("hub.utils.llm_assist")
    fake.llm_conversion_advisor = lambda **kwargs: {"success": False, "warning": "specific reason"}
    monkeypatch.setitem(sys.modules, "hub.utils.llm_assist", fake)

    response = dataset_formatting.ai_assist_mapping_response(
        AiAssistMappingRequest(columns = ["content"], samples = [{"content": "x"}]),
    )

    assert response.success is False
    assert response.warning == "specific reason"
