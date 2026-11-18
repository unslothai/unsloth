# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A top-level llama-server ``grammar`` / ``json_schema`` on /v1/chat/completions must reach
llama-server on GGUF, and be refused where it cannot be applied, never silently dropped."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.inference import ChatCompletionRequest
from routes import inference as inference_route

_GRAMMAR = 'root ::= "yes" | "no"'
_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
_TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}


class _PlainBackend:
    supports_tools = False
    supports_tool_passthrough = False


def _payload(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model = "default", messages = [{"role": "user", "content": "hi"}], **fields
    )


def _refusal(payload) -> dict:
    with pytest.raises(HTTPException) as caught:
        inference_route._reject_unhonorable_raw_grammar(payload)
    assert caught.value.status_code == 400
    return caught.value.detail["error"]


@pytest.mark.parametrize("field, value", [("grammar", _GRAMMAR), ("json_schema", _SCHEMA)])
def test_passthrough_body_forwards_the_constraint(field, value):
    payload = _payload(**{field: value})

    assert inference_route._takes_tool_passthrough(payload, _PlainBackend()) is True
    inference_route._reject_unhonorable_raw_grammar(payload)
    body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)

    assert body[field] == value


def test_no_constraint_keeps_the_plain_path():
    for payload in (_payload(), _payload(grammar = ""), _payload(json_schema = None)):
        assert inference_route._takes_tool_passthrough(payload, _PlainBackend()) is False
        body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)
        assert "grammar" not in body and "json_schema" not in body


def test_tool_choice_none_lets_the_grammar_through():
    payload = _payload(grammar = _GRAMMAR, tools = [_TOOL], tool_choice = "none")

    inference_route._reject_unhonorable_raw_grammar(payload)


@pytest.mark.parametrize(
    "fields, param",
    [
        ({"grammar": _GRAMMAR, "json_schema": _SCHEMA}, "grammar"),
        ({"grammar": _GRAMMAR, "response_format": {"type": "json_object"}}, "grammar"),
        ({"json_schema": _SCHEMA, "tools": [_TOOL]}, "json_schema"),
        ({"grammar": _GRAMMAR, "tools": [_TOOL], "tool_choice": "auto"}, "grammar"),
        ({"grammar": ["root ::= x"]}, "grammar"),
        ({"json_schema": "object"}, "json_schema"),
    ],
)
def test_unhonorable_combinations_are_refused(fields, param):
    assert _refusal(_payload(**fields))["param"] == param


def test_npu_refuses_the_constraint():
    with pytest.raises(HTTPException) as caught:
        inference_route._refuse_npu_chat_request(
            _payload(grammar = _GRAMMAR), supports_vision = False, supports_tools = False
        )

    assert caught.value.detail["error"]["param"] == "grammar"
