# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from models.inference import ChatCompletionRequest


def test_chat_passthrough_refuses_an_empty_conversation():
    # A request whose every turn was pruned (or that sent none) used to reach
    # llama-server as messages: [] and fail inside the Jinja template; the shared
    # body builder now refuses it with a clear 400 first.
    from fastapi import HTTPException

    from routes import inference as routes_inference

    request = ChatCompletionRequest(model = "default", messages = [])
    try:
        routes_inference._build_openai_passthrough_body(request)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "No messages provided."
    else:
        raise AssertionError("expected a 400 for an empty conversation")


def test_chat_passthrough_allows_a_lone_question():
    # The guard must not reject a single normal turn: this is the shape every
    # Studio send takes, and it must still assemble into a llama-server body.
    from routes import inference as routes_inference

    request = ChatCompletionRequest(
        model = "default",
        messages = [{"role": "user", "content": "hi"}],
        stream = False,
    )
    body = routes_inference._build_openai_passthrough_body(request)
    assert [msg["role"] for msg in body["messages"]] == ["user"]