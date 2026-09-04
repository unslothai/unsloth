# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from pathlib import Path


CHAT_API = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "chat"
    / "api"
    / "chat-api.ts"
)


def test_length_detection_classifies_visible_and_reasoning_content():
    source = CHAT_API.read_text(encoding = "utf-8")

    assert "return value.trim().length > 0;" in source
    assert 'record.type === "thinking" || record.type === "reasoning"' in source
    assert 'record.type === "text" || record.type === "output_text"' in source
    assert "sawAssistantContent ||= contentState.hasAssistantContent;" in source
    assert "sawReasoningContent ||= contentState.hasReasoningContent;" in source
