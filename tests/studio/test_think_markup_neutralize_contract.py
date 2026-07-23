# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frontend contract for #7066 think-markup neutralization."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARSE_TS = REPO / "studio/frontend/src/features/chat/utils/parse-assistant-content.ts"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"


def test_frontend_exports_neutralize_think_markup():
    src = PARSE_TS.read_text(encoding = "utf-8")
    assert "export function neutralizeThinkMarkup" in src
    assert "export function drainThinkMarkupBuffer" in src
    assert "\\u200b" in src or "\u200b" in src
    assert "#7066" in src


def test_chat_adapter_neutralizes_reasoning_before_think_wrap():
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "drainThinkMarkupBuffer" in src
    assert "reasoningMarkupBuffer" in src
    assert "safeReasoning" in src
