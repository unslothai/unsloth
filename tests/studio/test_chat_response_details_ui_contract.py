# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for the chat response-details action and metadata."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
DETAILS_TSX = (
    REPO / "studio/frontend/src/components/assistant-ui/message-response-details-sheet.tsx"
)
REASONING_TSX = REPO / "studio/frontend/src/components/assistant-ui/reasoning.tsx"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CHAT_PREFS_TS = REPO / "studio/frontend/src/features/chat/stores/chat-preferences-store.ts"
CHAT_TAB_TSX = REPO / "studio/frontend/src/features/settings/tabs/chat-tab.tsx"


def test_assistant_more_menu_exposes_response_details_action():
    src = THREAD_TSX.read_text()
    assert "MessageResponseDetailsSheet" in src
    assert "See response details" in src
    assert "setDetailsOpen(true)" in src


def test_response_details_sheet_uses_unsloth_sheet_and_key_sections():
    src = DETAILS_TSX.read_text()
    assert "SheetContent" in src
    assert "Response details" in src
    assert "MessageResponseModelBadge" in src
    assert "showResponseModel" in src
    assert "ChipIcon" not in src
    assert "s.params.checkpoint" not in src
    assert "Not recorded" in src
    assert "min-w-0 break-words font-heading" in src
    assert "toolCallsFromContent(message.content)" in src
    assert 'label="Called"' in src
    for section in ["Response", "Tokens", "Timing", "Tools"]:
        assert f'title="{section}"' in src
    for field in ["Model", "Provider", "Total", "Cache hits", "Enabled", "Called"]:
        assert f'label="{field}"' in src


def test_response_model_badge_is_user_configurable_and_rendered_once_per_message():
    prefs_src = CHAT_PREFS_TS.read_text()
    chat_tab_src = CHAT_TAB_TSX.read_text()
    thread_src = THREAD_TSX.read_text()
    reasoning_src = REASONING_TSX.read_text()

    assert "showResponseModel: boolean" in prefs_src
    assert "showResponseModel: false" in prefs_src
    assert "showResponseModel: saved?.showResponseModel ?? false" in prefs_src
    assert "Show response model" in chat_tab_src
    assert "setShowResponseModel" in chat_tab_src
    details_src = DETAILS_TSX.read_text()
    assert (
        "aui-response-model-badge pointer-events-none relative inline-flex min-h-5" in details_src
    )
    assert "cursor-text select-text" in details_src
    assert "leading-5" in details_src
    assert "after:top-full after:h-1" in details_src
    assert "hover:opacity-100" in details_src
    assert "group-hover/assistant-message:opacity-100" in details_src
    # Pointer events gated behind hover/focus so the hidden badge stays inert when idle.
    assert "group-hover/assistant-message:pointer-events-auto" in details_src
    assert "group-focus-within/assistant-message:pointer-events-auto" in details_src
    assert thread_src.count("<MessageResponseModelBadge") == 1
    assert "hasReasoningParts" not in thread_src
    assert "group/assistant-message aui-assistant-message-root" in thread_src
    assert "pointer-events-none relative h-0" in thread_src
    assert "MessageResponseModelBadge" not in reasoning_src
    assert 'className="min-w-0 flex-1"' in reasoning_src


def test_response_details_metadata_is_persisted_without_backend_schema_change():
    src = ADAPTER_TS.read_text()
    assert "interface ResponseDetailsMetadata" in src
    assert "buildResponseDetails" in src
    assert "responseDetails: buildResponseDetails(finishedAt)" in src
    assert "toolCalls: Array.from(" in src
    # Response-details metadata: Search/Code/MCP can come from hosted builtins
    # OR from Unsloth's local tool runtime on OAI-compat Connections (#7282).
    assert "localWebSearchEnabledForThisTurn" in src
    assert "localCodeToolsEnabledForThisTurn" in src
    assert "localMcpEnabledForThisTurn" in src
    assert "!isExternalRequest && supportsTools && toolsEnabled" in src
    assert "!isExternalRequest && supportsTools && codeToolsEnabled" in src
    assert "providerSupportsLocalToolRuntime" in src
    assert re.search(r"selectedModelSummary\?\.name\s*\|\|\s*responseModelId", src)
    assert "providerName" in src
    assert "cancelId" in src
    metadata_block = src[
        src.find("interface ResponseDetailsMetadata") : src.find("type RunMessages")
    ]
    builder_block = src[
        src.find("const buildResponseDetails") : src.find("const externalCapabilities")
    ]
    for forbidden in [
        "encrypted_api_key",
        "externalApiKey",
        "apiKey",
        "providerKey",
        "secret",
    ]:
        assert forbidden not in metadata_block
        assert forbidden not in builder_block
