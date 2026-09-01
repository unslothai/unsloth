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
DOCUMENT_PREVIEW_TSX = (
    REPO / "studio/frontend/src/features/rag/components/document-preview-sheet.tsx"
)
SHEET_TSX = REPO / "studio/frontend/src/components/ui/sheet.tsx"
REASONING_TSX = REPO / "studio/frontend/src/components/assistant-ui/reasoning.tsx"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CHAT_PREFS_TS = REPO / "studio/frontend/src/features/chat/stores/chat-preferences-store.ts"
CHAT_TAB_TSX = REPO / "studio/frontend/src/features/settings/tabs/chat-tab.tsx"
EN_LOCALE_TS = REPO / "studio/frontend/src/i18n/locales/en.ts"


def test_assistant_more_menu_exposes_response_details_action():
    src = THREAD_TSX.read_text(encoding = "utf-8")
    assert "MessageResponseDetailsSheet" in src
    assert "See response details" in src
    assert "setDetailsOpen(true)" in src


def test_response_details_sheet_uses_unsloth_sheet_and_key_sections():
    src = DETAILS_TSX.read_text(encoding = "utf-8")
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


def assert_sheet_close_button_tracks_title_center(src: str) -> None:
    content_start = src.index("<SheetContent")
    content_tail = src[content_start:]
    content_end = re.search(r"(?m)^\s*>\s*$", content_tail)
    assert content_end is not None
    content_open = content_tail[: content_end.end()]
    header = src[src.index("<SheetHeader") : src.index("</SheetHeader>")]
    close_start = header.index("<SheetCloseButton")
    close = header[close_start : header.index("/>", close_start)]
    class_name = re.search(r'className="([^"]+)"', close)

    assert "showCloseButton={false}" in content_open
    assert '<div className="relative">' in header
    assert class_name is not None
    class_tokens = class_name.group(1).split()
    for token in ["absolute", "top-1/2", "right-0", "-translate-y-1/2"]:
        assert token in class_tokens


def test_sheet_headers_center_the_shared_close_button_on_the_title():
    assert_sheet_close_button_tracks_title_center(
        DETAILS_TSX.read_text(encoding = "utf-8"),
    )
    assert_sheet_close_button_tracks_title_center(
        DOCUMENT_PREVIEW_TSX.read_text(encoding = "utf-8"),
    )

    sheet_src = SHEET_TSX.read_text(encoding = "utf-8")
    close_button = sheet_src[
        sheet_src.index("function SheetCloseButton") : sheet_src.index("function SheetPortal")
    ]
    assert 'variant="ghost"' in close_button
    assert 'size="icon-sm"' in close_button
    assert "Cancel01Icon" in close_button
    assert '<span className="sr-only">Close</span>' in close_button
    assert '<SheetCloseButton className="absolute top-4 right-4" />' in sheet_src


def test_response_model_badge_is_user_configurable_and_rendered_once_per_message():
    prefs_src = CHAT_PREFS_TS.read_text(encoding = "utf-8")
    chat_tab_src = CHAT_TAB_TSX.read_text(encoding = "utf-8")
    thread_src = THREAD_TSX.read_text(encoding = "utf-8")
    reasoning_src = REASONING_TSX.read_text(encoding = "utf-8")

    assert "showResponseModel: boolean" in prefs_src
    assert "showResponseModel: false" in prefs_src
    assert "showResponseModel: saved?.showResponseModel ?? false" in prefs_src
    # The visible label lives in the locale file; the tab holds only the key that resolves to it.
    assert 'showResponseModel: "Show response model"' in EN_LOCALE_TS.read_text(encoding = "utf-8")
    assert 't("settings.chat.showResponseModel")' in chat_tab_src
    assert "setShowResponseModel" in chat_tab_src
    details_src = DETAILS_TSX.read_text(encoding = "utf-8")
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


def test_reasoning_keeps_streaming_height_cap_through_automatic_collapse():
    src = REASONING_TSX.read_text(encoding = "utf-8")

    assert "const [retainStreamingHeight, setRetainStreamingHeight]" in src
    assert "setRetainStreamingHeight(false)" in src
    assert "setRetainStreamingHeight(isReasoningStreaming)" in src
    # Still zero while streaming, and still the animation's length once it stops.
    # cannot change what they animate and ANIMATION_DURATION is right for them, while
    assert "isReasoningStreaming ? 0 : closeDelay" in src
    assert "const closeDelay = GRID_COLLAPSE_REASONING_ENABLED" in src
    assert "? ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS" in src
    assert ": ANIMATION_DURATION;" in src
    assert "streaming={isReasoningStreaming || retainStreamingHeight}" in src


def test_reasoning_clears_manual_open_on_a_new_stream():
    """A hand-opened block must not stay pinned open when the stream restarts.

    isOpen is `(streaming && !dismissed) || manualOpen` and manualOpen is only
    settable while idle, so the new-stream reset has to clear it too.
    """
    src = REASONING_TSX.read_text(encoding = "utf-8")

    marker = "setDismissedWhileStreaming(false)"
    start = src.find(marker)
    assert start != -1, "new-stream reset effect is missing"
    effect = src[src.rfind("useEffect(() => {", 0, start) : src.find("});", start)]
    assert "setManualOpen(false)" in effect


def test_response_details_metadata_is_persisted_without_backend_schema_change():
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "interface ResponseDetailsMetadata" in src
    assert "buildResponseDetails" in src
    assert "responseDetails: buildResponseDetails(finishedAt)" in src
    assert "toolCalls: Array.from(" in src
    assert "!isExternalRequest && supportsTools && toolsEnabled" in src
    assert "!isExternalRequest && supportsTools && codeToolsEnabled" in src
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
