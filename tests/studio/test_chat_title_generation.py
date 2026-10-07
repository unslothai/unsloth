# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression checks for Unsloth chat title generation context."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
# Pinned as source text: the frontend runner cannot import JSX.
RUNTIME_TSX = REPO / "studio/frontend/src/features/chat/runtime-provider.tsx"
CHAT_ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"


def _source_until(src: str, anchor: str, end_anchor: str) -> str:
    start = src.find(anchor)
    assert start != -1, f"anchor {anchor!r} not found"
    end = src.find(end_anchor, start)
    assert end != -1, f"end anchor {end_anchor!r} not found"
    return src[start:end]


def _balanced_block(src: str, anchor: str) -> str:
    # Brace-counting only; assumes no unbalanced braces in strings, regexes, or comments.
    start = src.find(anchor)
    assert start != -1, f"anchor {anchor!r} not found"
    body_start = src.find("{", start)
    assert body_start != -1, f"body opener after {anchor!r} not found"

    depth = 0
    for index in range(body_start, len(src)):
        char = src[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return src[start : index + 1]
    raise AssertionError(f"unbalanced block after {anchor!r}")


def test_title_model_payload_includes_optional_assistant_reply():
    block = _source_until(
        RUNTIME_TSX.read_text(encoding = "utf-8"),
        "async function generateTitleWithModel",
        "\nconst inflightTitleByKey",
    )

    assert "assistantText?: string;" in block
    assert 'const assistant = clip(payload.assistantText ?? "", 384);' in block
    assert "const parts: string[] = [`User: ${user}`];" in block
    assert "if (assistant)" in block
    assert "parts.push(`Assistant: ${assistant}`);" in block
    assert 'parts.join("\\n")' in block


def test_generate_title_passes_first_assistant_reply_after_first_user():
    block = _balanced_block(
        RUNTIME_TSX.read_text(encoding = "utf-8"),
        "async generateTitle(remoteId",
    )

    assert 'const firstUserIndex = messages.findIndex((m) => m.role === "user");' in block
    assert 'm.role === "assistant" && i > firstUserIndex' in block
    assert "const assistantText = extractTextParts(firstAssistant);" in block


def test_tool_call_only_first_assistant_still_uses_first_user_message():
    source = RUNTIME_TSX.read_text(encoding = "utf-8")
    extract_block = " ".join(_balanced_block(source, "function extractTextParts").split())
    generate_block = " ".join(_balanced_block(source, "async generateTitle(remoteId").split())

    assert (
        '.filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")'
        in extract_block
    )
    assert (
        "const userText = titleTextOf(firstUser) || defaultTitle; const assistantText = extractTextParts(firstAssistant);"
        in generate_block
    )
    title_block = " ".join(_balanced_block(source, "function titleTextOf").split())
    assert "const text = extractTextParts(m);" in title_block
    assert (
        "(await generateTitleWithModel({ checkpoint: titleCheckpoint( answeredWith, "
        "useChatRuntimeStore.getState().params.checkpoint, ), userText, assistantText, })) "
        "|| fallbackTitleFromUserText(userText);" in generate_block
    )


def test_auto_title_disabled_uses_deterministic_user_text_fallback():
    block = _balanced_block(
        RUNTIME_TSX.read_text(encoding = "utf-8"),
        "async generateTitle(remoteId",
    )
    auto_title_off = _balanced_block(block, "if (!autoTitle)")

    assert "fallbackTitleFromUserText(userText)" in auto_title_off
    assert "generateTitleWithModel" not in auto_title_off


def test_reply_stamps_the_checkpoint_its_request_captured():
    source = " ".join(CHAT_ADAPTER_TS.read_text(encoding = "utf-8").split())

    assert source.count("modelId: params.checkpoint,") == 2
    assert "modelId: useChatRuntimeStore.getState().params" not in source
    assert "const externalRouting = resolveExternalRouting(params.checkpoint);" in source
    assert (
        "...(await buildExternalRoutingFields( { provider: externalProvider, "
        "modelId: externalModelId, apiKey: externalApiKey, }, "
        "{ forceRefreshPublicKey }, ))" in source
    )
    assert (
        'const externalProvider = externalRouting.kind === "external" '
        "? externalRouting.provider : null;" in source
    )
    assert (
        'const externalApiKey = externalRouting.kind === "external" '
        '? externalRouting.apiKey : "";' in source
    )
    assert (
        'const externalModelId = externalRouting.kind === "external" '
        '? externalRouting.modelId : "";' in source
    )
