# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Normalize chat-message `content` (string or OpenAI multimodal list) to text.

String-only formatting paths called string ops directly on `content` and broke
on the list form (#4383). `content_to_text` collapses either shape to a string,
dropping non-text parts. No heavy imports, so it is unit-testable alone.
"""

from __future__ import annotations

from typing import Any

# The composer sends a long paste as a text attachment wrapped in this tag, so
# a paste-only turn carries no `content` text at all.
_PASTED_TEXT_OPEN = "<pasted_text name="
_PASTED_TEXT_CLOSE = "</pasted_text>"


def content_to_text(content: Any) -> str:
    """Plain text of a `content`: str unchanged, list/tuple text parts newline-joined
    (non-text dropped), None to "", else str(content)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
            elif isinstance(item, dict):
                # Skip non-text parts (image_url, input_audio, ...).
                part_type = item.get("type")
                if part_type is not None and part_type != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def pasted_text_body(text: str) -> str:
    """Body of a wrapped paste, or "" for anything else."""
    if not text.startswith(_PASTED_TEXT_OPEN):
        return ""
    body_start = text.find("\n")
    if body_start == -1:
        return ""
    body_end = len(text)
    closing = "\n" + _PASTED_TEXT_CLOSE
    if text.endswith(closing):
        body_end = max(len(text) - len(closing), body_start + 1)
    return text[body_start + 1 : body_end]


def message_text_with_pastes(message: Any) -> str:
    """Text of a message including any pasted attachment bodies.

    A paste above the composer's threshold is sent as an attachment rather than
    inline, so `content_to_text` alone reads a paste-only turn as empty. Other
    attachment types are left out: they were never part of a message's text.
    """
    if not isinstance(message, dict):
        return ""
    parts = [content_to_text(message.get("content"))]
    attachments = message.get("attachments")
    if isinstance(attachments, (list, tuple)):
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            body = pasted_text_body(content_to_text(attachment.get("content")))
            if body:
                parts.append(body)
    return "\n\n".join(part for part in parts if part)
