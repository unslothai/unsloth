# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Normalize chat-message `content` (string or OpenAI multimodal list) to text.

String-only formatting paths called string ops directly on `content` and broke
on the list form (#4383). `content_to_text` collapses either shape to a string,
dropping non-text parts. No heavy imports, so it is unit-testable alone.
"""

from __future__ import annotations

from typing import Any


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
