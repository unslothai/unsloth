# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for normalizing chat-message `content` to plain text.

Studio receives message `content` in two shapes:

  - the legacy string form: ``"hello"``
  - the OpenAI multimodal list form:
    ``[{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {...}}]``

Several string-only formatting paths (vision prompt extraction, the manual
chat-template formatters, the text `format_chat_prompt`) call ``.strip()`` /
``re.sub()`` / f-string interpolation directly on `content`. When handed the list
form they raise ``'list' object has no attribute 'replace'`` / ``'strip'`` (see
issue #4383) or render the Python list repr into the prompt.

``content_to_text`` collapses either shape to a plain string by concatenating the
text parts (image / audio / other non-text parts are dropped, since these paths
handle media separately). It is a deliberate no-op for plain strings, so the
common path is unchanged. This module intentionally has no heavy imports so it
can be unit-tested without loading torch/unsloth.
"""

from __future__ import annotations

from typing import Any


def content_to_text(content: Any) -> str:
    """Return the plain-text portion of an OpenAI-style message `content`.

    - ``str`` -> returned unchanged.
    - ``list``/``tuple`` -> the ``text`` of each text part (and bare-string items)
      joined by a single space; non-text parts (image_url, input_audio, ...) are
      dropped.
    - ``None`` -> ``""``.
    - anything else -> ``str(content)``.
    """
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
                # Skip explicit non-text parts (image_url, input_audio, ...).
                part_type = item.get("type")
                if part_type is not None and part_type != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts)
    return str(content)
