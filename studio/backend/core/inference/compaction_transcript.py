# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plain-text transcript of a compacted chat, linked from the <carried_forward> block.

Written to the chat's sandbox folder (where the python and terminal tools read) on every compacted
request, so it always holds the whole branch the client sent: the turns the reset dropped included.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_OPEN = "<carried_forward>"


def path_for(thread_id: str) -> Optional[str]:
    try:
        from core.inference.tools import _get_workdir
        name = "transcript-" + re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)[:120] + ".md"
        return os.path.join(_get_workdir(thread_id), name)
    except Exception:  # noqa: BLE001 -- no sandbox, no transcript; the search still works
        return None


def write(thread_id: Optional[str], messages: list[dict]) -> Optional[str]:
    """The branch as plain text; a thought the window cut is labelled and ends with the model's own
    closing marker. Returns the path, or None."""
    if not thread_id:
        return None
    path = path_for(thread_id)
    if path is None:
        return None
    from core.inference import cut_thought

    cut = cut_thought.recent(thread_id)
    try:
        lines = [f"# Chat transcript ({thread_id})", ""]
        shown = False
        for message in messages:
            role = str(message.get("role") or "")
            if role in ("system", "developer"):
                continue
            text = _text(message.get("content"))
            reasoning = message.get("reasoning_content")
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
            if cut is not None and reasoning == cut.thought:
                shown = True
                lines += _cut_section(role, cut)
            elif reasoning:
                lines += [f"## {role} (thinking)", "", reasoning, ""]
            if text:
                lines += [f"## {role}", "", text, ""]
        if cut is not None and not shown:
            lines += _cut_section("assistant", cut)
        with open(path, "w", encoding = "utf-8") as handle:
            handle.write("\n".join(lines))
        return path
    except Exception:  # noqa: BLE001 -- the search still works without it
        return None


def mention(path: str) -> str:
    return f"The whole conversation so far is also saved as a transcript at {path}."


def link_in_block(messages: list[dict], path: str) -> list[dict]:
    """Adds the transcript's path to the block's opening paragraph (not a quoted bullet, so it is
    never carried into the next epoch as one of the user's instructions). Idempotent."""
    sentence = mention(path)
    out = list(messages)
    for index, message in enumerate(out):
        content = message.get("content")
        if message.get("role") not in ("system", "developer") or not isinstance(content, str):
            continue
        start = content.find(_OPEN)
        if start == -1:
            continue
        if sentence in content:
            return out
        end = content.find("\n\n", start)
        if end == -1:
            end = content.find("\n</carried_forward>", start)
        if end == -1:
            return out
        out[index] = {**message, "content": content[:end] + " " + sentence + content[end:]}
        return out
    return out


def _cut_section(role: str, cut) -> list[str]:
    return [
        f"## {role} (thinking, stopped by the context window)",
        "",
        cut.thought + (cut.close_marker or ""),
        "",
    ]


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""
