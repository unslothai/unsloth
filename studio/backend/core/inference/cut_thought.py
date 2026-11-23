# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thought the context window cut before any answer (llama-server stops with `length`, no EOS).

The reply ends there. The thread's next request (the chat's automatic continuation, or the user's
next message) does not resume the thought, which cannot fit: the thought, cut back a few tokens, is
kept in the conversation as a finished thought (the model's own chat template writes its closing
marker), the request is compacted as /compact does, the thought is archived where
search_conversation finds it, a transcript is written to the chat's sandbox folder, and the
compacted request starts with a note saying the previous model stopped thinking.

One-shot, held in memory: a restart in between leaves today's behavior.
"""

from __future__ import annotations

import dataclasses
import os
import re
import threading
import time
from typing import Callable, Optional

# Archived under its own role, so recall can tell it from a turn that must still be on the live
# branch: a thought is never part of the branch text the branch filter compares against.
THOUGHT_ROLE = "thinking"
THOUGHT_DOCUMENT = f"earlier turn ({THOUGHT_ROLE})"
# Tokens dropped from the end of the cut thought: the window can stop it mid-word.
BACKTRACK_TOKENS = 8

# Long enough for the automatic continuation or a reply typed straight after; a later turn is a new topic.
_TTL_S = 3600.0
_REASONING_PROBE = "\u2063R\u2063"
_ANSWER_PROBE = "\u2063A\u2063"


@dataclasses.dataclass(frozen = True)
class Cut:
    thought: str
    # The model's own closing marker (e.g. "</think>", "<channel|>"), for the transcript.
    close_marker: str = ""


_lock = threading.Lock()
_pending: dict[str, tuple[float, Cut]] = {}


def mark(
    thread_id: Optional[str],
    thought: str,
    close_marker: str = "",
) -> bool:
    """Records the cut thought for the thread's next request. Only for threads Studio persists, since
    recovery reads the thread back; an API client's next request is left as it was."""
    if not thread_id or not thought.strip() or not _persisted(thread_id):
        return False
    with _lock:
        _pending[thread_id] = (time.monotonic(), Cut(thought.strip(), close_marker))
    return True


def take(thread_id: Optional[str]) -> Optional[Cut]:
    """The thread's cut thought, once; None when there is none or it is stale."""
    if not thread_id:
        return None
    with _lock:
        entry = _pending.pop(thread_id, None)
    if entry is None or time.monotonic() - entry[0] > _TTL_S:
        return None
    return entry[1]


def close_marker_from_template(render: Callable[[list[dict]], Optional[str]]) -> str:
    """The text a chat template writes between a finished thought and the answer, read by rendering
    one probe turn, so no per-model table is needed ("\\n</think>\\n\\n" for Qwen3,
    "\\n<channel|>" for Gemma 4, "<|end_message|><|message_model|><|content_text|>" for Inkling 2).
    Empty when the template does not render the thought."""
    try:
        rendered = render(
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": _ANSWER_PROBE,
                    "reasoning_content": _REASONING_PROBE,
                },
            ]
        )
    except Exception:  # noqa: BLE001 -- the marker is cosmetic: the template closes the thought anyway
        return ""
    if not isinstance(rendered, str):
        return ""
    start = rendered.find(_REASONING_PROBE)
    end = rendered.find(_ANSWER_PROBE, start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        return ""
    return rendered[start + len(_REASONING_PROBE) : end]


def archive(thread_id: str, thought: str) -> bool:
    """Stores the thought where search_conversation looks; True when it is there to be found."""
    try:
        from core.rag import conversation_archive
        conversation_archive.archive_turns(thread_id, [{"role": THOUGHT_ROLE, "content": thought}])
        return bool(conversation_archive.enabled() and conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- recovery still compacts; only the search goes
        return False


def write_transcript(
    session_id: Optional[str], thread_id: str, messages: list[dict], cut: Cut
) -> Optional[str]:
    """This chat's branch as plain text in its sandbox folder (where the python and terminal tools
    read), ending with the cut thought and its closing marker. Returns the path, or None."""
    try:
        from core.inference.tools import _get_workdir

        folder = _get_workdir(session_id or thread_id)
        name = "transcript-" + re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)[:120] + ".md"
        path = os.path.join(folder, name)
        lines = [f"# Chat transcript ({thread_id})", ""]
        for message in messages:
            role = str(message.get("role") or "")
            if role in ("system", "developer"):
                continue
            text = _text(message.get("content"))
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                lines += [f"## {role} (thinking)", "", reasoning.strip(), ""]
            if text:
                lines += [f"## {role}", "", text, ""]
        lines += [
            "## assistant (thinking, stopped by the context window)",
            "",
            cut.thought + (cut.close_marker or ""),
            "",
        ]
        with open(path, "w", encoding = "utf-8") as handle:
            handle.write("\n".join(lines))
        return path
    except Exception:  # noqa: BLE001 -- the search still works without it
        return None


def note(searchable: bool, transcript_path: Optional[str]) -> str:
    where = []
    if searchable:
        where.append("re-check via search_conversation")
    if transcript_path:
        where.append(f"read the transcript at {transcript_path}")
    if not where:
        return (
            "The previous model stopped thinking before it answered. Continue the work and "
            "answer the user's request."
        )
    return f"The previous model stopped thinking, so {' or '.join(where)} and continue work."


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


def _persisted(thread_id: str) -> bool:
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- unknown means "not ours": leave the next request alone
        return False


def _reset_for_tests() -> None:
    with _lock:
        _pending.clear()
