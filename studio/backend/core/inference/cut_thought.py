# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thought the context window cut before any answer (llama-server stops with `length`, no EOS).

The reply ends there. The thread's next request (the chat's automatic continuation, or the user's
next message) does not resume the thought, which cannot fit: the thought, cut back a few tokens, is
kept in the conversation as a finished thought (the model's own chat template writes its closing
marker), the request is compacted as /compact does, the thought is archived where
search_conversation finds it, and the compacted request starts with a note saying the previous
model stopped thinking and where the transcript is (core.inference.compaction_transcript).

One-shot, held in memory: a restart in between leaves today's behavior.
"""

from __future__ import annotations

import dataclasses
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
_taken: dict[str, tuple[float, Cut]] = {}


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
        _prune()
        _pending[thread_id] = (time.monotonic(), Cut(thought.strip(), close_marker))
    return True


def _prune() -> None:
    """Drops expired entries; held under _lock. Without it every thread that ever cut keeps its thought."""
    now = time.monotonic()
    for cache in (_pending, _taken):
        for key in [key for key, (at, _) in cache.items() if now - at > _TTL_S]:
            del cache[key]


def take(thread_id: Optional[str]) -> Optional[Cut]:
    """The thread's cut thought, once; None when there is none or it is stale."""
    if not thread_id:
        return None
    with _lock:
        entry = _pending.pop(thread_id, None)
        if entry is None or time.monotonic() - entry[0] > _TTL_S:
            return None
        _prune()
        _taken[thread_id] = (time.monotonic(), entry[1])
    return entry[1]


def recent(thread_id: Optional[str]) -> Optional[Cut]:
    """The cut this thread's recovery took, while its requests still run."""
    if not thread_id:
        return None
    with _lock:
        entry = _taken.get(thread_id)
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


def window_budget(
    context_length: Optional[int],
    prompt_tokens: Optional[int],
    max_tokens: Optional[int] = None,
    requested: Optional[int] = None,
) -> Optional[int]:
    """thinking_budget_tokens that leaves the answer room in the window, so llama-server closes the
    thought with the model's own end tag instead of the window cutting it. None: send nothing."""
    if not context_length or prompt_tokens is None:
        return requested
    room = context_length - prompt_tokens
    # A smaller max_tokens stops the reply first; the window cannot cut the thought then.
    if room <= 0 or (max_tokens is not None and 0 < max_tokens < room):
        return requested
    budget = room - max(room // 4, min(room // 2, 1024))
    return budget if requested is None or requested < 0 else min(budget, requested)


def _persisted(thread_id: str) -> bool:
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- unknown means "not ours": leave the next request alone
        return False


def _reset_for_tests() -> None:
    with _lock:
        _pending.clear()
        _taken.clear()
