# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thought the context window cut before any answer (llama-server stops with `length`, no EOS).

The reply ends there, with the thought closed. The thread's next request (the chat's automatic
continuation, or the user's next message) is compacted instead of resuming the thought: the thought
goes to the thread's conversation archive, where search_conversation can find it, and the request
starts with a note saying so. One-shot, held in memory: a restart in between leaves today's behavior.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# Archived under its own role, so recall can tell it from a turn that must still be on the live
# branch: a thought is never part of the branch text the branch filter compares against.
THOUGHT_ROLE = "thinking"
THOUGHT_DOCUMENT = f"earlier turn ({THOUGHT_ROLE})"

NOTE_SEARCHABLE = (
    "The previous model stopped thinking before it answered, so re-check via "
    "search_conversation and continue work."
)
NOTE_NOT_SEARCHABLE = (
    "The previous model stopped thinking before it answered. Continue the work and answer the "
    "user's newest message."
)

# Long enough for the automatic continuation or a reply typed straight after; a later turn is a new topic.
_TTL_S = 3600.0

_lock = threading.Lock()
_pending: dict[str, tuple[float, str]] = {}


def mark(thread_id: Optional[str], thought: str) -> bool:
    """Records the cut thought for the thread's next request. Only for threads Studio persists, since
    recovery reads the thread back; an API client's next request is left as it was."""
    if not thread_id or not thought.strip() or not _persisted(thread_id):
        return False
    with _lock:
        _pending[thread_id] = (time.monotonic(), thought.strip())
    return True


def take(thread_id: Optional[str]) -> Optional[str]:
    """The thread's cut thought, once; None when there is none or it is stale."""
    if not thread_id:
        return None
    with _lock:
        entry = _pending.pop(thread_id, None)
    if entry is None or time.monotonic() - entry[0] > _TTL_S:
        return None
    return entry[1]


def archive(thread_id: str, thought: str) -> bool:
    """Stores the thought where search_conversation looks; True when it is there to be found."""
    try:
        from core.rag import conversation_archive
        conversation_archive.archive_turns(thread_id, [{"role": THOUGHT_ROLE, "content": thought}])
        return bool(conversation_archive.enabled() and conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- recovery still compacts; only the search goes
        return False


def _persisted(thread_id: str) -> bool:
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- unknown means "not ours": leave the next request alone
        return False


def _reset_for_tests() -> None:
    with _lock:
        _pending.clear()
