# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Wiring for the legacy chat title repair, where a unit test cannot reach.

The repair module pulls in the chat API, so these are source checks on the
seams: which map it reads from, and what keeps it going page to page.
"""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src/features/chat"
REPAIR = FRONTEND / "utils/repair-legacy-chat-titles.ts"
HOOK = FRONTEND / "hooks/use-chat-sidebar-items.ts"
STORAGE = FRONTEND / "utils/chat-history-storage.ts"


def _read(path: Path) -> str:
    return " ".join(path.read_text(encoding = "utf-8").split())


def test_the_sidebar_hands_its_messages_to_the_repair():
    """The with-messages load already fetched every thread's messages, so the
    repair must not go and fetch the same rows a second time."""
    storage = _read(STORAGE)
    assert "export interface ChatThreadsWithMessages" in storage
    assert (
        "messagesByThreadId: new Map( entries .filter((e): e is typeof e & { messages: MessageRecord[] } => e.messages !== null, )"
        in storage
    )
    # A read that failed is unknown. Reporting it as an empty chat would let the
    # repair write the row off for the session.
    assert "messages: legacy," in storage

    hook = _read(HOOK)
    assert "? await listStoredChatThreadsWithMessages(args)" in hook
    assert "void repairLegacyChatTitles( loaded.threads, loaded.messagesByThreadId, )" in hook

    repair = _read(REPAIR)
    assert "messages = known ? new Map(known) : await batchListChatMessages(ids);" in repair


def test_a_legacy_only_chat_falls_back_to_a_local_read():
    """A chat still only in Dexie reads empty from the backend. Without the
    local read it would sit in `attempted` with its title still clipped."""
    repair = _read(REPAIR)
    assert "threadsMissingMessages(ids, messages)" in repair
    assert "messages.set(id, await listLegacyChatMessages(id))" in repair

    storage = _read(STORAGE)
    assert "export async function listLegacyChatMessages(" in storage
    assert 'db.messages .where("threadId") .equals(threadId) .toArray()' in storage


def test_the_next_page_is_scheduled_rather_than_waited_for():
    """A page that writes nothing fires no history update, so nothing else
    would come back for the rest of the backlog."""
    repair = _read(REPAIR)
    assert "if (hasMore) { setTimeout(" in repair
    # On `rest`: a row this page unmarked must not be drawn again by the same
    # drain, or a failing PATCH starves every row behind it.
    assert "void repairLegacyChatTitles(rest, known)" in repair
    assert "REPAIR_PAGE_PAUSE_MS" in repair


def test_rows_with_nothing_found_stay_retryable():
    repair = _read(REPAIR)
    assert "for (const id of threadsMissingMessages(ids, messages)) attempted.delete(id);" in repair


def test_a_local_read_is_sorted_like_the_backend_lists_messages():
    """planLegacyTitleRepairs wants the opening message, and a Dexie read
    arrives in index order."""
    storage = _read(STORAGE)
    assert (
        "messages.sort( (a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id), )"
        in storage
    )


def test_a_failed_read_leaves_the_rows_retryable():
    repair = _read(REPAIR)
    assert (
        "} catch { // Nothing was decided, so let a later refresh try these again. for (const id of ids) attempted.delete(id); return 0; }"
        in repair
    )
