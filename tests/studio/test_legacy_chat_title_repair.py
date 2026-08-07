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


def test_the_repair_reads_its_own_messages_as_late_as_it_can():
    """Not the sidebar's map. That map takes its empty-backend entries from
    listStoredChatMessages, which merges Dexie rows the backend has pruned, and
    it is fetched at load time rather than at write time."""
    hook = _read(HOOK)
    assert "void repairLegacyChatTitles(threads).catch(() => undefined);" in hook

    repair = _read(REPAIR)
    assert "messages = await batchListChatMessages(ids);" in repair
    assert (
        "export function repairLegacyChatTitles( threads: ThreadRecord[], ): Promise<number> {"
        in repair
    )

    storage = _read(STORAGE)
    # The shared list function keeps its original shape: nothing hands a
    # message map out of it.
    assert "messagesByThreadId" not in storage


def test_the_repair_reads_only_stored_messages():
    """Dexie keeps rows the backend has pruned, because deleting a message never
    clears them. Reading it here could put a deleted prompt back into a title,
    so a chat whose messages are not stored yet is left for a later refresh
    instead."""
    repair = _read(REPAIR)
    assert "listUnimportedChatMessages" not in repair
    assert "mergeMessagesById" not in repair
    assert "db.messages" not in repair
    assert "for (const id of threadsMissingMessages(ids, messages)) attempted.delete(id);" in repair


def test_the_next_page_is_scheduled_rather_than_waited_for():
    """A page that writes nothing fires no history update, so nothing else
    would come back for the rest of the backlog."""
    repair = _read(REPAIR)
    assert "if (hasMore) { setTimeout(" in repair
    # On `rest`: a row this page unmarked must not be drawn again by the same
    # drain, or a failing PATCH starves every row behind it.
    assert "void repairLegacyChatTitles(rest)" in repair
    assert "REPAIR_PAGE_PAUSE_MS" in repair


def test_only_one_repair_pass_runs_at_a_time():
    """Several sidebars can be mounted at once, so the write concurrency cap
    only holds if their passes queue."""
    repair = _read(REPAIR)
    assert "const serial = createSerialQueue();" in repair
    assert "return serial(() => runRepairPass(threads));" in repair


def test_a_failed_read_leaves_the_rows_retryable():
    repair = _read(REPAIR)
    assert (
        "} catch { // Nothing was decided, so let a later refresh try these again. for (const id of ids) attempted.delete(id); return 0; }"
        in repair
    )
