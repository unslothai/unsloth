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
BACKEND = REPO / "studio/backend"
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


def test_an_emptied_chat_is_not_retried_for_the_session():
    """A chat the user deleted every message from reads back the same as one
    still importing. Retrying it would re-read it on every refresh, since its
    title stays clipped and keeps matching the pre-filter."""
    repair = _read(REPAIR)
    assert "const withoutMessages = threadsMissingMessages(ids, messages);" in repair
    # Only fetched when there is something to decide, so a page of ordinary
    # rows does not pay for it.
    assert "if (withoutMessages.length > 0) {" in repair
    assert "imported = await listChatImportLedger();" in repair
    assert (
        "for (const id of threadsAwaitingImport(ids, messages, imported)) { attempted.delete(id); }"
        in repair
    )


def test_the_repair_never_creates_anything():
    """updateStoredChatThread ensures the thread first, which re-imports one
    deleted on another client from the Dexie rows still sitting here. A
    migration patching a row that is gone has to 404, not resurrect it."""
    repair = _read(REPAIR)
    # The storage layer is out of the picture entirely, not just at this call.
    assert 'from "./chat-history-storage"' not in repair
    assert "await updateChatThread( repair.threadId, { title: repair.title }," in repair


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


def test_a_rename_is_left_to_the_backend_guard():
    """The pass only runs where the guard is enforced, so re-listing the whole
    history per page to check titles client side was both redundant and
    quadratic: every page of 100 pulled every thread the account has."""
    repair = _read(REPAIR)
    assert "listChatThreads" not in repair
    assert "repairsStillValid" not in repair
    assert "await runWithConcurrency(repairs, REPAIR_CONCURRENCY," in repair
    assert "expectedTitle: repair.previousTitle," in repair


def test_the_write_is_guarded_on_the_message_it_took_the_title_from():
    """Deleting the opening prompt does not change the title, so expectedTitle
    alone still matches and would expand the deleted text."""
    repair = _read(REPAIR)
    assert "expectedOpeningMessageId: repair.openingMessageId," in repair

    backend = (BACKEND / "routes/chat_history.py").read_text(encoding = "utf-8")
    assert "expectedOpeningMessageId: Optional[str] = None" in backend


def test_the_migration_stays_off_where_the_guard_is_not_enforced():
    """A backend from before expectedTitle drops it and writes anyway. Sending
    one to find out is the destructive act itself, so the served schema is what
    answers, and anything unreadable counts as unsupported."""
    repair = _read(REPAIR)
    assert "if (!(await backendEnforcesTitleGuard())) return 0;" in repair
    assert 'const response = await authFetch("/openapi.json");' in repair
    assert (
        "probe = readGuardProbe( response.ok, response.ok ? await response.json() : null, );"
        in repair
    )
    # Only a settled answer is cached. A 401 while the token warms up, or a 503
    # during startup, would otherwise park the migration for the session.
    assert "if (!probe.settled) guardSupport = null;" in repair

    backend = (BACKEND / "routes/chat_history.py").read_text(encoding = "utf-8")
    # The probe reads the schema, so the fields have to be declared on the
    # model. It looks for both, and both are what make the write safe.
    assert "expectedTitle: Optional[str] = None" in backend


def test_a_failed_read_leaves_the_rows_retryable():
    repair = _read(REPAIR)
    assert (
        "} catch { // Nothing was decided, so let a later refresh try these again. for (const id of ids) attempted.delete(id); return 0; }"
        in repair
    )
