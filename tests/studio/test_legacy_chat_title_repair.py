# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checks the repair's seams by source text, since importing the module pulls in the chat API."""

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
    """The repair reads its own messages as late as it can, not the sidebar map: that map merges
    pruned Dexie rows via listStoredChatMessages and is fetched at load time."""
    hook = _read(HOOK)
    assert "void repairLegacyChatTitles(threads).catch(() => undefined);" in hook

    repair = _read(REPAIR)
    assert "messages = await batchListChatMessages(ids);" in repair
    assert (
        "export function repairLegacyChatTitles( threads: ThreadRecord[], ): Promise<number> {"
        in repair
    )

    storage = _read(STORAGE)
    assert "messagesByThreadId" not in storage


def test_the_repair_reads_only_stored_messages():
    """Dexie keeps rows for deleted messages, so only stored messages may feed a title."""
    repair = _read(REPAIR)
    assert "listUnimportedChatMessages" not in repair
    assert "mergeMessagesById" not in repair
    assert "db.messages" not in repair


def test_an_emptied_chat_is_not_retried_for_the_session():
    """An emptied chat must not be retried for the session: it reads back like one still importing,
    and its clipped title keeps matching the pre-filter on every refresh."""
    repair = _read(REPAIR)
    assert "const withoutMessages = threadsMissingMessages(ids, messages);" in repair
    assert "if (withoutMessages.length > 0) {" in repair
    assert "imported = await listChatImportLedger();" in repair
    assert (
        "for (const id of threadsAwaitingImport(ids, messages, imported)) { attempted.delete(id); }"
        in repair
    )


def test_the_repair_never_creates_anything():
    """The repair must never create threads: updateStoredChatThread ensures the thread first and
    would re-import one deleted elsewhere, so a patch to a gone row must 404."""
    repair = _read(REPAIR)
    assert 'from "./chat-history-storage"' not in repair
    assert "await updateChatThread( repair.threadId, { title: repair.title }," in repair


def test_the_next_page_is_scheduled_rather_than_waited_for():
    """A page that writes nothing fires no history update, so nothing else
    would come back for the rest of the backlog."""
    repair = _read(REPAIR)
    assert "if (hasMore) { setTimeout(" in repair
    # On `rest`: a row this page unmarked must not starve the rows behind it.
    assert "void repairLegacyChatTitles(rest)" in repair
    assert "REPAIR_PAGE_PAUSE_MS" in repair


def test_only_one_repair_pass_runs_at_a_time():
    """Several sidebars can be mounted at once, so the write concurrency cap
    only holds if their passes queue."""
    repair = _read(REPAIR)
    assert "const serial = createSerialQueue();" in repair
    assert "return serial(() => runRepairPass(threads));" in repair


def test_a_rename_is_left_to_the_backend_guard():
    """Renames are left to the backend guard: the pass runs only where it is enforced, and
    re-listing all threads per page client side would be redundant and quadratic."""
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
    # The thread list's advertised flag decides first; only an older backend falls back to the schema.
    assert (
        "const probe = await resolveGuardProbe( advertisedThreadPatchGuards(), probeServedSchema, );"
        in repair
    )
    assert 'const response = await authFetch("/openapi.json");' in repair
    assert (
        "return readGuardProbe( response.ok, response.ok ? await response.json() : null, );"
        in repair
    )
    # Only a settled answer is cached, or a 401 at startup parks the migration.
    assert "if (!probe.settled) guardSupport = null;" in repair

    backend = (BACKEND / "routes/chat_history.py").read_text(encoding = "utf-8")
    # The probe reads the schema, so the fields must be declared on the model.
    assert "expectedTitle: Optional[str] = None" in backend


def test_a_failed_read_leaves_the_rows_retryable():
    repair = _read(REPAIR)
    assert (
        "} catch { // Nothing was decided, so let a later refresh try these again. for (const id of ids) attempted.delete(id); return 0; }"
        in repair
    )
