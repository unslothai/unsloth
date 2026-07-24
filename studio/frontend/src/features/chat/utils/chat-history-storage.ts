// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  buildBackendChatExport,
  clearBackendChats,
  deleteChatProject,
  deleteChatThreads,
  getChatProject,
  getChatMessage,
  getChatThread,
  batchListChatMessages,
  listChatProjects,
  listChatImportLedger,
  listChatMessages,
  listChatThreads,
  notifyChatHistoryUpdated,
  recordChatImportLedger,
  saveChatProject,
  saveChatMessage,
  saveChatThread,
  syncChatMessages,
  updateChatProject,
  updateChatThread,
} from "../api/chat-api";
import { db, DEXIE_DB_NAME } from "../db";
import type {
  MessageRecord,
  ModelType,
  ProjectRecord,
  ThreadRecord,
} from "../types";
import {
  isChatThreadDeleted,
  markChatThreadsDeleted,
} from "./chat-thread-tombstones";

// Thread ids that belong to a temporary/incognito session. A thread is
// tagged once, at creation (ensureThreadRecord, when the toggle is on), and
// stays tagged for its whole lifetime -- the readers and writers below
// consult this set, never the live toggle. That decoupling is what makes
// mid-stream toggling safe: flipping the toggle can neither leak an
// in-flight incognito run into history nor drop a normal thread's writes.
// Per-thread reads short-circuit too (nothing is stored to fetch); only the
// thread list stays ungated, so real history still loads next to a
// temporary chat.
const incognitoThreadIds = new Set<string>();

export function markThreadIncognito(threadId: string): void {
  incognitoThreadIds.add(threadId);
}

export function isThreadIncognito(threadId: string): boolean {
  return incognitoThreadIds.has(threadId);
}

type ThreadListArgs = {
  modelType?: ModelType;
  pairId?: string;
  projectId?: string | null;
  includeArchived?: boolean;
};

// localStorage perf-hint that the Dexie -> studio.db import already
// finished. NOT the import gate -- the server-side ledger
// (chat_legacy_imports) is the source of truth so a studio.db wipe stays
// recoverable. The hint only short-circuits the listing paths' "also
// surface Dexie threads?" branches once the ledger has covered everything.
const LEGACY_CHAT_IMPORT_KEY = "unsloth_chat_legacy_imported_to_studio_db";

let legacyChatImportPromise: Promise<void> | null = null;

interface ExportedChat {
  exportedAt: string;
  version: 1;
  threadCount: number;
  projects?: unknown[];
  threads: unknown[];
  messages: unknown[];
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isLegacyChatImportDone(): boolean {
  if (!canUseStorage()) return true;
  try {
    return localStorage.getItem(LEGACY_CHAT_IMPORT_KEY) === "true";
  } catch {
    return false;
  }
}

function markLegacyChatImportDone(): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(LEGACY_CHAT_IMPORT_KEY, "true");
  } catch {
    // ignore
  }
}

function matchesThreadListArgs(
  thread: ThreadRecord,
  args: ThreadListArgs,
): boolean {
  return (
    !isChatThreadDeleted(thread.id) &&
    (!args.pairId || thread.pairId === args.pairId) &&
    (args.projectId === undefined ||
      (thread.projectId ?? null) === args.projectId) &&
    (!args.modelType || thread.modelType === args.modelType) &&
    (args.includeArchived !== false || !thread.archived)
  );
}

async function listLegacyThreads(
  args: ThreadListArgs,
): Promise<ThreadRecord[]> {
  const legacyQuery = args.pairId
    ? db.threads.where("pairId").equals(args.pairId)
    : args.modelType
      ? db.threads.where("modelType").equals(args.modelType)
      : db.threads.toCollection();
  return (await legacyQuery.toArray()).filter((thread) =>
    matchesThreadListArgs(thread, args),
  );
}

function sortMessages(messages: MessageRecord[]): MessageRecord[] {
  const roleOrder: Record<string, number> = {
    system: 0,
    user: 1,
    assistant: 2,
  };
  return [...messages].sort((a, b) => {
    if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
    const aOrder = roleOrder[a.role] ?? 99;
    const bOrder = roleOrder[b.role] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

export function isExpectedBackgroundChatStorageError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message === "Invalid or expired token" ||
      error.message === "Not authenticated" ||
      error.message === "Request failed (401)" ||
      error.message === "Unsloth isn't running -- please relaunch it.")
  );
}

function normalizeLegacyMessages(messages: MessageRecord[]): MessageRecord[] {
  let previousId: string | null = null;
  return sortMessages(messages).map((message) => {
    const parentId = hasOwn(message, "parentId")
      ? (message.parentId ?? null)
      : previousId;
    previousId = message.id;
    return {
      ...message,
      parentId,
    };
  });
}

function messageNeedsBackfill(
  backend: MessageRecord,
  legacy: MessageRecord,
): boolean {
  return (
    (backend.parentId == null && legacy.parentId != null) ||
    (backend.attachments == null && legacy.attachments != null) ||
    (backend.metadata == null && legacy.metadata != null)
  );
}

function mergeLegacyMessageFields(
  backend: MessageRecord,
  legacy: MessageRecord,
): MessageRecord {
  return {
    ...backend,
    ...(backend.parentId == null && legacy.parentId != null
      ? { parentId: legacy.parentId }
      : {}),
    ...(backend.attachments == null && legacy.attachments != null
      ? { attachments: legacy.attachments }
      : {}),
    ...(backend.metadata == null && legacy.metadata != null
      ? { metadata: legacy.metadata }
      : {}),
  };
}

function mergeMessages(
  backendMessages: MessageRecord[],
  legacyMessages: MessageRecord[],
  options: { includeLegacyOnly?: boolean } = {},
): { messages: MessageRecord[]; shouldSync: boolean } {
  const byId = new Map<string, MessageRecord>();
  const includeLegacyOnly = options.includeLegacyOnly ?? true;
  const backendIds = new Set(
    backendMessages
      .filter((message) => !isChatThreadDeleted(message.threadId))
      .map((message) => message.id),
  );
  let shouldSync = false;
  for (const message of normalizeLegacyMessages(legacyMessages)) {
    if (!isChatThreadDeleted(message.threadId)) {
      if (includeLegacyOnly || backendIds.has(message.id)) {
        byId.set(message.id, message);
      }
      if (includeLegacyOnly && !backendIds.has(message.id)) shouldSync = true;
    }
  }
  for (const message of backendMessages) {
    if (!isChatThreadDeleted(message.threadId)) {
      const legacyMessage = byId.get(message.id);
      if (legacyMessage && messageNeedsBackfill(message, legacyMessage)) {
        byId.set(message.id, mergeLegacyMessageFields(message, legacyMessage));
        shouldSync = true;
      } else {
        byId.set(message.id, message);
      }
    }
  }
  return { messages: Array.from(byId.values()), shouldSync };
}

async function importLegacyThread(
  thread: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  const saved = await saveChatThread(thread);
  const legacyMessages = await db.messages
    .where("threadId")
    .equals(thread.id)
    .toArray();
  if (legacyMessages.length > 0) {
    await syncChatMessages(thread.id, normalizeLegacyMessages(legacyMessages), {
      pruneMissing: false,
    });
  }
  return saved;
}

async function backfillLegacyThreadFields(
  backendThread: ThreadRecord,
  legacyThread: ThreadRecord | undefined,
): Promise<ThreadRecord> {
  if (!legacyThread) return backendThread;
  const patch: Partial<ThreadRecord> = {};
  if (
    !backendThread.openaiCodeExecContainerId &&
    legacyThread.openaiCodeExecContainerId
  ) {
    patch.openaiCodeExecContainerId = legacyThread.openaiCodeExecContainerId;
  }
  if (
    !backendThread.anthropicCodeExecContainerId &&
    legacyThread.anthropicCodeExecContainerId
  ) {
    patch.anthropicCodeExecContainerId =
      legacyThread.anthropicCodeExecContainerId;
  }
  if (Object.keys(patch).length === 0) return backendThread;
  try {
    return (
      (await updateChatThread(backendThread.id, patch)) ?? {
        ...backendThread,
        ...patch,
      }
    );
  } catch {
    return backendThread;
  }
}

// Fast-path: check whether the "unsloth-chat" DB exists without opening
// it. Supported on modern Chromium/Firefox/Safari; older browsers return
// undefined and we fall through to the next probe.
async function dexieDbAbsent(): Promise<boolean> {
  if (typeof indexedDB === "undefined") return true;
  const dbs = (indexedDB as IDBFactory).databases;
  if (typeof dbs !== "function") return false;
  try {
    const list = await dbs.call(indexedDB);
    if (!Array.isArray(list)) return false;
    return !list.some((entry) => entry?.name === DEXIE_DB_NAME);
  } catch {
    return false;
  }
}

// Fast-path: Dexie exists but is empty. count() reads the IndexedDB
// store metadata, not the rows -- cheap regardless of record count.
async function dexieIsEmpty(): Promise<boolean> {
  try {
    const [threadCount, messageCount] = await Promise.all([
      db.threads.count(),
      db.messages.count(),
    ]);
    return threadCount === 0 && messageCount === 0;
  } catch {
    // Dexie threw (corrupt DB / version mismatch / quota). Returning
    // false forces the slow path (same Dexie underneath); it'll throw
    // too and reset the import promise so the next caller can retry.
    return false;
  }
}

async function importLegacyChatsIfNeeded(): Promise<void> {
  // Session-level cache: repeated sidebar mounts in the same tab share
  // one import. localStorage is NOT consulted -- the server-side ledger
  // is the source of truth, so a studio.db wipe re-triggers the import
  // even if the browser kept its old hint.
  if (legacyChatImportPromise) return legacyChatImportPromise;

  legacyChatImportPromise = (async () => {
    // Fast-path: no Dexie DB -- new user, never had browser-only Unsloth.
    if (await dexieDbAbsent()) {
      markLegacyChatImportDone();
      return;
    }

    // Fast-path: Dexie exists but is empty (already migrated long
    // ago and Dexie just hasn't been GC'd, or the browser created an
    // empty DB for some reason).
    if (await dexieIsEmpty()) {
      markLegacyChatImportDone();
      return;
    }

    // Slow path: diff Dexie against the server-side ledger and import
    // any threads not already recorded.
    const [legacyThreads, backendThreads, importedThreadIds] = await Promise.all([
      db.threads.toArray(),
      listChatThreads({ includeArchived: true }),
      listChatImportLedger(),
    ]);

    const backendThreadsById = new Map(
      backendThreads.map((thread) => [thread.id, thread]),
    );
    const unimportedIds: string[] = [];
    const unimportedThreads: ThreadRecord[] = [];

    // "Unimported" = missing from the ledger. Include threads already in
    // the backend (without a ledger row) so the ledger gets backfilled
    // for old-FE-then-new-FE users; else the next launch re-diffs forever.
    for (const thread of legacyThreads) {
      if (isChatThreadDeleted(thread.id)) continue;
      if (importedThreadIds.has(thread.id)) continue;
      unimportedIds.push(thread.id);
      unimportedThreads.push(thread);
    }

    if (unimportedIds.length === 0) {
      markLegacyChatImportDone();
      return;
    }

    // Two bulk reads instead of 2N per-thread round-trips.
    const allLegacyMessages = await db.messages
      .where("threadId")
      .anyOf(unimportedIds)
      .toArray()
      .catch(() => [] as MessageRecord[]);
    const legacyByThread = new Map<string, MessageRecord[]>();
    for (const message of allLegacyMessages) {
      const arr = legacyByThread.get(message.threadId);
      if (arr) arr.push(message);
      else legacyByThread.set(message.threadId, [message]);
    }
    const backendByThread = await batchListChatMessages(unimportedIds).catch(
      () => new Map<string, MessageRecord[]>(),
    );

    const newlyImportedIds: string[] = [];
    for (const thread of unimportedThreads) {
      const backendThread = backendThreadsById.get(thread.id);
      if (!backendThread) {
        await saveChatThread(thread);
        backendThreadsById.set(thread.id, thread);
      } else {
        backendThreadsById.set(
          thread.id,
          await backfillLegacyThreadFields(backendThread, thread),
        );
      }

      const legacyMessages = legacyByThread.get(thread.id) ?? [];
      if (legacyMessages.length === 0) {
        newlyImportedIds.push(thread.id);
        continue;
      }

      const backendMessages = backendByThread.get(thread.id) ?? [];
      const merged = mergeMessages(backendMessages, legacyMessages);
      if (merged.shouldSync) {
        await syncChatMessages(thread.id, sortMessages(merged.messages), {
          pruneMissing: false,
        });
      }
      newlyImportedIds.push(thread.id);
    }

    if (newlyImportedIds.length === 0) {
      markLegacyChatImportDone();
      return;
    }
    let result: { supported: boolean };
    try {
      result = await recordChatImportLedger(newlyImportedIds);
    } catch {
      // Network error: leave the perf hint alone so the next launch
      // retries. Import is idempotent via UPSERT, so no duplicates.
      return;
    }
    // Only flip the hint when the backend actually has the ledger. On
    // older deployments (404/405/501) it would lie ("import done" with an
    // empty ledger), defeating recovery after a studio.db wipe.
    if (result.supported) {
      markLegacyChatImportDone();
    }
  })();

  try {
    await legacyChatImportPromise;
  } catch (error) {
    legacyChatImportPromise = null;
    throw error;
  }
}

export async function getStoredChatThread(
  threadId: string,
): Promise<ThreadRecord | undefined> {
  // Incognito threads are never stored, so the lookup can only come back
  // empty -- short-circuit it instead of doing a Dexie read + backend GET.
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  const legacyThread = await db.threads.get(threadId);
  let backendThread: ThreadRecord | null;
  try {
    backendThread = await getChatThread(threadId);
  } catch (error) {
    if (legacyThread && !isChatThreadDeleted(legacyThread.id)) {
      return legacyThread;
    }
    throw error;
  }
  if (backendThread && !isChatThreadDeleted(backendThread.id)) {
    return backfillLegacyThreadFields(backendThread, legacyThread);
  }
  if (!legacyThread || isChatThreadDeleted(legacyThread.id)) return undefined;
  return importLegacyThread(legacyThread).catch(() => legacyThread);
}

export async function ensureStoredChatThread(
  threadId: string,
  fallback?: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  // An incognito thread is never persisted, so there's genuinely nothing
  // to ensure -- skip the backend round-trips this would otherwise make
  // on every autosave (runStart/runEnd) and message append.
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  const legacyThread = fallback ?? (await db.threads.get(threadId));
  let backendThread: ThreadRecord | null;
  try {
    backendThread = await getChatThread(threadId);
  } catch (error) {
    if (!legacyThread || isChatThreadDeleted(legacyThread.id)) {
      throw error;
    }
    return legacyThread;
  }
  if (backendThread) {
    return backfillLegacyThreadFields(backendThread, legacyThread);
  }
  if (!legacyThread || isChatThreadDeleted(legacyThread.id)) return undefined;
  return importLegacyThread(legacyThread).catch(() => legacyThread);
}

export async function listStoredChatMessages(
  threadId: string,
): Promise<MessageRecord[]> {
  if (isThreadIncognito(threadId)) return [];
  if (isChatThreadDeleted(threadId)) return [];
  const legacyMessages = await db.messages
    .where("threadId")
    .equals(threadId)
    .toArray();
  const [backendThread, backendMessages] = await Promise.all([
    getChatThread(threadId).catch(() => undefined),
    listChatMessages(threadId).catch((error) => {
      if (legacyMessages.length > 0) {
        return undefined;
      }
      throw error;
    }),
  ]);
  if (backendMessages && (backendThread || backendMessages.length > 0)) {
    const merged = mergeMessages(backendMessages, legacyMessages, {
      includeLegacyOnly:
        !isLegacyChatImportDone() ||
        (backendMessages.length === 0 && legacyMessages.length > 0),
    });
    if (legacyMessages.length > 0 && merged.shouldSync) {
      return syncChatMessages(threadId, merged.messages, {
        pruneMissing: false,
      }).catch(() => merged.messages);
    }
    return merged.messages;
  }
  if (
    backendMessages &&
    isLegacyChatImportDone() &&
    legacyMessages.length === 0
  ) {
    return [];
  }
  return legacyMessages.filter(
    (message) => !isChatThreadDeleted(message.threadId),
  );
}

export async function getStoredChatMessage(
  threadId: string,
  messageId: string,
): Promise<MessageRecord | undefined> {
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  const legacyMessage = await db.messages.get(messageId);
  const matchingLegacyMessage =
    legacyMessage?.threadId === threadId ? legacyMessage : undefined;
  let backendMessage: MessageRecord | null;
  try {
    backendMessage = await getChatMessage(threadId, messageId);
  } catch (error) {
    if (matchingLegacyMessage) {
      return matchingLegacyMessage;
    }
    throw error;
  }
  if (backendMessage) {
    if (
      matchingLegacyMessage &&
      messageNeedsBackfill(backendMessage, matchingLegacyMessage)
    ) {
      return mergeLegacyMessageFields(backendMessage, matchingLegacyMessage);
    }
    return backendMessage;
  }
  return matchingLegacyMessage;
}

export async function listStoredChatThreads(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const legacyThreads = await listLegacyThreads(args);
  let backendThreads = await listChatThreads(args).catch((error) => {
    if (legacyThreads.length > 0) {
      return undefined;
    }
    throw error;
  });
  if (backendThreads) {
    await importLegacyChatsIfNeeded().catch(() => undefined);
    backendThreads = await listChatThreads(args).catch(() => backendThreads);
  }
  const includeLegacyOnly =
    !backendThreads ||
    !isLegacyChatImportDone() ||
    (backendThreads.length === 0 && legacyThreads.length > 0);
  const byId = new Map<string, ThreadRecord>();
  if (includeLegacyOnly) {
    for (const thread of legacyThreads) byId.set(thread.id, thread);
  }
  for (const thread of backendThreads ?? []) {
    if (!isChatThreadDeleted(thread.id)) byId.set(thread.id, thread);
  }
  return Array.from(byId.values())
    .filter((thread) => matchesThreadListArgs(thread, args))
    .sort(
      (a, b) =>
        (b.updatedAt ?? b.createdAt) - (a.updatedAt ?? a.createdAt),
    );
}

export async function listStoredChatThreadsWithMessages(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const threads = await listStoredChatThreads(args);
  if (threads.length === 0) return [];
  // One batched HTTP call instead of N. Per-thread legacy Dexie fallback
  // only fires when the batch result is empty.
  const threadIds = threads.map((t) => t.id);
  let backendByThread: Map<string, MessageRecord[]>;
  try {
    backendByThread = await batchListChatMessages(threadIds);
  } catch {
    backendByThread = new Map();
  }
  const entries = await Promise.all(
    threads.map(async (thread) => {
      const backendMessages = backendByThread.get(thread.id) ?? [];
      if (backendMessages.length > 0) {
        return { thread, hasContent: true };
      }
      const legacy = await listStoredChatMessages(thread.id).catch(() => null);
      return { thread, hasContent: legacy === null || legacy.length > 0 };
    }),
  );
  return entries.filter((e) => e.hasContent).map((e) => e.thread);
}

export async function listStoredChatProjects(
  args: { includeArchived?: boolean } = {},
): Promise<ProjectRecord[]> {
  return listChatProjects(args);
}

export async function getStoredChatProject(
  projectId: string,
): Promise<ProjectRecord | null> {
  return getChatProject(projectId);
}

export async function createStoredChatProject(
  name: string,
): Promise<ProjectRecord> {
  const trimmed = name.trim();
  if (!trimmed) {
    throw new Error("Project name is required.");
  }
  const now = Date.now();
  return saveChatProject({
    id: crypto.randomUUID(),
    name: trimmed,
    instructions: "",
    archived: false,
    createdAt: now,
    updatedAt: now,
  });
}

export async function updateStoredChatProject(
  projectId: string,
  patch: Partial<ProjectRecord>,
): Promise<ProjectRecord> {
  return updateChatProject(projectId, {
    ...patch,
    updatedAt: patch.updatedAt ?? Date.now(),
  });
}

export async function deleteStoredChatProject(
  projectId: string,
  args: { deleteFiles?: boolean } = {},
): Promise<void> {
  await deleteChatProject(projectId, args);
}

export async function moveStoredChatItemToProject(
  item: { type: "single" | "compare"; id: string },
  projectId: string | null,
): Promise<void> {
  const threadIds =
    item.type === "single"
      ? [item.id]
      : (await listStoredChatThreads({
          pairId: item.id,
          includeArchived: true,
        })).map((thread) => thread.id);

  await Promise.all(
    Array.from(new Set(threadIds)).map((threadId) =>
      updateStoredChatThread(threadId, { projectId }),
    ),
  );
}

export async function saveStoredChatMessage(
  message: MessageRecord,
): Promise<MessageRecord> {
  if (isThreadIncognito(message.threadId)) return message;
  if (isChatThreadDeleted(message.threadId)) {
    throw new Error(`Thread ${message.threadId} was deleted`);
  }
  await ensureStoredChatThread(message.threadId);
  return saveChatMessage(message);
}

export async function syncStoredChatMessages(
  threadId: string,
  messages: MessageRecord[],
  options: { pruneMissing?: boolean } = {},
): Promise<MessageRecord[]> {
  if (isThreadIncognito(threadId)) return messages;
  if (isChatThreadDeleted(threadId)) return [];
  await ensureStoredChatThread(threadId);
  return syncChatMessages(threadId, messages, options);
}

export async function saveStoredChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord> {
  if (isThreadIncognito(thread.id)) return thread;
  if (isChatThreadDeleted(thread.id)) {
    throw new Error(`Thread ${thread.id} was deleted`);
  }
  return saveChatThread(thread);
}

export async function updateStoredChatThread(
  threadId: string,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord | undefined> {
  if (isThreadIncognito(threadId)) return undefined;
  const thread = await ensureStoredChatThread(threadId);
  if (!thread) return undefined;
  return updateChatThread(threadId, patch);
}

export async function deleteStoredChatThreads(
  idsToDelete: string[],
): Promise<void> {
  // Incognito threads were never stored, so there's nothing to delete --
  // drop them to skip the no-op backend DELETE (and the history-refresh
  // event it would fire) when the active temporary chat is closed.
  const ids = idsToDelete.filter((id) => !isThreadIncognito(id));
  if (ids.length === 0) return;
  await deleteChatThreads(ids);
  await db
    .transaction("rw", db.threads, db.messages, async () => {
      await db.messages.where("threadId").anyOf(ids).delete();
      await db.threads.bulkDelete(ids);
    })
    .catch(() => undefined);
  markChatThreadsDeleted(ids);
}

export async function countStoredChats(): Promise<number> {
  return (await listStoredChatThreads()).length;
}

export interface ClearStoredChatsResult {
  backend: "cleared" | "failed" | "skipped";
  legacy: "cleared" | "failed" | "skipped";
  deletedThreadIds: string[];
  failedThreadIds: string[];
}

export async function clearStoredChats(): Promise<ClearStoredChatsResult> {
  // Clear both sides independently and report each outcome so the toast
  // can distinguish full vs partial success.
  const [backendThreadsResult, legacyThreads] = await Promise.all([
    listChatThreads()
      .then((threads) => ({ ok: true as const, threads }))
      .catch(() => ({ ok: false as const, threads: [] as ThreadRecord[] })),
    db.threads.toArray().catch(() => []),
  ]);
  const backendInventoryLoaded = backendThreadsResult.ok;
  const backendThreadIds = new Set(
    backendThreadsResult.threads.map((thread) => thread.id),
  );
  const legacyThreadIds = new Set(legacyThreads.map((thread) => thread.id));
  const allThreadIds = Array.from(
    new Set([...backendThreadIds, ...legacyThreadIds]),
  );

  const result: ClearStoredChatsResult = {
    backend: "skipped",
    legacy: "skipped",
    deletedThreadIds: [],
    failedThreadIds: [],
  };
  try {
    // Defer the history refresh until Dexie clear and tombstones finalize,
    // so listeners never observe the composite clear mid-flight.
    await clearBackendChats({ notify: false });
    result.backend = "cleared";
  } catch (error) {
    result.backend = "failed";
    console.error("clearStoredChats: backend clear failed", error);
  }

  try {
    await db.transaction("rw", db.threads, db.messages, async () => {
      await db.messages.clear();
      await db.threads.clear();
    });
    result.legacy = "cleared";
  } catch (error) {
    result.legacy = "failed";
    console.error("clearStoredChats: legacy Dexie clear failed", error);
  }

  result.deletedThreadIds = allThreadIds.filter((id) => {
    const backendDeleted =
      result.backend === "cleared" ||
      (backendInventoryLoaded && !backendThreadIds.has(id));
    const legacyDeleted =
      !legacyThreadIds.has(id) || result.legacy === "cleared";
    return backendDeleted && legacyDeleted;
  });
  const deleted = new Set(result.deletedThreadIds);
  result.failedThreadIds = allThreadIds.filter((id) => !deleted.has(id));

  markChatThreadsDeleted(result.deletedThreadIds);
  notifyChatHistoryUpdated();

  if (result.backend === "failed" && result.legacy === "failed") {
    throw new Error("clearStoredChats: both backend and legacy clear failed");
  }
  return result;
}

export async function buildStoredChatExport(): Promise<ExportedChat> {
  await importLegacyChatsIfNeeded().catch(() => undefined);
  const [legacyThreads, legacyMessages] = await Promise.all([
    db.threads.toArray(),
    db.messages.toArray(),
  ]);
  const hasLegacyData =
    legacyThreads.some((thread) => !isChatThreadDeleted(thread.id)) ||
    legacyMessages.some((message) => !isChatThreadDeleted(message.threadId));
  const backend = await buildBackendChatExport().catch((error) => {
    if (hasLegacyData) {
      return null;
    }
    throw error;
  });
  const threadsById = new Map<string, unknown>();
  const backendThreadIds = new Set<string>();
  const messagesById = new Map<string, unknown>();

  for (const thread of backend?.threads ?? []) {
    if (isChatThreadDeleted(thread.id)) continue;
    backendThreadIds.add(thread.id);
    threadsById.set(thread.id, thread);
  }
  for (const message of backend?.messages ?? []) {
    if (isChatThreadDeleted(message.threadId)) continue;
    messagesById.set(message.id, message);
  }
  const includeLegacyOnly = backend === null || !isLegacyChatImportDone();
  for (const thread of legacyThreads as ThreadRecord[]) {
    if (
      isChatThreadDeleted(thread.id) ||
      backendThreadIds.has(thread.id) ||
      !includeLegacyOnly
    ) {
      continue;
    }
    threadsById.set(thread.id, thread);
  }
  for (const message of legacyMessages as MessageRecord[]) {
    if (isChatThreadDeleted(message.threadId)) {
      continue;
    }
    if (!includeLegacyOnly) continue;
    if (!messagesById.has(message.id)) {
      messagesById.set(message.id, message);
    }
  }

  const threads = Array.from(threadsById.values());
  const messages = Array.from(messagesById.values());
  return {
    exportedAt: new Date().toISOString(),
    version: 1,
    threadCount: threads.length,
    projects: backend?.projects ?? [],
    threads,
    messages,
  };
}
