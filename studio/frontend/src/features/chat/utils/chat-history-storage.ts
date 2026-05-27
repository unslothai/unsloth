// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  buildBackendChatExport,
  clearBackendChats,
  deleteChatThreads,
  getChatMessage,
  getChatThread,
  batchListChatMessages,
  listChatImportLedger,
  listChatMessages,
  listChatThreads,
  notifyChatHistoryUpdated,
  recordChatImportLedger,
  saveChatMessage,
  saveChatThread,
  syncChatMessages,
  updateChatThread,
} from "../api/chat-api";
import { db, DEXIE_DB_NAME } from "../db";
import type { MessageRecord, ModelType, ThreadRecord } from "../types";
import {
  isChatThreadDeleted,
  markChatThreadsDeleted,
} from "./chat-thread-tombstones";

type ThreadListArgs = {
  modelType?: ModelType;
  pairId?: string;
  includeArchived?: boolean;
};

// localStorage perf-hint that the Dexie -> studio.db import already
// finished in a previous session. NOT consulted by the import gate
// itself -- the server-side ledger (chat_legacy_imports) is the source
// of truth so a studio.db wipe stays recoverable. The hint only short-
// circuits the listing paths' "should I also surface Dexie threads?"
// branches once the ledger has covered everything.
const LEGACY_CHAT_IMPORT_KEY = "unsloth_chat_legacy_imported_to_studio_db";

let legacyChatImportPromise: Promise<void> | null = null;

interface ExportedChat {
  exportedAt: string;
  version: 1;
  threadCount: number;
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
      error.message === "Studio isn't running -- please relaunch it.")
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

// Fast-path: ask IndexedDB whether the "unsloth-chat" database exists
// without opening it. Modern Chromium / Firefox / Safari support this;
// older browsers return undefined and we fall through to the next probe.
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
    // Dexie threw (corrupted DB / version mismatch / quota). Returning
    // false forces the slow path, which uses the same Dexie under the
    // hood; that path will throw too and the import promise gets reset
    // so the next caller can retry rather than silently doing nothing.
    return false;
  }
}

async function importLegacyChatsIfNeeded(): Promise<void> {
  // Session-level cache: same tab, repeated sidebar mounts share one
  // import. localStorage is NOT consulted here -- the server-side ledger
  // is the source of truth so a studio.db wipe still re-triggers the
  // import even if the browser kept its old hint.
  if (legacyChatImportPromise) return legacyChatImportPromise;

  legacyChatImportPromise = (async () => {
    // Fast-path: no Dexie database at all. New user, never had the
    // browser-only Studio. ~0.1 ms, zero network.
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

    // "Unimported" = missing from the ledger. We also include threads
    // already present in the backend (without a ledger row) so the ledger
    // gets backfilled for old-FE-then-new-FE users -- otherwise the next
    // launch would redo the diff for the same threads forever.
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
      // retries. The import itself is idempotent via UPSERT, no
      // duplicates.
      return;
    }
    // Only flip the localStorage hint when the backend actually has the
    // ledger. On older deployments (404/405/501) the hint would lie:
    // "import done" while the ledger stays empty, defeating recovery
    // when studio.db gets wiped later.
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
    .sort((a, b) => b.createdAt - a.createdAt);
}

export async function listStoredChatThreadsWithMessages(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const threads = await listStoredChatThreads(args);
  if (threads.length === 0) return [];
  // One batched HTTP call instead of N. Per-thread legacy Dexie
  // fallback only fires when the batch result is empty.
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

export async function saveStoredChatMessage(
  message: MessageRecord,
): Promise<MessageRecord> {
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
  if (isChatThreadDeleted(threadId)) return [];
  await ensureStoredChatThread(threadId);
  return syncChatMessages(threadId, messages, options);
}

export async function saveStoredChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord> {
  if (isChatThreadDeleted(thread.id)) {
    throw new Error(`Thread ${thread.id} was deleted`);
  }
  return saveChatThread(thread);
}

export async function updateStoredChatThread(
  threadId: string,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord | undefined> {
  const thread = await ensureStoredChatThread(threadId);
  if (!thread) return undefined;
  return updateChatThread(threadId, patch);
}

export async function deleteStoredChatThreads(
  idsToDelete: string[],
): Promise<void> {
  if (idsToDelete.length === 0) return;
  await deleteChatThreads(idsToDelete);
  await db
    .transaction("rw", db.threads, db.messages, async () => {
      await db.messages.where("threadId").anyOf(idsToDelete).delete();
      await db.threads.bulkDelete(idsToDelete);
    })
    .catch(() => undefined);
  markChatThreadsDeleted(idsToDelete);
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
  // Clear both sides independently and report each outcome so the
  // toast can distinguish full vs partial success.
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
    // Defer the history refresh until Dexie clear and tombstone state are
    // finalized, so listeners never observe the composite clear mid-flight.
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
    threads,
    messages,
  };
}
