// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChatMessageProtectedError,
  ChatThreadDeletedError,
  batchCountChatMessages,
  batchListChatMessages,
  buildBackendChatExport,
  clearBackendChats,
  deleteChatProject,
  deleteChatThreads,
  getChatMessage,
  getChatProject,
  getChatThread,
  listChatImportLedger,
  listChatMessages,
  listChatProjects,
  listChatThreads,
  notifyChatHistoryUpdated,
  recordChatImportLedger,
  saveChatMessage,
  saveChatProject,
  saveChatThread,
  syncChatMessages,
  updateChatProject,
  type ChatThreadWritePatch,
  updateChatThread,
} from "../api/chat-api";
import { DEXIE_DB_NAME, db } from "../db";
import type {
  MessageRecord,
  ModelType,
  ProjectRecord,
  ThreadRecord,
} from "../types";
import {
  isChatThreadDeleted,
  markChatThreadDeleted,
  markChatThreadsDeleted,
} from "./chat-thread-tombstones";
import { ThreadRecordWriteCoordinator } from "./thread-record-write-coordinator";
// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import { setForkBoundary } from "../stores/fork-boundary-store";

// Incognito is tagged per thread at creation, never read from the live toggle, so mid-stream
// toggling is safe.
const incognitoThreadIds = new Set<string>();

export function markThreadIncognito(threadId: string): void {
  incognitoThreadIds.add(threadId);
}

export function unmarkThreadIncognito(threadId: string): void {
  incognitoThreadIds.delete(threadId);
}

export function isThreadIncognito(threadId: string): boolean {
  return incognitoThreadIds.has(threadId);
}

// Read live: initialize() clears newThreadId before persisting the thread.
const newThreadIdSources = new Set<() => string | null | undefined>();
const writtenThreadIds = new Set<string>();

export function registerNewThreadIdSource(
  source: () => string | null | undefined,
): () => void {
  newThreadIdSources.add(source);
  return () => {
    newThreadIdSources.delete(source);
  };
}

function isPendingNewThread(threadId: string): boolean {
  if (writtenThreadIds.has(threadId)) return false;
  for (const source of newThreadIdSources) {
    if (source() === threadId) return true;
  }
  return false;
}

type ThreadListArgs = {
  modelType?: ModelType;
  pairId?: string;
  projectId?: string | null;
  includeArchived?: boolean;
};

// Perf hint only, not the import gate: the server ledger chat_legacy_imports is authoritative.
const LEGACY_CHAT_IMPORT_KEY = "unsloth_chat_legacy_imported_to_studio_db";

let legacyChatImportPromise: Promise<void> | null = null;

let legacyChatImportGeneration = 0;

// The delete tombstones before removing the row, so a confirmed delete beats a later save.
const threadRecordWrites = new ThreadRecordWriteCoordinator(
  (threadId) =>
    new Error(
      `Chat history was cleared before thread ${threadId} could be persisted`,
    ),
  (error) => error instanceof ChatThreadDeletedError,
);

const initializingThreadRecords = new Map<string, Promise<void>>();

// assistant-ui caches a resolved initialize(), so failed creators are retried from here.
const failedThreadRecordByThreadId = new Map<string, () => Promise<void>>();

// Bumped by a history clear so a retried creator cannot resurrect a removed thread.
let threadRecordClearEpoch = 0;

export function awaitStoredChatThreadWrites(threadId: string): Promise<void> {
  return threadRecordWrites.settleCurrent(threadId);
}

/** Starts one background initializer per id; returns the tracked write so a retry can adopt it. */
export function trackStoredChatThreadRecord(
  threadId: string,
  createRecord: () => Promise<void>,
): Promise<void> {
  const inFlight = initializingThreadRecords.get(threadId);
  if (inFlight) {
    return inFlight;
  }
  const epoch = threadRecordClearEpoch;
  const work = threadRecordWrites.observe(
    threadId,
    Promise.resolve().then(createRecord),
  );
  initializingThreadRecords.set(threadId, work);
  work.then(
    () => {
      if (initializingThreadRecords.get(threadId) === work) {
        initializingThreadRecords.delete(threadId);
      }
      failedThreadRecordByThreadId.delete(threadId);
    },
    () => {
      if (initializingThreadRecords.get(threadId) === work) {
        initializingThreadRecords.delete(threadId);
      }
      // A clear bumps the epoch first, so this creator retires without tombstoning the thread.
      if (epoch === threadRecordClearEpoch && !isChatThreadDeleted(threadId)) {
        failedThreadRecordByThreadId.set(threadId, createRecord);
      }
    },
  );
  return work;
}

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

class LegacyStoreGate {
  private available = true;
  private readonly timeoutMs: number;

  constructor(timeoutMs = 1_000) {
    this.timeoutMs = timeoutMs;
  }

  async read<T>(read: () => Promise<T>, fallback: T): Promise<T> {
    if (!this.available) return fallback;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        read(),
        new Promise<T>((resolve) => {
          timer = setTimeout(() => {
            this.available = false;
            resolve(fallback);
          }, this.timeoutMs);
        }),
      ]);
    } catch {
      this.available = false;
      return fallback;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }
}

const legacyStore = new LegacyStoreGate();
const legacyDatabaseList = new LegacyStoreGate();

function readLegacyStore<T>(read: () => Promise<T>, fallback: T): Promise<T> {
  return legacyStore.read(read, fallback);
}

async function listLegacyThreads(
  args: ThreadListArgs,
): Promise<ThreadRecord[]> {
  return readLegacyStore(async () => {
    const legacyQuery = args.pairId
      ? db.threads.where("pairId").equals(args.pairId)
      : args.modelType
        ? db.threads.where("modelType").equals(args.modelType)
        : db.threads.toCollection();
    return (await legacyQuery.toArray()).filter((thread) =>
      matchesThreadListArgs(thread, args),
    );
  }, []);
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
  // Match the transport marker, not the copy text, which may change.
  if (
    error instanceof Error &&
    (error as { unslothTransportFailure?: boolean }).unslothTransportFailure ===
      true
  ) {
    return true;
  }
  return (
    error instanceof Error &&
    (error.message === "Invalid or expired token" ||
      error.message === "Not authenticated" ||
      error.message === "Request failed (401)")
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

// Point imports commit before the generation bump, so listings wait on in-flight ones.
const pendingLegacyThreadImports = new Set<Promise<unknown>>();

function importLegacyThread(
  thread: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  const work = importLegacyThreadRow(thread);
  pendingLegacyThreadImports.add(work);
  const forget = () => pendingLegacyThreadImports.delete(work);
  work.then(forget, forget);
  return work;
}

async function importLegacyThreadRow(
  thread: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  const saved = await saveLegacyChatThread(thread);
  if (!saved) {
    return undefined;
  }
  legacyChatImportGeneration += 1;
  const legacyMessages = await readLegacyStore(
    () => db.messages.where("threadId").equals(thread.id).toArray(),
    [] as MessageRecord[],
  );
  if (legacyMessages.length > 0) {
    await syncChatMessages(thread.id, normalizeLegacyMessages(legacyMessages), {
      pruneMissing: false,
    });
  }
  return saved;
}

async function saveLegacyChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  try {
    return await writeChatThreadRecord(thread);
  } catch (error) {
    if (!(error instanceof ChatThreadDeletedError)) {
      throw error;
    }
    forgetChatThread(thread.id);
    return undefined;
  }
}

function writeChatThreadRecord(thread: ThreadRecord): Promise<ThreadRecord> {
  writtenThreadIds.add(thread.id);
  return threadRecordWrites.write(thread.id, () => saveChatThread(thread));
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
  const work = applyLegacyThreadBackfill(backendThread, patch);
  pendingLegacyThreadImports.add(work);
  const forget = () => pendingLegacyThreadImports.delete(work);
  work.then(forget, forget);
  return work;
}

async function applyLegacyThreadBackfill(
  backendThread: ThreadRecord,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord> {
  try {
    const updated = (await updateChatThread(backendThread.id, patch)) ?? {
      ...backendThread,
      ...patch,
    };
    // Bump only after the patch commits so a mid-flight listing re-reads.
    legacyChatImportGeneration += 1;
    return updated;
  } catch {
    return backendThread;
  }
}

// Older browsers return undefined here and fall through to the next probe.
async function dexieDbAbsent(): Promise<boolean> {
  if (typeof indexedDB === "undefined") return true;
  const dbs = (indexedDB as IDBFactory).databases;
  if (typeof dbs !== "function") return false;
  try {
    const list = await legacyDatabaseList.read<IDBDatabaseInfo[] | null>(
      () => dbs.call(indexedDB),
      null,
    );
    if (!Array.isArray(list)) return false;
    return !list.some((entry) => entry?.name === DEXIE_DB_NAME);
  } catch {
    return false;
  }
}

async function dexieIsEmpty(): Promise<boolean> {
  try {
    const counts = await readLegacyStore<[number, number] | null>(
      () => Promise.all([db.threads.count(), db.messages.count()]),
      null,
    );
    if (counts === null) return false;
    const [threadCount, messageCount] = counts;
    return threadCount === 0 && messageCount === 0;
  } catch {
    // Returning false forces the slow path, which throws and resets the promise for a retry.
    return false;
  }
}

async function importLegacyChatsIfNeeded(): Promise<void> {
  // Session cache only; localStorage is not consulted so a studio.db wipe re-imports.
  if (legacyChatImportPromise) return legacyChatImportPromise;

  legacyChatImportPromise = (async () => {
    if (await dexieDbAbsent()) {
      markLegacyChatImportDone();
      return;
    }

    if (await dexieIsEmpty()) {
      markLegacyChatImportDone();
      return;
    }

    const legacyThreads = await readLegacyStore(
      () => db.threads.toArray(),
      null,
    );
    if (legacyThreads === null) return;
    const [backendThreads, importedThreadIds] = await Promise.all([
      listChatThreads({ includeArchived: true }),
      listChatImportLedger(),
    ]);

    const backendThreadsById = new Map(
      backendThreads.map((thread) => [thread.id, thread]),
    );
    const unimportedIds: string[] = [];
    const unimportedThreads: ThreadRecord[] = [];

    // Include backend threads lacking a ledger row so they get backfilled, else re-diffs forever.
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

    const allLegacyMessages = await readLegacyStore<MessageRecord[] | null>(
      () => db.messages.where("threadId").anyOf(unimportedIds).toArray(),
      null,
    );
    if (allLegacyMessages === null) return;
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
      if (backendThread) {
        backendThreadsById.set(
          thread.id,
          await backfillLegacyThreadFields(backendThread, thread),
        );
      } else {
        const saved = await saveLegacyChatThread(thread);
        if (!saved) {
          // Record deletions in the ledger too, so stale Dexie copies stop retrying the import.
          newlyImportedIds.push(thread.id);
          continue;
        }
        backendThreadsById.set(thread.id, thread);
        legacyChatImportGeneration += 1;
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
      // Leave the hint so the next launch retries; import is idempotent via UPSERT.
      return;
    }
    // Only flip the hint when the backend has the ledger (older ones return 404/405/501).
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

export type StoredChatThreadReadResult = {
  thread: ThreadRecord | undefined;
  cacheable: boolean;
};

export async function getStoredChatThreadReadResult(
  threadId: string,
  options: { bounded?: boolean; timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<StoredChatThreadReadResult> {
  if (isThreadIncognito(threadId)) {
    return { thread: undefined, cacheable: true };
  }
  if (isChatThreadDeleted(threadId)) {
    return { thread: undefined, cacheable: true };
  }
  if (isPendingNewThread(threadId)) {
    return { thread: undefined, cacheable: true };
  }
  const legacyThread = await readLegacyStore(
    () => db.threads.get(threadId),
    undefined,
  );
  let backendThread: ThreadRecord | null;
  try {
    // Bounded: a never-answering GET would stay open for the page's life.
    backendThread = await getChatThread(threadId, {
      bounded: options.bounded,
      timeoutMs: options.timeoutMs,
      signal: options.signal,
    });
  } catch (error) {
    if (legacyThread && !isChatThreadDeleted(legacyThread.id)) {
      return { thread: legacyThread, cacheable: false };
    }
    throw error;
  }
  if (backendThread && !isChatThreadDeleted(backendThread.id)) {
    return {
      thread: await backfillLegacyThreadFields(backendThread, legacyThread),
      cacheable: true,
    };
  }
  if (!legacyThread || isChatThreadDeleted(legacyThread.id)) {
    return { thread: undefined, cacheable: true };
  }
  try {
    return { thread: await importLegacyThread(legacyThread), cacheable: true };
  } catch {
    return { thread: legacyThread, cacheable: false };
  }
}

export async function getStoredChatThread(
  threadId: string,
): Promise<ThreadRecord | undefined> {
  return (await getStoredChatThreadReadResult(threadId)).thread;
}

export async function ensureStoredChatThread(
  threadId: string,
  fallback?: ThreadRecord,
  options: { bounded?: boolean; signal?: AbortSignal } = {},
): Promise<ThreadRecord | undefined> {
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  // Outcome ignored on purpose so retryFailedThreadRecord still runs for waiting callers.
  await awaitStoredChatThreadWrites(threadId);
  const legacyThread =
    fallback ??
    (await readLegacyStore(() => db.threads.get(threadId), undefined));
  let backendThread: ThreadRecord | null;
  try {
    // Bounded: this runs before the caller's own deadline applies.
    backendThread = await getChatThread(threadId, {
      bounded: options.bounded,
      signal: options.signal,
    });
  } catch (error) {
    if (!legacyThread || isChatThreadDeleted(legacyThread.id)) {
      throw error;
    }
    return legacyThread;
  }
  if (backendThread) {
    return backfillLegacyThreadFields(backendThread, legacyThread);
  }
  if (!legacyThread || isChatThreadDeleted(legacyThread.id)) {
    return retryFailedThreadRecord(threadId);
  }
  return importLegacyThread(legacyThread).catch(() => legacyThread);
}

async function retryFailedThreadRecord(
  threadId: string,
): Promise<ThreadRecord | undefined> {
  const createRecord = failedThreadRecordByThreadId.get(threadId);
  if (!createRecord && !threadRecordWrites.hasPending(threadId)) {
    return undefined;
  }
  if (createRecord) {
    failedThreadRecordByThreadId.delete(threadId);
    // Rethrows on purpose: undefined reads as "no row" and callers would drop their patch.
    await trackStoredChatThreadRecord(threadId, createRecord);
  } else {
    await awaitStoredChatThreadWrites(threadId);
  }
  return (await getChatThread(threadId)) ?? undefined;
}

/** Backend-only record: null if absent, undefined if unreachable (not a deletion). Unlike
 *  getStoredChatThread, it never falls back to the legacy browser row. */
export async function readBackendChatThread(
  threadId: string,
): Promise<ThreadRecord | null | undefined> {
  if (isThreadIncognito(threadId)) return null;
  if (isChatThreadDeleted(threadId)) return null;
  try {
    return await getChatThread(threadId);
  } catch {
    return undefined;
  }
}

async function publishForkBoundary(
  thread: ThreadRecord,
  messages: readonly MessageRecord[],
): Promise<void> {
  let inherited = inheritedMessageIds(thread.forkBoundaryMessageId, messages);
  if (thread.forkBoundaryMessageId && inherited.size > 0) {
    // The anchor placed against these messages, which is the ordinary case.
  } else if (thread.forkBoundaryMessageId) {
    // Thread and messages are read in parallel and may disagree; re-read rather than blank it.
    const fresh = await getChatThread(thread.id).catch(() => undefined);
    if (!fresh) return;
    inherited = inheritedMessageIds(fresh.forkBoundaryMessageId, messages);
    if (fresh.forkBoundaryMessageId && inherited.size === 0) return;
  }
  setForkBoundary(
    thread.id,
    inherited,
    thread.forkedFromThreadId && !isChatThreadDeleted(thread.forkedFromThreadId)
      ? thread.forkedFromThreadId
      : null,
  );
}

/** The anchor and its ancestors; edits can branch the anchor off screen. */
function inheritedMessageIds(
  anchorId: string | null | undefined,
  messages: readonly MessageRecord[],
): Set<string> {
  const ids = new Set<string>();
  if (!anchorId) return ids;
  const byId = new Map(messages.map((message) => [message.id, message]));
  let cursor = byId.get(anchorId);
  // Stops on a repeat too, so a corrupt chain cannot spin.
  while (cursor !== undefined && !ids.has(cursor.id)) {
    ids.add(cursor.id);
    cursor = cursor.parentId ? byId.get(cursor.parentId) : undefined;
  }
  return ids;
}

export async function listStoredChatMessages(
  threadId: string,
): Promise<MessageRecord[]> {
  return (await readStoredChatMessages(threadId)).messages;
}

export async function readStoredChatMessages(
  threadId: string,
): Promise<{ messages: MessageRecord[]; fromBackend: boolean }> {
  if (isThreadIncognito(threadId)) return { messages: [], fromBackend: false };
  if (isChatThreadDeleted(threadId)) return { messages: [], fromBackend: false };
  if (isPendingNewThread(threadId)) return { messages: [], fromBackend: false };
  const legacyMessages = await readLegacyStore(
    () => db.messages.where("threadId").equals(threadId).toArray(),
    [] as MessageRecord[],
  );
  const [backendThread, backendMessages] = await Promise.all([
    getChatThread(threadId).catch(() => undefined),
    listChatMessages(threadId).catch((error) => {
      if (legacyMessages.length > 0) {
        return undefined;
      }
      throw error;
    }),
  ]);
  // No list means a failed read, not an empty thread, so the divider keeps its state.
  if (backendThread && backendMessages) {
    void publishForkBoundary(backendThread, backendMessages);
  }
  if (backendMessages && (backendThread || backendMessages.length > 0)) {
    const merged = mergeMessages(backendMessages, legacyMessages, {
      includeLegacyOnly:
        !isLegacyChatImportDone() ||
        (backendMessages.length === 0 && legacyMessages.length > 0),
    });
    if (legacyMessages.length > 0 && merged.shouldSync) {
      const messages = await syncChatMessages(threadId, merged.messages, {
        pruneMissing: false,
      }).catch(() => merged.messages);
      return { messages, fromBackend: true };
    }
    return { messages: merged.messages, fromBackend: true };
  }
  if (
    backendMessages &&
    isLegacyChatImportDone() &&
    legacyMessages.length === 0
  ) {
    return { messages: [], fromBackend: true };
  }
  return {
    messages: legacyMessages.filter(
      (message) => !isChatThreadDeleted(message.threadId),
    ),
    fromBackend: false,
  };
}

export async function getStoredChatMessage(
  threadId: string,
  messageId: string,
): Promise<MessageRecord | undefined> {
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  const legacyMessage = await readLegacyStore(
    () => db.messages.get(messageId),
    undefined,
  );
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
  const importGenerationBeforeRead = legacyChatImportGeneration;
  const [legacyThreads, backendResult] = await Promise.all([
    listLegacyThreads(args),
    listChatThreads(args).then(
      (threads) => ({ threads }),
      (error: unknown) => ({ error }),
    ),
  ]);
  if ("error" in backendResult && legacyThreads.length === 0) {
    throw backendResult.error;
  }
  let backendThreads =
    "threads" in backendResult ? backendResult.threads : undefined;
  if (backendThreads) {
    await importLegacyChatsIfNeeded().catch(() => undefined);
    // Point imports commit before the generation bump, so wait on in-flight ones.
    await Promise.allSettled([...pendingLegacyThreadImports]);
    if (legacyChatImportGeneration !== importGenerationBeforeRead) {
      backendThreads = await listChatThreads(args).catch(() => backendThreads);
    }
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
      (a, b) => (b.updatedAt ?? b.createdAt) - (a.updatedAt ?? a.createdAt),
    );
}

export async function listStoredChatThreadsWithMessages(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const threads = await listStoredChatThreads(args);
  if (threads.length === 0) return [];
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
): Promise<string[]> {
  return deleteChatProject(projectId, args);
}

export async function moveStoredChatItemToProject(
  item: { type: "single" | "compare"; id: string },
  projectId: string | null,
): Promise<void> {
  const threadIds =
    item.type === "single"
      ? [item.id]
      : (
          await listStoredChatThreads({
            pairId: item.id,
            includeArchived: true,
          })
        ).map((thread) => thread.id);

  await Promise.all(
    Array.from(new Set(threadIds)).map((threadId) =>
      updateStoredChatThread(threadId, { projectId }),
    ),
  );
}

// Keyed by payload, not id: a 409 may be a transient generationSeq race.
const rejectedChatMessagePayloads = new Map<string, Map<string, string>>();

// Only delete paths clear entries; overflow costs one extra request.
const MAX_REJECTED_PAYLOADS = 32;

function evictOldestRejectedPayloads(): void {
  let total = 0;
  for (const perThread of rejectedChatMessagePayloads.values()) {
    total += perThread.size;
  }
  while (total > MAX_REJECTED_PAYLOADS) {
    const oldestThread = rejectedChatMessagePayloads.entries().next().value;
    if (!oldestThread) return;
    const [threadId, perThread] = oldestThread;
    const oldestMessage = perThread.keys().next().value;
    if (oldestMessage === undefined) {
      rejectedChatMessagePayloads.delete(threadId);
      continue;
    }
    perThread.delete(oldestMessage);
    if (perThread.size === 0) rejectedChatMessagePayloads.delete(threadId);
    total -= 1;
  }
}

function rememberRejectedPayload(
  threadId: string,
  messageId: string,
  payload: string,
): void {
  const perThread = rejectedChatMessagePayloads.get(threadId) ?? new Map<string, string>();
  perThread.delete(messageId);
  perThread.set(messageId, payload);
  rejectedChatMessagePayloads.delete(threadId);
  rejectedChatMessagePayloads.set(threadId, perThread);
  evictOldestRejectedPayloads();
}

/** Deterministic JSON: key order must not decide whether two payloads look equal. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, entry]) => entry !== undefined)
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0));
  return `{${entries
    .map(([key, entry]) => `${JSON.stringify(key)}:${stableStringify(entry)}`)
    .join(",")}}`;
}

// Clear every entry: collisions are recorded under the thread we wrote to.
export function clearServerOwnedChatMessages(): void {
  rejectedChatMessagePayloads.clear();
}

function forgetChatThread(threadId: string): void {
  markChatThreadDeleted(threadId);
  clearServerOwnedChatMessages();
}

function forgetChatThreads(threadIds: string[]): void {
  markChatThreadsDeleted(threadIds);
  clearServerOwnedChatMessages();
}

export async function saveStoredChatMessage(
  message: MessageRecord,
): Promise<MessageRecord> {
  if (isThreadIncognito(message.threadId)) return message;
  if (isChatThreadDeleted(message.threadId)) {
    throw new Error(`Thread ${message.threadId} was deleted`);
  }
  const payload = stableStringify(message);
  if (rejectedChatMessagePayloads.get(message.threadId)?.get(message.id) === payload) {
    rememberRejectedPayload(message.threadId, message.id, payload);
    return message;
  }
  await ensureStoredChatThread(message.threadId);
  try {
    return await saveChatMessage(message, { coalesce: true });
  } catch (error) {
    if (error instanceof ChatMessageProtectedError) {
      rememberRejectedPayload(message.threadId, message.id, payload);
      return message;
    }
    throw error;
  }
}

export async function syncStoredChatMessages(
  threadId: string,
  messages: MessageRecord[],
  options: { pruneMissing?: boolean; deletedMessageIds?: string[] } = {},
): Promise<MessageRecord[]> {
  if (isThreadIncognito(threadId)) return messages;
  if (isChatThreadDeleted(threadId)) return [];
  await ensureStoredChatThread(threadId);
  const synced = await syncChatMessages(threadId, messages, options);
  // Only on actual deletions; ordinary syncs run constantly and would wipe the cache.
  if (options.pruneMissing || (options.deletedMessageIds?.length ?? 0) > 0) {
    clearServerOwnedChatMessages();
    // Deleting the divider's message moves it; refresh here since nothing else re-reads the thread.
    const thread = await getChatThread(threadId).catch(() => undefined);
    if (thread) await publishForkBoundary(thread, synced);
  }
  return synced;
}

export async function saveStoredChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord> {
  if (isThreadIncognito(thread.id)) return thread;
  if (isChatThreadDeleted(thread.id)) {
    throw new Error(`Thread ${thread.id} was deleted`);
  }
  try {
    return await writeChatThreadRecord(thread);
  } catch (error) {
    if (error instanceof ChatThreadDeletedError) {
      forgetChatThread(thread.id);
    }
    throw error;
  }
}

export async function updateStoredChatThread(
  threadId: string,
  patch: ChatThreadWritePatch,
  options: { notify?: boolean; signal?: AbortSignal } = {},
): Promise<ThreadRecord | undefined> {
  if (isThreadIncognito(threadId)) return undefined;
  // Same bound as the following write, or the settings write chain can hang.
  const thread = await ensureStoredChatThread(threadId, undefined, {
    bounded: true,
    signal: options.signal,
  });
  if (!thread) return undefined;
  return updateChatThread(threadId, patch, options);
}

export async function countStoredChatMessages(
  threadIds: string[],
): Promise<Map<string, number> | null> {
  const ids = threadIds.filter((id) => !isThreadIncognito(id) && !isChatThreadDeleted(id));
  const counts = await batchCountChatMessages(ids);
  if (!counts) return null;
  await Promise.all(
    ids
      .filter((id) => (counts.get(id) ?? 0) === 0)
      .map(async (id) => {
        const legacy = await readLegacyStore(
          () => db.messages.where("threadId").equals(id).toArray(),
          [] as MessageRecord[],
        );
        const n = legacy.filter((m) => m.role === "user" || m.role === "assistant").length;
        if (n > 0) counts.set(id, n);
      }),
  );
  return counts;
}

export async function listStoredChatMessagesMany(
  threadIds: string[],
): Promise<Map<string, MessageRecord[]>> {
  const ids = threadIds.filter((id) => !isThreadIncognito(id) && !isChatThreadDeleted(id));
  const out = await batchListChatMessages(ids).catch(() => new Map<string, MessageRecord[]>());
  await Promise.all(
    ids
      .filter((id) => (out.get(id)?.length ?? 0) === 0)
      .map(async (id) => out.set(id, await listStoredChatMessages(id).catch(() => []))),
  );
  return out;
}

export async function deleteStoredChatThreads(
  idsToDelete: string[],
  args: { deleteFiles?: boolean } = {},
): Promise<string[]> {
  // Incognito ids still name sandboxes, so send all ids for file cleanup.
  idsToDelete = Array.from(new Set(idsToDelete));
  const ids = idsToDelete.filter((id) => !isThreadIncognito(id));
  if (idsToDelete.length === 0) return [];
  let kept: string[] = [];
  try {
    kept = await deleteChatThreads(idsToDelete, args);
  } catch (error) {
    if (ids.length === 0) throw error;
    // An aborted response is not proof of failure; confirm rows survived before the caller rolls back.
    const survived = await Promise.all(
      ids.map((id) =>
        getChatThread(id, { bounded: true }).then(
          (thread) => thread !== null,
          () => true,
        ),
      ),
    );
    if (survived.some(Boolean)) {
      throw error;
    }
  }
  for (const id of ids) {
    failedThreadRecordByThreadId.delete(id);
    initializingThreadRecords.delete(id);
  }
  threadRecordWrites.confirmFinalState(ids);
  if (ids.length === 0) return kept;
  await readLegacyStore(
    () =>
      db
        .transaction("rw", db.threads, db.messages, async () => {
          await db.messages.where("threadId").anyOf(ids).delete();
          await db.threads.bulkDelete(ids);
        })
        .catch(() => undefined),
    undefined,
  );
  forgetChatThreads(ids);
  return kept;
}

export async function countStoredChats(): Promise<number> {
  return (await listStoredChatThreads()).length;
}

export interface ClearStoredChatsResult {
  backend: "cleared" | "failed" | "skipped";
  legacy: "cleared" | "failed" | "skipped";
  deletedThreadIds: string[];
  failedThreadIds: string[];
  sandboxesKept: string[];
}

let clearStoredChatsPromise: Promise<ClearStoredChatsResult> | null = null;

export function clearStoredChats(
  options: { deleteFiles?: boolean } = {},
): Promise<ClearStoredChatsResult> {
  // Dedupe in-flight clears so two cannot race.
  if (clearStoredChatsPromise) return clearStoredChatsPromise;

  threadRecordClearEpoch += 1;
  failedThreadRecordByThreadId.clear();
  const reopenAdmission = threadRecordWrites.closeAdmission();
  const operation = clearStoredChatsWithAdmissionClosed(options);
  const tracked = operation.finally(() => {
    reopenAdmission();
    if (clearStoredChatsPromise === tracked) {
      clearStoredChatsPromise = null;
    }
  });
  clearStoredChatsPromise = tracked;
  return tracked;
}

async function clearStoredChatsWithAdmissionClosed(
  options: { deleteFiles?: boolean },
): Promise<ClearStoredChatsResult> {
  // Admission is closed before this one-shot fence snapshot.
  const pendingThreadIds = threadRecordWrites.idsRequiringFence();
  const operationId = crypto.randomUUID();
  const legacyThreads = await readLegacyStore(
    () => db.threads.toArray(),
    [] as ThreadRecord[],
  );
  const legacyThreadIds = new Set(legacyThreads.map((thread) => thread.id));
  const idsToFence = Array.from(
    new Set([...legacyThreadIds, ...pendingThreadIds]),
  );

  const result: ClearStoredChatsResult = {
    backend: "skipped",
    legacy: "skipped",
    deletedThreadIds: [],
    failedThreadIds: [],
    sandboxesKept: [],
  };
  let backendDeletedThreadIds: string[] = [];
  const runBackendClear = () =>
    clearBackendChats({
      notify: false,
      operationId,
      deleteFiles: options.deleteFiles,
      tombstoneThreadIds: idsToFence,
    });
  try {
    // Retry once with the same operationId: a timeout is not proof it did not run, and the retry
    // replays the recorded result.
    const backendResult = await runBackendClear().catch(() =>
      runBackendClear(),
    );
    backendDeletedThreadIds = backendResult.deletedThreadIds;
    result.sandboxesKept = backendResult.sandboxesKept;
    result.backend = "cleared";
    threadRecordWrites.confirmFinalState(idsToFence);
  } catch (error) {
    result.backend = "failed";
    console.error("clearStoredChats: backend clear failed", error);
  }

  const legacyCleared = await readLegacyStore(
    () =>
      db
        .transaction("rw", db.threads, db.messages, async () => {
          await db.messages.clear();
          await db.threads.clear();
        })
        .then(() => true)
        .catch((error) => {
          console.error("clearStoredChats: legacy Dexie clear failed", error);
          return false;
        }),
    false,
  );
  result.legacy = legacyCleared ? "cleared" : "failed";

  // Report removed rows, not the fence set: fenced ids may never have had a chat.
  const allThreadIds = Array.from(
    new Set([...legacyThreadIds, ...backendDeletedThreadIds]),
  );
  result.deletedThreadIds =
    result.backend === "cleared"
      ? allThreadIds.filter(
          (id) => !legacyThreadIds.has(id) || result.legacy === "cleared",
        )
      : [];
  const deleted = new Set(result.deletedThreadIds);
  result.failedThreadIds = allThreadIds.filter((id) => !deleted.has(id));

  forgetChatThreads(result.deletedThreadIds);
  notifyChatHistoryUpdated();

  if (result.backend === "failed" && result.legacy === "failed") {
    throw new Error("clearStoredChats: both backend and legacy clear failed");
  }
  return result;
}

export async function buildStoredChatExport(): Promise<ExportedChat> {
  await importLegacyChatsIfNeeded().catch(() => undefined);
  const [legacyThreads, legacyMessages] = await readLegacyStore<
    [ThreadRecord[], MessageRecord[]]
  >(() => Promise.all([db.threads.toArray(), db.messages.toArray()]), [[], []]);
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
