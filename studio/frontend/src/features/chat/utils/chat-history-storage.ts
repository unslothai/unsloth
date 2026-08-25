// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChatThreadDeletedError,
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

/** True for a temporary-session thread, which is deliberately never persisted. */
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

// Bumped whenever a backend thread row is created or backfilled, so a listing can tell whether its read raced one.
let legacyChatImportGeneration = 0;

// no browser-side ordering: the delete transaction tombstones before removing the row, so a
// confirmed delete always beats a save that reaches sqlite later
const threadRecordWrites = new ThreadRecordWriteCoordinator(
  (threadId) =>
    new Error(
      `Chat history was cleared before thread ${threadId} could be persisted`,
    ),
  (error) => error instanceof ChatThreadDeletedError,
);

const initializingThreadRecords = new Map<string, Promise<void>>();

// creators whose last write failed, since assistant-ui caches a resolved initialize() and never re-asks
const failedThreadRecordByThreadId = new Map<string, () => Promise<void>>();

// bumped by a history clear, so a retried creator cannot resurrect a thread the user removed
let threadRecordClearEpoch = 0;

/** Wait for the row work already registered for this thread, without adopting its failure. */
export function awaitStoredChatThreadWrites(threadId: string): Promise<void> {
  return threadRecordWrites.settleCurrent(threadId);
}

/** Start one background initializer for an id so the first message can render at once.
 * Returns the tracked write so a retry can adopt its outcome; callers that only start one
 * ignore it, and the rejection is always handled below. */
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
      // a clear bumps the epoch before closing admission, so it retires this creator without
      // tombstoning a thread the clear may yet fail to remove
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

// point imports commit their row before the bump lands, so a listing waits for the ones already
// in flight instead of trusting the generation alone
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
  // A point lookup can import a row too, and a listing mid-flight has to re-read to see it.
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

/** A backend tombstone is authoritative: retire the stale browser row instead of retrying it. */
async function saveLegacyChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord | undefined> {
  try {
    return await writeChatThreadRecord(thread);
  } catch (error) {
    if (!(error instanceof ChatThreadDeletedError)) {
      throw error;
    }
    markChatThreadDeleted(thread.id);
    return undefined;
  }
}

function writeChatThreadRecord(thread: ThreadRecord): Promise<ThreadRecord> {
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
    // Bump only once the patch has committed: a listing that read the row mid-flight still needs
    // to see the generation move so it re-reads.
    legacyChatImportGeneration += 1;
    return updated;
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

// Fast-path: Dexie exists but is empty. count() reads the IndexedDB
// store metadata, not the rows -- cheap regardless of record count.
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
          // Record the authoritative deletion in the migration ledger too, so another browser's
          // stale Dexie copy does not keep attempting the same forbidden import.
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

export type StoredChatThreadReadResult = {
  thread: ThreadRecord | undefined;
  cacheable: boolean;
};

export async function getStoredChatThreadReadResult(
  threadId: string,
  options: { bounded?: boolean; timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<StoredChatThreadReadResult> {
  // Incognito threads are never stored, so the lookup can only come back
  // empty -- short-circuit it instead of doing a Dexie read + backend GET.
  if (isThreadIncognito(threadId)) {
    return { thread: undefined, cacheable: true };
  }
  if (isChatThreadDeleted(threadId)) {
    return { thread: undefined, cacheable: true };
  }
  const legacyThread = await readLegacyStore(
    () => db.threads.get(threadId),
    undefined,
  );
  let backendThread: ThreadRecord | null;
  try {
    // Bounded for a caller that is gating the UI on this read: an unbounded GET that
    // never answers leaves the request open for the life of the page, and every retry
    // opens another.
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
  // An incognito thread is never persisted, so there's genuinely nothing
  // to ensure -- skip the backend round-trips this would otherwise make
  // on every autosave (runStart/runEnd) and message append.
  if (isThreadIncognito(threadId)) return undefined;
  if (isChatThreadDeleted(threadId)) return undefined;
  // Outcome ignored on purpose: adopting the failure here would skip the retryFailedThreadRecord
  // branch below for exactly the callers already waiting when the write rejected.
  await awaitStoredChatThreadWrites(threadId);
  const legacyThread =
    fallback ??
    (await readLegacyStore(() => db.threads.get(threadId), undefined));
  let backendThread: ThreadRecord | null;
  try {
    // Bounded for a caller whose own request carries a deadline: this read runs BEFORE
    // it, so an unbounded one here means neither the caller's signal nor the write
    // timeout ever applies and the write chain behind it never settles.
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

/** Re-run a row write that failed, for a thread the reads above could not find. */
async function retryFailedThreadRecord(
  threadId: string,
): Promise<ThreadRecord | undefined> {
  const createRecord = failedThreadRecordByThreadId.get(threadId);
  if (!createRecord && !threadRecordWrites.hasPending(threadId)) {
    return undefined;
  }
  if (createRecord) {
    failedThreadRecordByThreadId.delete(threadId);
    // Through the same initializer path, so a retry that fails again stays retryable, and
    // rethrowing on purpose: a caller handed undefined reads it as "no row to update" and drops
    // its patch, which is how the prompt queue loses its model correction.
    await trackStoredChatThreadRecord(threadId, createRecord);
  } else {
    await awaitStoredChatThreadWrites(threadId);
  }
  return (await getChatThread(threadId)) ?? undefined;
}

export async function listStoredChatMessages(
  threadId: string,
): Promise<MessageRecord[]> {
  if (isThreadIncognito(threadId)) return [];
  if (isChatThreadDeleted(threadId)) return [];
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
    // a point import can commit its row before its generation bump lands, so wait on the ones
    // already in flight rather than trusting the generation alone
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

export async function saveStoredChatMessage(
  message: MessageRecord,
): Promise<MessageRecord> {
  if (isThreadIncognito(message.threadId)) return message;
  if (isChatThreadDeleted(message.threadId)) {
    throw new Error(`Thread ${message.threadId} was deleted`);
  }
  await ensureStoredChatThread(message.threadId);
  // The per-chunk autosave behind a streaming response.
  return saveChatMessage(message, { coalesce: true });
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
  try {
    return await writeChatThreadRecord(thread);
  } catch (error) {
    if (error instanceof ChatThreadDeletedError) {
      markChatThreadDeleted(thread.id);
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
  // Same bound and same signal as the write it precedes: a stall here left the settings
  // write chain pending for the life of the page, and reopening or forking that chat
  // waits on that chain.
  const thread = await ensureStoredChatThread(threadId, undefined, {
    bounded: true,
    signal: options.signal,
  });
  if (!thread) return undefined;
  return updateChatThread(threadId, patch, options);
}

/** Thread ids whose sandbox still holds files, passed through from the route. */
export async function deleteStoredChatThreads(
  idsToDelete: string[],
  args: { deleteFiles?: boolean } = {},
): Promise<string[]> {
  // Incognito chats have no history row, but their ids still name sandboxes. Send every id to
  // the backend for file cleanup while limiting Dexie and write-coordinator work to stored chats.
  idsToDelete = Array.from(new Set(idsToDelete));
  const ids = idsToDelete.filter((id) => !isThreadIncognito(id));
  if (idsToDelete.length === 0) return [];
  let kept: string[] = [];
  // the backend tombstones every requested id in the transaction that deletes its row, so a save
  // reaching sqlite later is rejected rather than resurrecting the thread
  try {
    kept = await deleteChatThreads(idsToDelete, args);
  } catch (error) {
    // With only incognito ids there is no row whose absence can reconcile an ambiguous response.
    if (ids.length === 0) throw error;
    // an aborted or dropped response is not proof the delete failed. the caller rolls its
    // tombstone back on a throw, and doing that for a row the backend did remove leaves the
    // thread 410 on every later write, so confirm the rows really survived first.
    // Bounded: the DELETE only got here by aborting on a wedged socket, and an unbounded read
    // would hang the delete instead. A read that cannot answer counts the row as surviving.
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
  markChatThreadsDeleted(ids);
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
  /** Ids whose sandbox still holds files, so the offer can be made once. */
  sandboxesKept: string[];
}

let clearStoredChatsPromise: Promise<ClearStoredChatsResult> | null = null;

export function clearStoredChats(
  options: { deleteFiles?: boolean } = {},
): Promise<ClearStoredChatsResult> {
  // A clear already in flight wins: the dedupe is what keeps two clears from
  // racing, and only one caller can start one.
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
      // the transaction finds existing rows itself; these ids additionally fence legacy rows and
      // writes that have not committed yet
      tombstoneThreadIds: idsToFence,
    });
  try {
    // Retried once under the same operationId, still with admission closed. A request that timed
    // out is not proof its transaction did not run, and the retry takes the writer lock behind
    // that transaction and replays its recorded result, so admission cannot reopen into a window
    // where a new chat is created and then deleted by a clear that lands late.
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

  // reported from the rows the backend says it removed, never from the fence set: an id fenced for
  // a write that never committed had no chat to delete
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

  markChatThreadsDeleted(result.deletedThreadIds);
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
