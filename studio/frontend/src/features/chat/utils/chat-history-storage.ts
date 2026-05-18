// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  buildBackendChatExport,
  clearBackendChats,
  deleteChatThreads,
  getChatThread,
  listChatMessages,
  listChatThreads,
  saveChatMessage,
  saveChatThread,
  syncChatMessages,
  updateChatThread,
} from "../api/chat-api";
import { db } from "../db";
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

async function listLegacyThreads(
  args: ThreadListArgs,
): Promise<ThreadRecord[]> {
  const legacyQuery = args.pairId
    ? db.threads.where("pairId").equals(args.pairId)
    : args.modelType
      ? db.threads.where("modelType").equals(args.modelType)
      : db.threads.toCollection();
  return (await legacyQuery.toArray()).filter(
    (thread) =>
      !isChatThreadDeleted(thread.id) &&
      (args.includeArchived !== false || !thread.archived),
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
): { messages: MessageRecord[]; shouldSync: boolean } {
  const byId = new Map<string, MessageRecord>();
  const backendIds = new Set(
    backendMessages
      .filter((message) => !isChatThreadDeleted(message.threadId))
      .map((message) => message.id),
  );
  let shouldSync = false;
  for (const message of normalizeLegacyMessages(legacyMessages)) {
    if (!isChatThreadDeleted(message.threadId)) {
      byId.set(message.id, message);
      if (!backendIds.has(message.id)) shouldSync = true;
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
  return (await updateChatThread(backendThread.id, patch)) ?? {
    ...backendThread,
    ...patch,
  };
}

async function importLegacyChatsIfNeeded(): Promise<void> {
  if (isLegacyChatImportDone()) return;
  if (legacyChatImportPromise) return legacyChatImportPromise;

  legacyChatImportPromise = (async () => {
    const [legacyThreads, backendThreads] = await Promise.all([
      db.threads.toArray(),
      listChatThreads({ includeArchived: true }),
    ]);
    const backendThreadsById = new Map(
      backendThreads.map((thread) => [thread.id, thread]),
    );

    for (const thread of legacyThreads) {
      if (isChatThreadDeleted(thread.id)) continue;
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

      const legacyMessages = await db.messages
        .where("threadId")
        .equals(thread.id)
        .toArray();
      if (legacyMessages.length === 0) continue;

      const backendMessages = await listChatMessages(thread.id).catch(() => []);
      const merged = mergeMessages(backendMessages, legacyMessages);
      if (merged.shouldSync) {
        await syncChatMessages(
          thread.id,
          sortMessages(merged.messages),
          { pruneMissing: false },
        );
      }
    }

    markLegacyChatImportDone();
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
  return legacyThread && !isChatThreadDeleted(legacyThread.id)
    ? legacyThread
    : undefined;
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
  return importLegacyThread(legacyThread);
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
    const merged = mergeMessages(backendMessages, legacyMessages);
    if (legacyMessages.length > 0 && merged.shouldSync) {
      return syncChatMessages(threadId, merged.messages, {
        pruneMissing: false,
      }).catch(() => merged.messages);
    }
    return merged.messages;
  }
  return legacyMessages.filter(
    (message) => !isChatThreadDeleted(message.threadId),
  );
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
  const byId = new Map<string, ThreadRecord>();
  for (const thread of legacyThreads) byId.set(thread.id, thread);
  for (const thread of backendThreads ?? []) {
    if (!isChatThreadDeleted(thread.id)) byId.set(thread.id, thread);
  }
  return Array.from(byId.values()).sort((a, b) => b.createdAt - a.createdAt);
}

export async function listStoredChatThreadsWithMessages(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const threads = await listStoredChatThreads(args);
  const entries = await Promise.all(
    threads.map(async (thread) => ({
      thread,
      messages: await listStoredChatMessages(thread.id),
    })),
  );
  return entries
    .filter((entry) => entry.messages.length > 0)
    .map((entry) => entry.thread);
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

export async function updateStoredChatThread(
  threadId: string,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord | undefined> {
  const thread = await ensureStoredChatThread(threadId);
  if (!thread) return undefined;
  return updateChatThread(threadId, patch);
}

export async function deleteStoredChatThreads(
  threadIds: string[],
): Promise<void> {
  if (threadIds.length === 0) return;
  await deleteChatThreads(threadIds);
  await db
    .transaction("rw", db.threads, db.messages, async () => {
      await db.messages.where("threadId").anyOf(threadIds).delete();
      await db.threads.bulkDelete(threadIds);
    })
    .catch(() => undefined);
  markChatThreadsDeleted(threadIds);
}

export async function countStoredChats(): Promise<number> {
  return (await listStoredChatThreads()).length;
}

export async function clearStoredChats(): Promise<void> {
  const [backendThreads, legacyThreads] = await Promise.all([
    listChatThreads().catch(() => []),
    db.threads.toArray(),
  ]);
  const threadIds = [...backendThreads, ...legacyThreads].map(
    (thread) => thread.id,
  );
  await clearBackendChats();
  await db
    .transaction("rw", db.threads, db.messages, async () => {
      await db.messages.clear();
      await db.threads.clear();
    })
    .catch(() => undefined);
  markChatThreadsDeleted(threadIds);
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
  for (const thread of legacyThreads as ThreadRecord[]) {
    if (isChatThreadDeleted(thread.id) || backendThreadIds.has(thread.id)) {
      continue;
    }
    threadsById.set(thread.id, thread);
  }
  for (const message of legacyMessages as MessageRecord[]) {
    if (isChatThreadDeleted(message.threadId)) {
      continue;
    }
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
