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

interface ExportedChat {
  exportedAt: string;
  version: 1;
  threadCount: number;
  threads: unknown[];
  messages: unknown[];
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

function firstRejected(results: PromiseSettledResult<unknown>[]): unknown {
  return results.find((result) => result.status === "rejected")?.reason;
}

function mergeMessages(
  backendMessages: MessageRecord[],
  legacyMessages: MessageRecord[],
): MessageRecord[] {
  const byId = new Map<string, MessageRecord>();
  for (const message of legacyMessages) {
    if (!isChatThreadDeleted(message.threadId)) {
      byId.set(message.id, message);
    }
  }
  for (const message of backendMessages) {
    if (!isChatThreadDeleted(message.threadId)) {
      byId.set(message.id, message);
    }
  }
  return Array.from(byId.values());
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
    await syncChatMessages(thread.id, legacyMessages, { pruneMissing: false });
  }
  return saved;
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
    return backendThread;
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
    return importLegacyThread(legacyThread);
  }
  if (backendThread) return backendThread;
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
    if (legacyMessages.length > 0 && merged.length > backendMessages.length) {
      return syncChatMessages(threadId, merged, { pruneMissing: false });
    }
    return merged;
  }
  return legacyMessages.filter(
    (message) => !isChatThreadDeleted(message.threadId),
  );
}

export async function listStoredChatThreads(
  args: ThreadListArgs = {},
): Promise<ThreadRecord[]> {
  const legacyThreads = await listLegacyThreads(args);
  const backendThreads = await listChatThreads(args).catch((error) => {
    if (legacyThreads.length > 0) {
      return undefined;
    }
    throw error;
  });
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
  markChatThreadsDeleted(threadIds);
  const results = await Promise.allSettled([
    deleteChatThreads(threadIds),
    db.transaction("rw", db.threads, db.messages, async () => {
      await db.messages.where("threadId").anyOf(threadIds).delete();
      await db.threads.bulkDelete(threadIds);
    }),
  ]);
  const error = firstRejected(results);
  if (error) throw error;
}

export async function countStoredChats(): Promise<number> {
  return (await listStoredChatThreads()).length;
}

export async function clearStoredChats(): Promise<void> {
  const [backendThreads, legacyThreads] = await Promise.all([
    listChatThreads().catch(() => []),
    db.threads.toArray(),
  ]);
  markChatThreadsDeleted(
    [...backendThreads, ...legacyThreads].map((thread) => thread.id),
  );
  const results = await Promise.allSettled([
    clearBackendChats(),
    db.transaction("rw", db.threads, db.messages, async () => {
      await db.messages.clear();
      await db.threads.clear();
    }),
  ]);
  const error = firstRejected(results);
  if (error) throw error;
}

export async function buildStoredChatExport(): Promise<ExportedChat> {
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
