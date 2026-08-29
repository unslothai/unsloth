// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  MessageRecord,
  ThreadRecord,
  ThreadSidebarSummaryRecord,
} from "../src/features/chat/types";
import {
  newestSidebarAssistantUsageUpdate,
  selectSidebarLastRequestUsage,
  selectSidebarLastRequestUsageFromMetadata,
} from "../src/features/chat/lib/sidebar-last-request-usage.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type SidebarThread = ThreadRecord & {
  sidebarLastRequestUsage?: { totalTokens: number };
};

type Storage = {
  buildStoredChatExport: () => Promise<{
    threadCount: number;
    threads: unknown[];
    messages: unknown[];
  }>;
  clearStoredChats: () => Promise<{ backend: string; legacy: string }>;
  deleteStoredChatThreads: (ids: string[]) => Promise<string[]>;
  listStoredChatThreads: () => Promise<ThreadRecord[]>;
  listStoredChatThreadsWithMessages: () => Promise<SidebarThread[]>;
  listStoredChatThreadsWithSidebarUsage: () => Promise<SidebarThread[]>;
};

const thread = (id: string): ThreadRecord => ({
  id,
  title: id,
  modelType: "base",
  archived: false,
  createdAt: 1,
});

class StubWriteCoordinator {
  closeAdmission(): () => void {
    return () => {};
  }

  confirmFinalState(): void {}

  idsRequiringFence(): string[] {
    return [];
  }
}

function loadStorage(options: {
  messages?: unknown;
  threads: unknown;
  transaction?: (...args: unknown[]) => Promise<unknown>;
  listMessages?: (threadId: string) => Promise<MessageRecord[]>;
  listServerSummaries?: () => Promise<ThreadSidebarSummaryRecord[]>;
  listServerThreads: () => Promise<ThreadRecord[]>;
}): Storage {
  return loadWithStubs<Storage>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        buildBackendChatExport: async () => ({ threads: [], messages: [] }),
        ChatThreadDeletedError: class extends Error {},
        clearBackendChats: async () => ({
          deletedThreadIds: [],
          sandboxesKept: [],
        }),
        deleteChatThreads: async () => [],
        getChatThread: async (threadId: string) =>
          (await options.listServerThreads()).find((row) => row.id === threadId) ??
          null,
        listChatMessages: options.listMessages ?? (async () => []),
        listChatThreadSidebarSummaries:
          options.listServerSummaries ??
          (async () => {
            throw new Error("summary endpoint unavailable");
          }),
        listChatThreads: options.listServerThreads,
        notifyChatHistoryUpdated: () => {},
      },
      "../db": {
        DEXIE_DB_NAME: "unsloth-chat",
        db: {
          threads: options.threads,
          messages: options.messages ?? {},
          transaction: options.transaction ?? (async () => undefined),
        },
      },
      "../lib/sidebar-last-request-usage": {
        newestSidebarAssistantUsageUpdate,
        selectSidebarLastRequestUsage,
        selectSidebarLastRequestUsageFromMetadata,
      },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "./thread-record-write-coordinator": {
        ThreadRecordWriteCoordinator: StubWriteCoordinator,
      },
    },
  );
}

Object.assign(globalThis, {
  indexedDB: { databases: async () => [{ name: "unsloth-chat" }] },
});

async function settlesWithin<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("operation did not settle")),
          1_500,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

test("a stalled legacy store cannot hide server chats", async () => {
  const never = () => new Promise<never>(() => {});
  const stalledCollection = { toArray: never };
  let serverReads = 0;
  const storage = loadStorage({
    threads: { toCollection: () => stalledCollection },
    listServerThreads: async () => {
      serverReads += 1;
      return [thread("server")];
    },
  });

  const first = storage.listStoredChatThreads();
  assert.equal(serverReads, 1, "the server read must start immediately");
  assert.deepEqual(
    (await first).map((row) => row.id),
    ["server"],
  );

  const started = Date.now();
  await storage.listStoredChatThreads();
  assert.ok(Date.now() - started < 500, "the timeout must latch for the session");
});

test("readable legacy chats remain available when the server fails", async () => {
  const legacy = thread("legacy");
  const storage = loadStorage({
    threads: {
      toCollection: () => ({ toArray: async () => [legacy] }),
    },
    listServerThreads: async () => {
      throw new Error("server unavailable");
    },
  });

  assert.deepEqual(await storage.listStoredChatThreads(), [legacy]);
});

test("sidebar summaries keep empty chats without loading message bodies", async () => {
  let messageReads = 0;
  const empty = {
    ...thread("empty"),
    hasMessages: false,
    hasAssistant: false,
  } satisfies ThreadSidebarSummaryRecord;
  const used = {
    ...thread("used"),
    hasMessages: true,
    hasAssistant: true,
    lastAssistantMetadata: {
      contextUsage: {
        promptTokens: 10,
        completionTokens: 2,
        totalTokens: 12,
        cachedTokens: 0,
      },
    },
  } satisfies ThreadSidebarSummaryRecord;
  const storage = loadStorage({
    threads: { toCollection: () => ({ toArray: async () => [] }) },
    messages: {
      where: () => ({ equals: () => ({ toArray: async () => [] }) }),
    },
    listMessages: async () => {
      messageReads += 1;
      return [];
    },
    listServerSummaries: async () => [empty, used],
    listServerThreads: async () => [empty, used],
  });

  const all = await storage.listStoredChatThreadsWithSidebarUsage();
  assert.deepEqual(
    all.map((row) => [row.id, row.sidebarLastRequestUsage]),
    [
      ["empty", undefined],
      ["used", { totalTokens: 12 }],
    ],
  );
  assert.deepEqual(
    (await storage.listStoredChatThreadsWithMessages()).map((row) => row.id),
    ["used"],
  );
  assert.equal(messageReads, 0);
});

test("sidebar fallback derives usage from the newest loaded assistant", async () => {
  const saved = thread("saved");
  const assistant: MessageRecord = {
    id: "assistant",
    threadId: "saved",
    role: "assistant",
    content: [],
    metadata: {
      contextUsage: {
        promptTokens: 20,
        completionTokens: 5,
        totalTokens: 25,
      },
    },
    createdAt: 2,
  };
  const storage = loadStorage({
    threads: { toCollection: () => ({ toArray: async () => [] }) },
    messages: {
      where: () => ({ equals: () => ({ toArray: async () => [] }) }),
    },
    listMessages: async () => [assistant],
    listServerSummaries: async () => {
      throw new Error("summary endpoint unavailable");
    },
    listServerThreads: async () => [saved],
  });

  const [row] = await storage.listStoredChatThreadsWithSidebarUsage();
  assert.deepEqual(row.sidebarLastRequestUsage, { totalTokens: 25 });
});

test("legacy maintenance operations settle when the store stalls", async () => {
  const never = () => new Promise<never>(() => {});
  const stalledTable = { count: never, toArray: never };
  const storage = loadStorage({
    messages: stalledTable,
    threads: stalledTable,
    transaction: never,
    listServerThreads: async () => [],
  });

  assert.deepEqual(
    await settlesWithin(storage.deleteStoredChatThreads(["legacy"])),
    [],
  );
  assert.deepEqual(await settlesWithin(storage.clearStoredChats()), {
    backend: "cleared",
    legacy: "failed",
    deletedThreadIds: [],
    failedThreadIds: [],
    sandboxesKept: [],
  });
  const exported = await settlesWithin(storage.buildStoredChatExport());
  assert.equal(exported.threadCount, 0);
  assert.deepEqual(exported.threads, []);
  assert.deepEqual(exported.messages, []);
});
