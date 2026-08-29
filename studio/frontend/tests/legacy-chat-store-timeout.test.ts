// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ThreadRecord } from "../src/features/chat/types";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Storage = {
  buildStoredChatExport: () => Promise<{
    threadCount: number;
    threads: unknown[];
    messages: unknown[];
  }>;
  clearStoredChats: () => Promise<{ backend: string; legacy: string }>;
  deleteStoredChatThreads: (ids: string[]) => Promise<string[]>;
  listStoredChatThreads: () => Promise<ThreadRecord[]>;
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
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "@/lib/chat-history-policy": {
        chatHistoryDisabledError: () => new Error("Chat history is disabled"),
        isChatHistoryDisabled: () => false,
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
