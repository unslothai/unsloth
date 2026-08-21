// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A WebKit downgrade can leave legacy IndexedDB requests pending forever.
// Server-backed chat history must remain usable when that happens.

import assert from "node:assert/strict";
import test from "node:test";

import type { MessageRecord, ThreadRecord } from "../src/features/chat/types";
import { LegacyStoreGate } from "../src/features/chat/utils/legacy-store-gate.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Storage = {
  buildStoredChatExport: () => Promise<unknown>;
  clearStoredChats: () => Promise<{ legacy: string }>;
  deleteStoredChatThreads: (ids: string[]) => Promise<string[]>;
  getStoredChatMessage: (
    threadId: string,
    messageId: string,
  ) => Promise<MessageRecord | undefined>;
  getStoredChatThread: (threadId: string) => Promise<ThreadRecord | undefined>;
  legacyChatStoreResponds: () => boolean;
  listStoredChatMessages: (threadId: string) => Promise<MessageRecord[]>;
  listStoredChatThreads: () => Promise<ThreadRecord[]>;
};

const thread = (id: string, title: string, updatedAt: number): ThreadRecord => ({
  id,
  title,
  modelType: "base",
  archived: false,
  createdAt: 1,
  updatedAt,
});

const message = (id: string, threadId: string): MessageRecord => ({
  id,
  threadId,
  role: "user",
  content: [{ type: "text", text: "hi" }],
  createdAt: 1,
});

const never = () => new Promise<never>(() => {});
const stalledCollection = { toArray: never, count: never, delete: never };
const stalled = {
  get: never,
  toArray: never,
  count: never,
  bulkDelete: never,
  clear: never,
  toCollection: () => stalledCollection,
  where: () => ({
    equals: () => stalledCollection,
    anyOf: () => stalledCollection,
  }),
};

class StubWriteCoordinator {
  confirmFinalState() {}
  closeAdmission() {
    return () => {};
  }
  idsRequiringFence() {
    return [] as string[];
  }
  settleCurrent() {
    return Promise.resolve();
  }
}

class PassthroughGate {
  responds = true;
  read<T>(read: () => Promise<T>): Promise<T> {
    return read();
  }
}

function loadStorage(options: {
  db: unknown;
  serverThreads?: ThreadRecord[];
  serverMessages?: Map<string, MessageRecord[]>;
  listChatThreads?: () => Promise<ThreadRecord[]>;
  legacyImportComplete?: boolean;
  calls?: string[];
  gate?: unknown;
  appVersion?: string;
}): Storage {
  const serverThreads = options.serverThreads ?? [];
  const serverMessages =
    options.serverMessages ?? new Map<string, MessageRecord[]>();
  const record = (call: string) => options.calls?.push(call);
  return loadWithStubs<Storage>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "@tauri-apps/api/app": {
        getVersion: async () => options.appVersion ?? "test-version",
      },
      "../api/chat-api": {
        ChatThreadDeletedError: class extends Error {},
        buildBackendChatExport: async () => ({ threads: [], messages: [] }),
        clearBackendChats: async () => ({
          deletedThreadIds: [],
          sandboxesKept: [],
        }),
        deleteChatThreads: async (ids: string[]) => ids,
        getChatMessage: async (threadId: string, messageId: string) => {
          record(`getChatMessage:${threadId}:${messageId}`);
          return (
            (serverMessages.get(threadId) ?? []).find(
              (candidate) => candidate.id === messageId,
            ) ?? null
          );
        },
        getChatThread: async (threadId: string) => {
          record(`getChatThread:${threadId}`);
          return serverThreads.find((row) => row.id === threadId) ?? null;
        },
        listChatImportLedger: async () => ({
          threadIds: new Set<string>(),
          complete: options.legacyImportComplete ?? false,
          supported: true,
        }),
        listChatMessages: async (threadId: string) => {
          record(`listChatMessages:${threadId}`);
          return serverMessages.get(threadId) ?? [];
        },
        listChatThreads:
          options.listChatThreads ??
          (async () => {
            record("listChatThreads");
            return serverThreads;
          }),
        listChatThreadsWithMigrationState: async () => {
          record("listChatThreads");
          if (options.listChatThreads) {
            return {
              threads: await options.listChatThreads(),
              legacyImportComplete: options.legacyImportComplete ?? false,
            };
          }
          return {
            threads: serverThreads,
            legacyImportComplete: options.legacyImportComplete ?? false,
          };
        },
        notifyChatHistoryUpdated: () => {},
        recordChatImportLedger: async () => {
          record("recordChatImportLedger:complete");
          return {
            accepted: 0,
            inserted: 0,
            complete: true,
            supported: true,
          };
        },
      },
      "../db": { DEXIE_DB_NAME: "unsloth-chat", db: options.db },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "./legacy-store-gate": {
        LegacyStoreGate: options.gate ?? LegacyStoreGate,
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

test("legacy storage cannot delay any server chat read", async () => {
  const calls: string[] = [];
  const storage = loadStorage({
    db: { threads: stalled, messages: stalled },
    serverThreads: [thread("t1", "yesterday", 2), thread("t2", "last week", 4)],
    serverMessages: new Map([["t1", [message("m1", "t1")]]]),
    calls,
  });

  const threadsPromise = storage.listStoredChatThreads();
  const threadPromise = storage.getStoredChatThread("t1");
  const messagesPromise = storage.listStoredChatMessages("t1");
  const messagePromise = storage.getStoredChatMessage("t1", "m1");

  // Every server read starts before the legacy timeout.
  assert.deepEqual(calls, [
    "listChatThreads",
    "getChatThread:t1",
    "getChatThread:t1",
    "listChatMessages:t1",
    "getChatMessage:t1:m1",
  ]);

  const [threads, one, messages, single] = await Promise.all([
    threadsPromise,
    threadPromise,
    messagesPromise,
    messagePromise,
  ]);
  assert.deepEqual(
    threads.map((row) => row.id),
    ["t2", "t1"],
  );
  assert.equal(one?.title, "yesterday");
  assert.deepEqual(
    messages.map((row) => row.id),
    ["m1"],
  );
  assert.equal(single?.id, "m1");
  assert.equal(
    storage.legacyChatStoreResponds(),
    false,
    "the unreadable store should have been given up on",
  );
});

test("a store already given up on costs the next read nothing", async () => {
  const storage = loadStorage({ db: { threads: stalled, messages: stalled } });
  await storage.listStoredChatThreads();
  assert.equal(storage.legacyChatStoreResponds(), false);

  const started = Date.now();
  await storage.listStoredChatThreads();
  assert.ok(
    Date.now() - started < 500,
    "later reads must not pay the timeout again",
  );
});

test("an unreadable desktop store is retried only after the app version changes", async () => {
  const previousWindow = globalThis.window;
  const previousLocalStorage = globalThis.localStorage;
  const values = new Map<string, string>();
  Object.assign(globalThis, {
    window: {
      __TAURI_INTERNALS__: {},
      location: { protocol: "tauri:" },
    },
    localStorage: {
      getItem: (key: string) => values.get(key) ?? null,
      removeItem: (key: string) => values.delete(key),
      setItem: (key: string, value: string) => values.set(key, value),
    },
  });

  try {
    const firstLaunch = loadStorage({
      db: { threads: stalled, messages: stalled },
    });
    await firstLaunch.listStoredChatThreads();
    assert.equal(firstLaunch.legacyChatStoreResponds(), false);

    const secondLaunch = loadStorage({
      db: { threads: stalled, messages: stalled },
    });
    const started = Date.now();
    await secondLaunch.listStoredChatThreads();
    assert.ok(Date.now() - started < 500);
    assert.equal(
      secondLaunch.legacyChatStoreResponds(),
      true,
      "the second launch must not consult IndexedDB",
    );

    const updatedApp = loadStorage({
      db: { threads: stalled, messages: stalled },
      appVersion: "next-version",
    });
    await updatedApp.listStoredChatThreads();
    assert.equal(updatedApp.legacyChatStoreResponds(), false);
  } finally {
    if (previousWindow === undefined) Reflect.deleteProperty(globalThis, "window");
    else globalThis.window = previousWindow;
    if (previousLocalStorage === undefined) {
      Reflect.deleteProperty(globalThis, "localStorage");
    }
    else globalThis.localStorage = previousLocalStorage;
  }
});

test("a completed migration never opens legacy storage", async () => {
  const storage = loadStorage({
    db: { threads: stalled, messages: stalled },
    serverThreads: [thread("server", "saved", 2)],
    legacyImportComplete: true,
  });

  assert.deepEqual(
    (await storage.listStoredChatThreads()).map((row) => row.id),
    ["server"],
  );
  assert.equal(storage.legacyChatStoreResponds(), true);
});

test("an empty legacy database is completed and retired", async () => {
  const calls: string[] = [];
  const emptyCollection = {
    toArray: async () => [],
    count: async () => 0,
  };
  const storage = loadStorage({
    calls,
    db: {
      threads: { ...emptyCollection, toCollection: () => emptyCollection },
      messages: emptyCollection,
      close: () => calls.push("close"),
      delete: async () => calls.push("delete"),
    },
  });

  assert.deepEqual(await storage.listStoredChatThreads(), []);
  assert.ok(calls.includes("recordChatImportLedger:complete"));
  assert.ok(calls.includes("close"));
  assert.ok(calls.includes("delete"));
});

test("deleting, clearing and exporting still finish", async () => {
  const storage = loadStorage({
    db: { threads: stalled, messages: stalled, transaction: never },
  });
  assert.deepEqual(await storage.deleteStoredChatThreads(["t1"]), ["t1"]);
  assert.equal((await storage.clearStoredChats()).legacy, "failed");
  assert.ok(await storage.buildStoredChatExport());
});

test("legacy-only chats remain available when the server read fails", async () => {
  const legacyThread = thread("legacy", "before the server", 2);
  const legacyCollection = { toArray: async () => [legacyThread] };
  const storage = loadStorage({
    db: {
      threads: {
        toCollection: () => legacyCollection,
        where: () => ({ equals: () => legacyCollection }),
      },
    },
    listChatThreads: () => Promise.reject(new Error("server unavailable")),
    gate: PassthroughGate,
  });

  assert.deepEqual(await storage.listStoredChatThreads(), [legacyThread]);
});
