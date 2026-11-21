// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ThreadRecord } from "../src/features/chat/types";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Storage = {
  registerNewThreadIdSource: (
    source: () => string | null | undefined,
  ) => () => void;
  getStoredChatThreadReadResult: (
    threadId: string,
  ) => Promise<{ thread: ThreadRecord | undefined; cacheable: boolean }>;
  listStoredChatMessages: (threadId: string) => Promise<unknown[]>;
  saveStoredChatThread: (thread: ThreadRecord) => Promise<ThreadRecord>;
};

const NEW_ID = "__LOCALID_unsent";

const thread = (id: string): ThreadRecord => ({
  id,
  title: "New Chat",
  modelType: "base",
  archived: false,
  createdAt: 1,
});

function loadStorage() {
  const rows = new Map<string, ThreadRecord>();
  const requests: string[] = [];
  const storage = loadWithStubs<Storage>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        ChatThreadDeletedError: class extends Error {},
        getChatThread: (threadId: string) => {
          requests.push(`GET thread ${threadId}`);
          return Promise.resolve(rows.get(threadId) ?? null);
        },
        listChatMessages: (threadId: string) => {
          requests.push(`GET messages ${threadId}`);
          return Promise.resolve([]);
        },
        saveChatThread: (record: ThreadRecord) => {
          rows.set(record.id, record);
          return Promise.resolve(record);
        },
        notifyChatHistoryUpdated: () => {},
      },
      "../db": {
        DEXIE_DB_NAME: "unsloth-chat",
        db: {
          threads: { get: () => Promise.resolve(undefined) },
          messages: {
            where: () => ({
              equals: () => ({ toArray: () => Promise.resolve([]) }),
            }),
          },
        },
      },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "./thread-record-write-coordinator": {
        ThreadRecordWriteCoordinator: class {
          write<T>(_threadId: string, run: () => Promise<T>): Promise<T> {
            return run();
          }
        },
      },
      "../stores/fork-boundary-store": { setForkBoundary: () => {} },
    },
  );
  return { storage, requests };
}

Object.assign(globalThis, {
  indexedDB: { databases: () => Promise.resolve([{ name: "unsloth-chat" }]) },
});

test("a runtime's unsaved new thread is answered as missing without a request", async () => {
  const { storage, requests } = loadStorage();
  storage.registerNewThreadIdSource(() => NEW_ID);
  assert.deepEqual(await storage.getStoredChatThreadReadResult(NEW_ID), {
    thread: undefined,
    cacheable: true,
  });
  assert.deepEqual(await storage.listStoredChatMessages(NEW_ID), []);
  assert.deepEqual(requests, []);
});

test("other threads, and a thread once it stops being new, still ask the backend", async () => {
  const { storage, requests } = loadStorage();
  let newThreadId: string | undefined = NEW_ID;
  storage.registerNewThreadIdSource(() => newThreadId);
  await storage.getStoredChatThreadReadResult("saved");
  newThreadId = undefined;
  await storage.getStoredChatThreadReadResult(NEW_ID);
  assert.deepEqual(requests, ["GET thread saved", `GET thread ${NEW_ID}`]);
});

test("a row written while the runtime still calls the thread new is read back", async () => {
  const { storage, requests } = loadStorage();
  storage.registerNewThreadIdSource(() => NEW_ID);
  // Code-exec settings can create the row before first send.
  await storage.saveStoredChatThread(thread(NEW_ID));
  const { thread: read } = await storage.getStoredChatThreadReadResult(NEW_ID);
  assert.equal(read?.id, NEW_ID);
  assert.deepEqual(await storage.listStoredChatMessages(NEW_ID), []);
  assert.deepEqual(requests, [
    `GET thread ${NEW_ID}`,
    `GET thread ${NEW_ID}`,
    `GET messages ${NEW_ID}`,
  ]);
});

test("each registered runtime answers for its own id until it is unregistered", async () => {
  const { storage, requests } = loadStorage();
  const unregister = storage.registerNewThreadIdSource(() => NEW_ID);
  storage.registerNewThreadIdSource(() => "__LOCALID_other_pane");
  await storage.getStoredChatThreadReadResult(NEW_ID);
  await storage.getStoredChatThreadReadResult("__LOCALID_other_pane");
  assert.deepEqual(requests, []);
  unregister();
  await storage.getStoredChatThreadReadResult(NEW_ID);
  await storage.getStoredChatThreadReadResult("__LOCALID_other_pane");
  assert.deepEqual(requests, [`GET thread ${NEW_ID}`]);
});
