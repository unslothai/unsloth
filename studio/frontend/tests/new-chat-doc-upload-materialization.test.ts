// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type { ThreadScopeMaterialization } from "../src/features/rag/utils/materialize-thread-scope";

const FRESH = "__LOCALID_fresh0001";

class ChatThreadDeletedErrorStub extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ChatThreadDeletedError";
  }
}

function load() {
  const { materializeThreadScope } = loadWithStubs<{
    materializeThreadScope: (m: ThreadScopeMaterialization) => Promise<string>;
  }>(
    new URL(
      "../src/features/rag/utils/materialize-thread-scope.ts",
      import.meta.url,
    ),
    {
      "@/features/chat/api/chat-api": {
        ChatThreadDeletedError: ChatThreadDeletedErrorStub,
      },
      "@/features/chat/utils/thread-ids": {
        isAssistantLocalThreadId: (id: string | null | undefined) =>
          typeof id === "string" && id.startsWith("__LOCALID_"),
      },
    },
  );
  return materializeThreadScope;
}

function deferredWriteStore() {
  const store = { resolvable: false };
  const requireStoredThread = (threadId: string): Promise<void> =>
    store.resolvable
      ? Promise.resolve()
      : Promise.reject(new Error(`Thread ${threadId} was not persisted`));
  const initialize = () => {
    store.resolvable = true;
    return requireStoredThread(FRESH).then(() => FRESH);
  };
  return { requireStoredThread, initialize };
}

function neverInitialize(): () => Promise<string> {
  const initialize = () => {
    throw new Error("initialize must not run");
  };
  return initialize;
}

test("a brand-new chat whose __LOCALID_ id reads unpersisted materializes the row", async () => {
  const materialize = load();
  const { requireStoredThread, initialize } = deferredWriteStore();

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
    isThreadDeleted: () => false,
    requireStoredThread,
    initialize,
  });

  assert.equal(result, FRESH);
});

test("an id-less composer materializes the thread immediately", async () => {
  const materialize = load();
  let initialized = 0;

  const result = await materialize({
    threadId: null,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
    isThreadDeleted: () => false,
    requireStoredThread: () => Promise.resolve(),
    initialize: () => {
      initialized += 1;
      return Promise.resolve(FRESH);
    },
  });

  assert.equal(result, FRESH);
  assert.equal(initialized, 1);
});

test("a saved chat (remoteId set) that reads missing is an error, never re-initialized", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: FRESH }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a real (non-__LOCALID_) id that reads missing stays an error", async () => {
  const materialize = load();
  const missing = new Error("Thread chat_001 was not persisted");
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: "chat_001",
      readCurrentThreadItem: () => ({ id: "chat_001", remoteId: undefined }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a tombstoned thread stays an error instead of being resurrected", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
      isThreadDeleted: () => true,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a backend-tombstoned read stays an error instead of being resurrected", async () => {
  const materialize = load();
  const deleted = new ChatThreadDeletedErrorStub(`Thread ${FRESH} was deleted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw deleted;
      },
      initialize,
    }),
    (error: unknown) => error === deleted,
  );
});

test("a thread switch while the stored check waits is an error, not a recovery", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({
        id: "__LOCALID_other",
        remoteId: undefined,
      }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});
