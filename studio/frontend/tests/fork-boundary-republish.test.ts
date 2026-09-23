// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Deleting the message a fork's "Continued from chat" divider sits under moves the boundary, in
// the same transaction that prunes the row. The delete path imports the returned messages and
// never reads the thread again, so the store kept naming the deleted id and the divider stayed
// gone for the rest of the view. Only a prune re-reads: the streaming autosave runs constantly.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type SyncOptions = { pruneMissing?: boolean; deletedMessageIds?: string[] };
type Module = {
  syncStoredChatMessages: (
    threadId: string,
    messages: unknown[],
    options?: SyncOptions,
  ) => Promise<unknown>;
  chatThreadExistsOnBackend: (
    threadId: string,
  ) => Promise<boolean | undefined>;
};

type Published = [string, string | null | undefined, string | null | undefined];

function harness(
  // null is the backend saying it has no such thread, which is what a 404 becomes.
  thread: Record<string, unknown> | null | undefined = {
    id: "fork-1",
    forkBoundaryMessageId: "m1",
    forkedFromThreadId: "src",
  },
  options: {
    deletedSources?: Set<string>;
    failReadsFrom?: number;
    legacyThreads?: Record<string, Record<string, unknown>>;
  } = {},
) {
  const published: Published[] = [];
  const threadReads: string[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        ChatMessageProtectedError: class extends Error {},
        saveChatMessage: async (message: unknown) => message,
        notifyChatHistoryUpdated: () => {},
        syncChatMessages: async (_threadId: string, messages: unknown[]) =>
          messages,
        getChatThread: async (id: string) => {
          threadReads.push(id);
          // The row ensure reads first, so a count is how this picks out the boundary's own read.
          if (threadReads.length >= (options.failReadsFrom ?? Infinity)) {
            throw new Error("offline");
          }
          return thread;
        },
        // Echoes the record back, like the real endpoint: returning nothing makes the legacy
        // re-import fail, which would hide the very fallback these tests are about.
        saveChatThread: async (t: unknown) => t,
      },
      "../db": {
        DEXIE_DB_NAME: "test",
        db: {
          threads: {
            get: async (id: string) => options.legacyThreads?.[id],
          },
          messages: {
            where: () => ({ equals: () => ({ toArray: async () => [] }) }),
          },
        },
      },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: (id: string) =>
          options.deletedSources?.has(id) ?? false,
      },
      "../stores/fork-boundary-store": {
        setForkBoundary: (
          threadId: string,
          messageId: string | null | undefined,
          sourceThreadId: string | null | undefined,
        ) => published.push([threadId, messageId, sourceThreadId]),
      },
      "./thread-record-write-coordinator": {
        ThreadRecordWriteCoordinator: class {
          async settleCurrent() {}
          async write(_id: string, fn: () => Promise<unknown>) {
            return fn();
          }
          observe() {}
          closeAdmission() {}
          confirmFinalState() {}
          hasPending() {
            return false;
          }
          idsRequiringFence() {
            return [];
          }
        },
      },
    },
  );
  return { module, published, threadReads };
}

test("deleting a message republishes the boundary the backend reseated", async () => {
  const { module, published } = harness({
    id: "fork-1",
    // Where the prune moved it: the surviving parent of the row that was deleted.
    forkBoundaryMessageId: "m1",
    forkedFromThreadId: "src",
  });

  await module.syncStoredChatMessages("fork-1", [], {
    pruneMissing: true,
    deletedMessageIds: ["m2"],
  });

  assert.deepEqual(published, [["fork-1", "m1", "src"]]);
});

test("a prune that cleared the boundary clears the divider too", async () => {
  // Every inherited message went, so there is no inherited history left to close.
  const { module, published } = harness({
    id: "fork-1",
    forkBoundaryMessageId: null,
    forkedFromThreadId: "src",
  });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", null, "src"]]);
});

test("the streaming autosave never pays for the extra read", async () => {
  const { module, published, threadReads } = harness();

  await module.syncStoredChatMessages("fork-1", []);
  const before = threadReads.length;
  await module.syncStoredChatMessages("fork-1", [], {
    pruneMissing: false,
    deletedMessageIds: [],
  });
  const plainSyncReads = threadReads.length - before;
  assert.deepEqual(published, []);

  const beforePrune = threadReads.length;
  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  // Exactly one read more than the same sync without a prune, and it is the boundary's.
  assert.equal(threadReads.length - beforePrune, plainSyncReads + 1);
  assert.deepEqual(published, [["fork-1", "m1", "src"]]);
});

test("a source deleted in this tab keeps the words and drops the link", async () => {
  const { module, published } = harness(
    { id: "fork-1", forkBoundaryMessageId: "m1", forkedFromThreadId: "src" },
    { deletedSources: new Set(["src"]) },
  );

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", "m1", null]]);
});

// --- the backlink's existence check ------------------------------------------

test("the backend's answer is what decides, not this browser's legacy row", async () => {
  // getStoredChatThread re-imports the Dexie row when the backend has none, which is the very
  // case a source deleted on another device produces. A 404 has to stay a "no" through it.
  const { module } = harness(null, {
    legacyThreads: {
      src: { id: "src", title: "Deleted elsewhere", modelType: "base", createdAt: 1 },
    },
  });

  assert.equal(await module.chatThreadExistsOnBackend("src"), false);
});

test("a source the backend still holds is openable", async () => {
  const { module } = harness({ id: "src" });

  assert.equal(await module.chatThreadExistsOnBackend("src"), true);
});

test("a backend that could not answer is not a deletion", async () => {
  const { module } = harness(undefined, { failReadsFrom: 1 });

  // Undefined, not false: the divider navigates rather than claiming the chat is gone.
  assert.equal(await module.chatThreadExistsOnBackend("src"), undefined);
});

test("a source this tab deleted needs no round trip", async () => {
  const { module, threadReads } = harness(
    { id: "src" },
    { deletedSources: new Set(["src"]) },
  );

  assert.equal(await module.chatThreadExistsOnBackend("src"), false);
  assert.deepEqual(threadReads, []);
});

test("a failed thread read leaves the delete alone", async () => {
  // The row ensure reads first and must succeed; the boundary's read is the one that fails.
  const { module, published } = harness(undefined, { failReadsFrom: 2 });

  // The messages are already pruned; losing the divider until reopen beats failing the delete.
  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, []);
});
