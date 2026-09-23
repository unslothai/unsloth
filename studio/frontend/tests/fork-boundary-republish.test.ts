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
};

type Published = [string, string | null | undefined, string | null | undefined];

function harness(
  thread: Record<string, unknown> | undefined = {
    id: "fork-1",
    forkBoundaryMessageId: "m1",
    forkedFromThreadId: "src",
  },
  options: { deletedSources?: Set<string>; failReadsFrom?: number } = {},
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
        saveChatThread: async () => {},
      },
      "../db": { DEXIE_DB_NAME: "test", db: {} },
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

test("a failed thread read leaves the delete alone", async () => {
  // The row ensure reads first and must succeed; the boundary's read is the one that fails.
  const { module, published } = harness(undefined, { failReadsFrom: 2 });

  // The messages are already pruned; losing the divider until reopen beats failing the delete.
  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, []);
});
