// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  readBackendChatThread: (
    threadId: string,
  ) => Promise<Record<string, unknown> | null | undefined>;
};

type Published = [string, string[], string | null | undefined];

function harness(
  thread: Record<string, unknown> | null | undefined = {
    id: "fork-1",
    forkBoundaryMessageId: "m1",
    forkedFromThreadId: "src",
  },
  options: {
    deletedSources?: Set<string>;
    failReadsFrom?: number;
    legacyThreads?: Record<string, Record<string, unknown>>;
    syncedMessages?: { id: string; parentId?: string | null }[];
    threadReadQueue?: (Record<string, unknown> | null)[];
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
          options.syncedMessages ?? messages,
        getChatThread: async (id: string) => {
          threadReads.push(id);
          if (threadReads.length >= (options.failReadsFrom ?? Infinity)) {
            throw new Error("offline");
          }
          const queued = options.threadReadQueue?.shift();
          return queued === undefined ? thread : queued;
        },
        // Echoes the record like the real endpoint; returning nothing breaks the legacy re-import.
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
          messageIds: Iterable<string> | null | undefined,
          sourceThreadId: string | null | undefined,
        ) =>
          published.push([threadId, [...(messageIds ?? [])], sourceThreadId]),
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
  const { module, published } = harness(
    {
      id: "fork-1",
      forkBoundaryMessageId: "m1",
      forkedFromThreadId: "src",
    },
    {
      syncedMessages: [
        { id: "m0", parentId: null },
        { id: "m1", parentId: "m0" },
      ],
    },
  );

  await module.syncStoredChatMessages("fork-1", [], {
    pruneMissing: true,
    deletedMessageIds: ["m2"],
  });

  assert.deepEqual(published, [["fork-1", ["m1", "m0"], "src"]]);
});

test("a prune that cleared the boundary clears the divider too", async () => {
  const { module, published } = harness({
    id: "fork-1",
    forkBoundaryMessageId: null,
    forkedFromThreadId: "src",
  });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", [], "src"]]);
});

test("the streaming autosave never pays for the extra read", async () => {
  const { module, published, threadReads } = harness(undefined, {
      syncedMessages: [
        { id: "m0", parentId: null },
        { id: "m1", parentId: "m0" },
      ],
    });

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

  assert.equal(threadReads.length - beforePrune, plainSyncReads + 1);
  assert.deepEqual(published, [["fork-1", ["m1", "m0"], "src"]]);
});

test("a source deleted in this tab keeps the words and drops the link", async () => {
  const { module, published } = harness(
    { id: "fork-1", forkBoundaryMessageId: "m1", forkedFromThreadId: "src" },
    {
      deletedSources: new Set(["src"]),
      syncedMessages: [
        { id: "m0", parentId: null },
        { id: "m1", parentId: "m0" },
      ],
    },
  );

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", ["m1", "m0"], null]]);
});

test("the backend's answer is what decides, not this browser's legacy row", async () => {
  const { module } = harness(null, {
    legacyThreads: {
      src: { id: "src", title: "Deleted elsewhere", modelType: "base", createdAt: 1 },
    },
  });

  assert.equal(await module.readBackendChatThread("src"), null);
});

test("a source the backend still holds comes back as its record", async () => {
  const { module } = harness({ id: "src", pairId: "pair-1" });

  assert.deepEqual(await module.readBackendChatThread("src"), {
    id: "src",
    pairId: "pair-1",
  });
});

test("a backend that could not answer is not a deletion", async () => {
  const { module } = harness(undefined, { failReadsFrom: 1 });

  assert.equal(await module.readBackendChatThread("src"), undefined);
});

test("a source this tab deleted needs no round trip", async () => {
  const { module, threadReads } = harness(
    { id: "src" },
    { deletedSources: new Set(["src"]) },
  );

  assert.equal(await module.readBackendChatThread("src"), null);
  assert.deepEqual(threadReads, []);
});

test("a failed thread read leaves the delete alone", async () => {
  const { module, published } = harness(undefined, { failReadsFrom: 2 });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, []);
});

test("an anchor the message list cannot place is re-read, not treated as gone", async () => {
  const { module, published } = harness(undefined, {
    syncedMessages: [
      { id: "m0", parentId: null },
      { id: "m1", parentId: "m0" },
    ],
    threadReadQueue: [
      { id: "fork-1", forkBoundaryMessageId: "m2", forkedFromThreadId: "src" },
      { id: "fork-1", forkBoundaryMessageId: "m2", forkedFromThreadId: "src" },
      { id: "fork-1", forkBoundaryMessageId: "m1", forkedFromThreadId: "src" },
    ],
  });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", ["m1", "m0"], "src"]]);
});

test("an anchor still unplaceable after the re-read leaves the divider alone", async () => {
  const { module, published } = harness(undefined, {
    syncedMessages: [{ id: "m0", parentId: null }],
    threadReadQueue: [
      { id: "fork-1", forkBoundaryMessageId: "m9", forkedFromThreadId: "src" },
      { id: "fork-1", forkBoundaryMessageId: "m9", forkedFromThreadId: "src" },
      { id: "fork-1", forkBoundaryMessageId: "m9", forkedFromThreadId: "src" },
    ],
  });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, []);
});

test("a boundary the backend really did clear still clears", async () => {
  const { module, published } = harness({
    id: "fork-1",
    forkBoundaryMessageId: null,
    forkedFromThreadId: "src",
  });

  await module.syncStoredChatMessages("fork-1", [], { pruneMissing: true });

  assert.deepEqual(published, [["fork-1", [], "src"]]);
});
