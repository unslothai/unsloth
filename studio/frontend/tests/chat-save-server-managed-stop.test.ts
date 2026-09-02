// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The autosave used to re-send on the next chunk after a 409, turning one 43s generation
// into 54 rejected PUTs. The stop is scoped to the refused payload, not the message id: a
// 409 is not proof of ownership, so only a resend of the same bytes is dropped.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  saveStoredChatMessage: (message: Record<string, unknown>) => Promise<unknown>;
  clearServerOwnedChatMessages: () => void;
  syncStoredChatMessages: (
    threadId: string,
    messages: unknown[],
    options?: { pruneMissing?: boolean; deletedMessageIds?: string[] },
  ) => Promise<unknown>;
};

class FakeProtectedError extends Error {}

function harness(options: { rejectIds?: Set<string>; failWith?: Error } = {}) {
  const attempts: string[] = [];
  const reject = options.rejectIds ?? new Set<string>();
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "../../../lib/account-transition.ts": {
        // This browser has only ever had one account in these tests, so the
        // legacy rows are its own.
        legacyBrowserDataBelongsToCurrentAccount: () => true,
      },
      "../api/chat-api": {
        ChatMessageProtectedError: FakeProtectedError,
        saveChatMessage: async (message: { id: string }) => {
          attempts.push(message.id);
          if (options.failWith) throw options.failWith;
          if (reject.has(message.id)) throw new FakeProtectedError("409");
          return message;
        },
        notifyChatHistoryUpdated: () => {},
        syncChatMessages: async (
          _threadId: string,
          messages: unknown[],
        ) => messages,
        getChatThread: async () => ({ id: "t1" }),
        saveChatThread: async () => {},
      },
      "../db": { DEXIE_DB_NAME: "test", db: {} },
      "./chat-thread-tombstones": { isChatThreadDeleted: () => false },
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
  return { module, attempts };
}

const message = (id: string, threadId = "t1") => ({
  id,
  threadId,
  role: "assistant",
  content: [],
  createdAt: 1,
});

test("a rejected message is never sent again", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  for (let chunk = 0; chunk < 5; chunk += 1) {
    await module.saveStoredChatMessage(message("m1"));
  }

  assert.deepEqual(
    attempts,
    ["m1"],
    "the server owns this message; retrying can never succeed",
  );
});

test("the rejection resolves rather than throwing, so the stream keeps following", async () => {
  const { module } = harness({ rejectIds: new Set(["m1"]) });
  const record = message("m1");
  assert.deepEqual(await module.saveStoredChatMessage(record), record);
});

test("only the rejected id is blocked", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage(message("m1"));
  await module.saveStoredChatMessage(message("m2"));
  await module.saveStoredChatMessage(message("m2"));

  assert.deepEqual(
    attempts,
    ["m1", "m2", "m2"],
    "a client-owned message must still autosave every chunk",
  );
});

test("the same id in another thread is unaffected", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage(message("m1", "t1"));
  await module.saveStoredChatMessage(message("m1", "t2"));

  assert.equal(attempts.length, 2, "the block is per thread, not per bare id");
});

test("clearing lets a reissued id save again", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage(message("m1"));
  module.clearServerOwnedChatMessages();
  await module.saveStoredChatMessage(message("m1"));

  assert.equal(attempts.length, 2);
});

test("a later monotonic update is still sent after a stale-seq rejection", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage({ ...message("m1"), metadata: { generationSeq: 4 } });
  await module.saveStoredChatMessage({ ...message("m1"), metadata: { generationSeq: 5 } });
  await module.saveStoredChatMessage({
    ...message("m1"),
    metadata: { generationSeq: 6, generationSettled: true },
  });

  assert.deepEqual(
    attempts,
    ["m1", "m1", "m1"],
    "each newer payload gets its own attempt; only a byte-identical resend is dropped",
  );
});

test("growing streamed content is never mistaken for a resend", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  for (const text of ["He", "Hell", "Hello"]) {
    await module.saveStoredChatMessage({
      ...message("m1"),
      content: [{ type: "text", text }],
    });
  }

  assert.equal(attempts.length, 3, "the follower publishes a longer body each chunk");
});

test("key order does not decide whether two payloads are the same", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage({
    id: "m1",
    threadId: "t1",
    role: "assistant",
    content: [],
    createdAt: 1,
  });
  await module.saveStoredChatMessage({
    createdAt: 1,
    content: [],
    role: "assistant",
    threadId: "t1",
    id: "m1",
  });

  assert.deepEqual(attempts, ["m1"], "same fields, same payload");
});

test("deleting a thread frees ids cached under every other thread", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1", "m2"]) });

  await module.saveStoredChatMessage(message("m1", "target"));
  await module.saveStoredChatMessage(message("m2", "other"));
  assert.deepEqual(attempts, ["m1", "m2"], "both are cached as refused");

  module.clearServerOwnedChatMessages();
  await module.saveStoredChatMessage(message("m1", "target"));
  await module.saveStoredChatMessage(message("m2", "other"));

  assert.deepEqual(
    attempts,
    ["m1", "m2", "m1", "m2"],
    "the delete can free an id cached under any thread, so every entry goes",
  );
});

test("pruning messages frees their ids too", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage(message("m1", "target"));
  await module.syncStoredChatMessages("other", [], {
    pruneMissing: true,
    deletedMessageIds: ["m1"],
  });
  await module.saveStoredChatMessage(message("m1", "target"));

  assert.deepEqual(attempts, ["m1", "m1"], "the id was freed, so retry it");
});

test("an ordinary sync does not clear the cache", async () => {
  const { module, attempts } = harness({ rejectIds: new Set(["m1"]) });

  await module.saveStoredChatMessage(message("m1", "target"));
  await module.syncStoredChatMessages("target", [], { pruneMissing: false });
  await module.saveStoredChatMessage(message("m1", "target"));

  assert.deepEqual(attempts, ["m1"], "nothing was deleted, so nothing was freed");
});

test("the cache is bounded, so a long session cannot grow without limit", async () => {
  const ids = Array.from({ length: 40 }, (_, index) => `m${index}`);
  const { module, attempts } = harness({ rejectIds: new Set(ids) });

  for (const id of ids) {
    await module.saveStoredChatMessage(message(id));
  }
  assert.equal(attempts.length, 40, "each is refused once");

  // Per id, not by re-walking: a bounded cache walked in insertion order misses every
  // lookup by construction, measuring the walk rather than the bound.
  await module.saveStoredChatMessage(message("m39"));
  assert.equal(attempts.length, 40, "the newest entry is still suppressed");

  await module.saveStoredChatMessage(message("m0"));
  assert.equal(attempts.length, 41, "the oldest fell out of the cap and is sent again");
});

test("an ordinary failure still propagates", async () => {
  const { module, attempts } = harness({ failWith: new Error("network down") });

  await assert.rejects(
    () => module.saveStoredChatMessage(message("m9")),
    /network down/,
  );
  await assert.rejects(() => module.saveStoredChatMessage(message("m9")));
  assert.deepEqual(attempts, ["m9", "m9"]);
});
