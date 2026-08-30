// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The per-chunk autosave used to treat the server's 409 as an anonymous error and re-send
// on the very next chunk. One 43s generation in a live session produced 54 rejected PUTs,
// each logging a full traceback server-side.
//
// The stop is scoped to the exact payload that was refused, not to the message id. A 409 on
// this route is not proof of permanent ownership: it also covers a message id colliding with
// another thread, and even a protected message may be refused only because its generationSeq
// lost a race, with the later monotonic and terminal writes still permitted. So a resend of
// the same bytes is dropped and anything the client has moved on to is sent.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  saveStoredChatMessage: (message: Record<string, unknown>) => Promise<unknown>;
  clearServerOwnedChatMessages: () => void;
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
      "../api/chat-api": {
        ChatMessageProtectedError: FakeProtectedError,
        saveChatMessage: async (message: { id: string }) => {
          attempts.push(message.id);
          if (options.failWith) throw options.failWith;
          if (reject.has(message.id)) throw new FakeProtectedError("409");
          return message;
        },
        notifyChatHistoryUpdated: () => {},
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
  // The caller's optimistic record comes back: the server's copy is already durable, and
  // throwing here would abort the recovery loop that is still reading the generation.
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
  // _safe_generation_assistant_update refuses an autosave whose generationSeq lost the race
  // to the producer, then permits the next one. The recovery follower keeps going and its
  // terminal write carries contextUsage, timings and response details, so muting the id here
  // would drop all of them on reload.
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
  // Two call sites build the record with different property order. Without a stable
  // serialization the storm would come straight back for whichever one loses the race.
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
  // A 409 also means "this id already belongs to another thread". That rejection is
  // recorded under the thread being written TO, so deleting the thread that actually owns
  // the id makes the same payload valid while the stale entry sits elsewhere in the map.
  // Clearing only the deleted thread's bucket would skip the now-valid write and the
  // message would be gone after a reload.
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

test("an ordinary failure still propagates", async () => {
  const { module, attempts } = harness({ failWith: new Error("network down") });

  // Only the permanent 409 is swallowed. A transient failure must still reach the caller,
  // which retries it -- turning that into a silent success would lose the message.
  await assert.rejects(
    () => module.saveStoredChatMessage(message("m9")),
    /network down/,
  );
  // And it must not be marked server-owned, or one blip would mute the message for good.
  await assert.rejects(() => module.saveStoredChatMessage(message("m9")));
  assert.deepEqual(attempts, ["m9", "m9"]);
});
