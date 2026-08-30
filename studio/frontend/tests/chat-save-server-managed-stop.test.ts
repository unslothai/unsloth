// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The per-chunk autosave used to treat the server's 409 as an anonymous error and re-send
// on the very next chunk. One 43s generation in a live session produced 54 rejected PUTs,
// each logging a full traceback server-side. A 409 here is permanent -- the server owns the
// message -- so the only correct reaction is to stop, not to back off.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  saveStoredChatMessage: (message: Record<string, unknown>) => Promise<unknown>;
  clearServerOwnedChatMessages: (threadId: string) => void;
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
  module.clearServerOwnedChatMessages("t1");
  await module.saveStoredChatMessage(message("m1"));

  assert.equal(attempts.length, 2);
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
