// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { restoredAssistantStatus } = await import("../src/features/chat/utils/continuation.ts");

type Saved = { id: string; metadata?: Record<string, unknown> };
type Module = {
  noteStoppedByUser: (id: string) => void;
  saveStoredChatMessage: (message: Saved & Record<string, unknown>) => Promise<unknown>;
  syncStoredChatMessages: (threadId: string, messages: unknown[]) => Promise<unknown>;
};

function harness() {
  const saved: Saved[] = [];
  const synced: Saved[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/utils/chat-history-storage.ts", import.meta.url),
    {
      "../api/chat-api": {
        ChatMessageProtectedError: class extends Error {},
        saveChatMessage: async (message: Saved) => {
          saved.push(message);
          return message;
        },
        notifyChatHistoryUpdated: () => {},
        syncChatMessages: async (_threadId: string, messages: Saved[]) => {
          synced.push(...messages);
          return messages;
        },
        getChatThread: async () => ({ id: "t1" }),
        saveChatThread: async () => {},
      },
      "../db": { DEXIE_DB_NAME: "test", db: {} },
      "./chat-thread-tombstones": { isChatThreadDeleted: () => false },
      "../stores/fork-boundary-store": { setForkBoundary: () => {} },
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
  return { module, saved, synced };
}

const reply = (id: string, metadata?: Record<string, unknown>) => ({
  id,
  threadId: "t1",
  role: "assistant",
  content: [],
  createdAt: 1,
  ...(metadata ? { metadata } : {}),
});

test("a Stopped reply saves as cancelled, so a reload shows it stopped, not an error", async () => {
  const { module, saved, synced } = harness();
  module.noteStoppedByUser("m1");
  const live = { incomplete: { reason: "interrupted" } };

  await module.saveStoredChatMessage(reply("m1", live));
  await module.syncStoredChatMessages("t1", [reply("m1", live)]);

  for (const message of [saved[0], synced[0]]) {
    assert.deepEqual(message.metadata?.incomplete, { reason: "cancelled" });
    assert.deepEqual(restoredAssistantStatus({ custom: message.metadata }), {
      type: "incomplete",
      reason: "cancelled",
    });
  }
});

test("a reply stopped before any chunk also saves as cancelled", async () => {
  const { module, saved } = harness();
  module.noteStoppedByUser("m2");
  await module.saveStoredChatMessage(reply("m2"));
  assert.deepEqual(saved[0].metadata?.incomplete, { reason: "cancelled" });
});

test("a closed tab keeps interrupted, and other reasons are left alone", async () => {
  const { module, saved } = harness();
  await module.saveStoredChatMessage(reply("m3", { incomplete: { reason: "interrupted" } }));
  module.noteStoppedByUser("m4");
  await module.saveStoredChatMessage(reply("m4", { incomplete: { reason: "length" } }));
  assert.deepEqual(saved[0].metadata?.incomplete, { reason: "interrupted" });
  assert.deepEqual(saved[1].metadata?.incomplete, { reason: "length" });
});

test("only an explicit Stop records it", () => {
  assert.match(
    readSrc("features/chat/api/chat-adapter.ts"),
    /incompleteReason = "cancelled";\s*noteStoppedByUser\(unstable_assistantMessageId\);/,
  );
});
