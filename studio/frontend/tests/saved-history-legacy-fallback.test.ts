// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { MessageRecord } from "../src/features/chat/types";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Storage = {
  readStoredChatMessages: (
    threadId: string,
  ) => Promise<{ messages: MessageRecord[]; fromBackend: boolean }>;
};

class StubWriteCoordinator {
  closeAdmission(): () => void {
    return () => {};
  }
  confirmFinalState(): void {}
  idsRequiringFence(): string[] {
    return [];
  }
}

const legacyRow: MessageRecord = {
  id: "assistant-1",
  threadId: "thread-1",
  parentId: "user-1",
  role: "assistant",
  content: [{ type: "text", text: "Obsolete browser copy" }],
  createdAt: 1,
};

function loadStorage(backend: {
  getChatThread: () => Promise<unknown>;
  listChatMessages: () => Promise<MessageRecord[]>;
}): Storage {
  return loadWithStubs<Storage>(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        ChatThreadDeletedError: class extends Error {},
        getChatThread: backend.getChatThread,
        listChatMessages: backend.listChatMessages,
        notifyChatHistoryUpdated: () => {},
      },
      "../db": {
        DEXIE_DB_NAME: "unsloth-chat",
        db: {
          threads: {},
          messages: {
            where: () => ({ equals: () => ({ toArray: async () => [legacyRow] }) }),
          },
          transaction: async () => undefined,
        },
      },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "./thread-record-write-coordinator": {
        ThreadRecordWriteCoordinator: StubWriteCoordinator,
      },
      "../stores/fork-boundary-store": { setForkBoundary: () => {} },
    },
  );
}

test("a failed backend read served from the legacy copy is not a backend read", async () => {
  const storage = loadStorage({
    getChatThread: async () => {
      throw new Error("offline");
    },
    listChatMessages: async () => {
      throw new Error("offline");
    },
  });
  const { messages, fromBackend } = await storage.readStoredChatMessages("thread-1");
  assert.deepEqual(messages.map((m) => m.id), ["assistant-1"]);
  assert.equal(fromBackend, false);
});

test("a successful backend read is a backend read", async () => {
  const storage = loadStorage({
    getChatThread: async () => ({ id: "thread-1" }),
    listChatMessages: async () => [
      { ...legacyRow, content: [{ type: "text", text: "Server copy" }] },
    ],
  });
  const { messages, fromBackend } = await storage.readStoredChatMessages("thread-1");
  assert.equal(fromBackend, true);
  assert.equal(
    (messages[0].content as unknown as { text: string }[])[0].text,
    "Server copy",
  );
});
