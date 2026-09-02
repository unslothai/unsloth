// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Module = {
  extractTaggedText: (content: unknown) => string;
  updateThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
    newText: string;
    isIncognito: boolean;
  }) => Promise<unknown>;
};

const RECORD_MODULE = loadWithStubs<{
  exportedItemToRecord: (
    threadId: string,
    parentId: string | null,
    message: unknown,
  ) => Record<string, unknown>;
}>(
  new URL(
    "../src/features/chat/utils/delete-thread-message.ts",
    import.meta.url,
  ),
  {
    "@assistant-ui/core/internal": { MessageRepository },
    "../api/chat-api": { listChatMessages: async () => [] },
    "./chat-history-storage": {
      ensureStoredChatThread: async () => {},
      syncStoredChatMessages: async (
        _threadId: string,
        records: Record<string, unknown>[],
      ) => records,
    },
    "./research-message-sync": researchSync,
  },
);

const CUSTOM = {
  incomplete: { reason: "length" },
  contextUsage: { promptTokens: 900, contextLength: 4096 },
  timing: { tokensPerSecond: 42.5, durationMs: 1200 },
  contextTruncation: { dropped: 4, fits: true },
};

function harness() {
  const saved: Record<string, unknown>[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/update-thread-message.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        saveChatMessage: async (record: Record<string, unknown>) => {
          saved.push(record);
          return record;
        },
      },
      "./delete-thread-message": RECORD_MODULE,
      "./research-message-sync": researchSync,
    },
  );
  return { module, saved };
}

function thread(content: unknown[]): Exported {
  return {
    headId: "a0",
    messages: [
      {
        parentId: null,
        message: {
          id: "u0",
          role: "user",
          content: [{ type: "text", text: "hello" }],
          createdAt: new Date(1000),
        },
      },
      {
        parentId: "u0",
        message: {
          id: "a0",
          role: "assistant",
          content,
          createdAt: new Date(2000),
          metadata: { custom: CUSTOM },
        },
      },
    ],
  };
}

async function save(
  newText: string,
  content: unknown[] = [{ type: "text", text: "the original reply" }],
) {
  const h = harness();
  const exported = thread(content);
  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText,
    isIncognito: false,
  });
  assert.equal(h.saved.length, 1, "the edit was never written");
  return h.saved[0];
}

test("an edited reply keeps the metadata the turn was stored with", async () => {
  const record = await save("an edited reply");

  assert.deepEqual(record.metadata, CUSTOM);
});

test("the edit still rewrites the content and its identity", async () => {
  const record = await save("an edited reply");

  assert.equal(record.id, "a0");
  assert.equal(record.threadId, "remote-1");
  assert.equal(record.parentId, "u0");
  assert.equal(record.role, "assistant");
  assert.equal(record.createdAt, 2000);
  assert.deepEqual(record.content, [{ type: "text", text: "an edited reply" }]);
});

test("a reply with no metadata of its own writes none", async () => {
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);
  const target = exported.messages[1].message;
  target.metadata = { custom: {} };

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });

  assert.equal("metadata" in h.saved[0], false);
});

test("an incognito edit is never written at all", async () => {
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: true,
  });

  assert.equal(h.saved.length, 0);
});

test("the turn keeps its timestamp when the export carries epoch millis", async () => {
  // Re-dating the turn to now would reorder the thread.
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);
  (exported.messages[1].message as Record<string, unknown>).createdAt = 2000;

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });

  assert.equal(h.saved[0].createdAt, 2000);
});

// What a generated reply carries on `metadata.custom` after a reload. Sent back with an edit,
// the backend refuses to detach the run and answers 409.
const GENERATION_OWNERSHIP = {
  serverManaged: true,
  generationRunId: "run-1",
  generationSeq: 3,
  generationStatus: "completed",
  generationSettled: true,
};

async function saveOwnedReply(custom: Record<string, unknown>) {
  const h = harness();
  const exported = thread([{ type: "text", text: "the original reply" }]);
  exported.messages[1].message.metadata = { custom };

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "an edited reply",
    isIncognito: false,
  });

  assert.equal(h.saved.length, 1, "the edit was never written");
  return h.saved[0];
}

test("editing a generated reply drops the run's claim on the turn", async () => {
  const record = await saveOwnedReply({ ...CUSTOM, ...GENERATION_OWNERSHIP });

  assert.deepEqual(record.metadata, CUSTOM);
  assert.deepEqual(
    Object.keys(GENERATION_OWNERSHIP).filter(
      (key) => key in (record.metadata as Record<string, unknown>),
    ),
    [],
  );
  assert.deepEqual(record.content, [{ type: "text", text: "an edited reply" }]);
});

test("the strip covers the backend's whole server-managed key set", async () => {
  // Parity only: research reports have no pencil and the backend still refuses to edit them.
  const record = await saveOwnedReply({
    ...CUSTOM,
    serverManaged: true,
    researchRunId: "research-1",
    researchStatus: "completed",
    researchPlanRevision: 2,
    researchRun: { id: "research-1" },
  });

  assert.deepEqual(record.metadata, CUSTOM);
});

test("a reply whose only metadata was the run's claim writes none", async () => {
  const record = await saveOwnedReply({ ...GENERATION_OWNERSHIP });

  assert.equal("metadata" in record, false);
});
