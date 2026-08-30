// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8977: deleting a message deep-cloned the whole thread with JSON.parse(JSON.stringify)
// on its way to a PUT that serializes it again, and ensured the thread row twice. The
// records are the same bytes on the wire either way, which is what these tests pin,
// alongside the delete semantics that must not move: last message, first message, a
// branch point, and a user prompt's cascaded replies.

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import * as dedupeSiblings from "../src/features/chat/utils/dedupe-identical-user-siblings.ts";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Module = {
  exportedItemToRecord: (
    threadId: string,
    parentId: string | null,
    message: unknown,
  ) => Record<string, unknown>;
  deleteThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
  }) => Promise<void>;
};

type Harness = {
  module: Module;
  calls: string[];
  synced: { records: Record<string, unknown>[]; pruneMissing?: boolean }[];
  stored: Record<string, unknown>[];
};

function harness(): Harness {
  const calls: string[] = [];
  const synced: Harness["synced"] = [];
  const stored: Record<string, unknown>[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/delete-thread-message.ts",
      import.meta.url,
    ),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": {
        listChatMessages: async () => {
          calls.push("listChatMessages");
          return stored;
        },
      },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => {
          calls.push("ensureStoredChatThread");
        },
        syncStoredChatMessages: async (
          _threadId: string,
          records: Record<string, unknown>[],
          options: { pruneMissing?: boolean },
        ) => {
          calls.push("syncStoredChatMessages");
          synced.push({ records, pruneMissing: options?.pruneMissing });
          return records;
        },
      },
      "./research-message-sync": researchSync,
      "./dedupe-identical-user-siblings": dedupeSiblings,
    },
  );
  return { module, calls, synced, stored };
}

function message(
  id: string,
  role: "user" | "assistant",
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    role,
    content: [{ type: "text", text: `text of ${id}` }],
    createdAt: new Date(1000),
    status: { type: "complete", reason: "stop" },
    ...extra,
  };
}

/** A linear thread `u0 -> a0 -> u1 -> a1 -> ...`. */
function linear(pairs: number): Exported {
  const messages: Exported["messages"] = [];
  let parentId: string | null = null;
  for (let i = 0; i < pairs; i++) {
    messages.push({ parentId, message: message(`u${i}`, "user") });
    messages.push({ parentId: `u${i}`, message: message(`a${i}`, "assistant") });
    parentId = `a${i}`;
  }
  return { headId: `a${pairs - 1}`, messages };
}

const idsOf = (exported: Exported): string[] =>
  exported.messages.map(({ message }) => String(message.id));

async function deleteFrom(
  h: Harness,
  exported: Exported,
  messageId: string,
): Promise<Exported> {
  let imported: Exported | null = null;
  await h.module.deleteThreadMessage({
    thread: { export: () => exported, import: (data) => (imported = data) },
    messageId,
    remoteId: "remote-1",
  });
  assert.notEqual(imported, null, "the thread was never re-imported");
  return imported as unknown as Exported;
}

test("records share the message's parts instead of deep-cloning the thread", async () => {
  const h = harness();
  const source = message("u0", "user", {
    attachments: [{ id: "att-1", type: "file", name: "a.txt" }],
    metadata: { custom: { modelId: "m" } },
  });
  const record = h.module.exportedItemToRecord("t", null, source);

  const parts = record.content as unknown[];
  const sourceParts = source.content as unknown[];
  assert.equal(parts[0], sourceParts[0], "the parts were copied, not shared");
  assert.notEqual(parts, sourceParts, "the list itself is still a snapshot");
  const attachments = record.attachments as unknown[];
  const sourceAttachments = source.attachments as unknown[];
  assert.equal(attachments[0], sourceAttachments[0]);
  assert.notEqual(attachments, sourceAttachments);

  // What matters on the wire: the same bytes the deep clone used to produce.
  const deepCloned = h.module.exportedItemToRecord(
    "t",
    null,
    JSON.parse(JSON.stringify(source)) as unknown,
  );
  assert.equal(
    JSON.stringify(record),
    JSON.stringify({ ...deepCloned, createdAt: record.createdAt }),
  );
});

test("a delete ensures the thread row once, not twice", async () => {
  const h = harness();
  await deleteFrom(h, linear(3), "u1");
  // ensureStoredChatThread belongs to syncStoredChatMessages, which does it for every
  // caller; doing it here as well cost a second GET /threads/{id} on every save.
  assert.deepEqual(h.calls, ["syncStoredChatMessages"]);
  assert.equal(h.synced[0].pruneMissing, true);
});

test("a research thread still reads its stored copy after the row is ensured", async () => {
  const h = harness();
  const exported = linear(2);
  exported.messages[3].message.metadata = { custom: { researchRunId: "r1" } };
  h.stored.push({
    id: "a1",
    threadId: "remote-1",
    parentId: "u1",
    role: "assistant",
    content: [{ type: "text", text: "stored report" }],
    metadata: { researchRunId: "r1" },
    createdAt: 1000,
  });

  await deleteFrom(h, exported, "u0");
  assert.deepEqual(h.calls, [
    "ensureStoredChatThread",
    "listChatMessages",
    "syncStoredChatMessages",
  ]);
  const report = h.synced[0].records.find((r) => r.id === "a1");
  assert.deepEqual(report?.content, [{ type: "text", text: "stored report" }]);
});

test("deleting the last message keeps the rest and moves the head", async () => {
  const h = harness();
  const next = await deleteFrom(h, linear(3), "a2");
  assert.deepEqual(idsOf(next), ["u0", "a0", "u1", "a1", "u2"]);
  assert.equal(next.headId, "u2");
  assert.deepEqual(
    h.synced[0].records.map((r) => r.id),
    idsOf(next),
  );
});

test("deleting the first message relinks its children to the root", async () => {
  const h = harness();
  const next = await deleteFrom(h, linear(2), "u0");
  // The prompt's own assistant reply cascades with it; the rest reparents to the root.
  assert.deepEqual(idsOf(next), ["u1", "a1"]);
  assert.equal(next.messages[0].parentId, null);
  assert.equal(next.headId, "a1");
});

test("deleting an assistant message relinks the turn that followed it", async () => {
  const h = harness();
  const next = await deleteFrom(h, linear(3), "a1");
  assert.deepEqual(idsOf(next), ["u0", "a0", "u1", "u2", "a2"]);
  assert.equal(
    next.messages.find(({ message }) => message.id === "u2")?.parentId,
    "u1",
  );
  assert.equal(next.headId, "a2");
});

/** `u0 -> a0 -> u1 -> {a1 -> u2, a1b}`: a regenerated turn, so u1 has two replies. */
function branched(): Exported {
  return {
    headId: "u2",
    messages: [
      { parentId: null, message: message("u0", "user") },
      { parentId: "u0", message: message("a0", "assistant") },
      { parentId: "a0", message: message("u1", "user") },
      { parentId: "u1", message: message("a1", "assistant") },
      { parentId: "u1", message: message("a1b", "assistant") },
      { parentId: "a1", message: message("u2", "user") },
    ],
  };
}

test("deleting one branch relinks its children and leaves the sibling", async () => {
  const h = harness();
  const next = await deleteFrom(h, branched(), "a1");
  assert.deepEqual(idsOf(next).sort(), ["a0", "a1b", "u0", "u1", "u2"]);
  assert.equal(
    next.messages.find(({ message }) => message.id === "u2")?.parentId,
    "u1",
    "the surviving turn hangs off the deleted reply's parent",
  );
});

test("deleting a branch point takes every reply branch with it", async () => {
  const h = harness();
  const next = await deleteFrom(h, branched(), "u1");
  // A prompt's assistant replies cascade, so both branches go and u2 reparents.
  assert.deepEqual(idsOf(next).sort(), ["a0", "u0", "u2"]);
  assert.equal(
    next.messages.find(({ message }) => message.id === "u2")?.parentId,
    "a0",
  );
  assert.deepEqual(
    h.synced[0].records.map((r) => r.id).sort(),
    ["a0", "u0", "u2"],
  );
});

test("an unsaved thread deletes locally without a sync", async () => {
  const h = harness();
  let imported: Exported | null = null;
  await h.module.deleteThreadMessage({
    thread: { export: () => linear(2), import: (data) => (imported = data) },
    messageId: "a0",
    remoteId: undefined,
  });
  assert.deepEqual(h.calls, []);
  assert.deepEqual(idsOf(imported as unknown as Exported), ["u0", "u1", "a1"]);
});
