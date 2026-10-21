// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { MessageRepository } from "@assistant-ui/core/internal";
import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import type { MessageRecord } from "../src/features/chat/types.ts";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import * as savedUsage from "../src/features/chat/utils/saved-context-usage.ts";
import { orderBySelectedBranch } from "../src/features/chat/utils/message-order.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type DeleteModule = typeof import("../src/features/chat/utils/delete-thread-message.ts");

const legacyUsage = {
  modelId: "local-model", promptTokens: 28_902, completionTokens: 0,
  totalTokens: 28_902, cachedTokens: 0,
};

function message(id: string, role: "user" | "assistant", count?: typeof legacyUsage): ThreadMessage {
  return {
    id, role, createdAt: new Date(0), content: [{ type: "text", text: `Text of ${id}` }],
    metadata: { custom: count ? { contextUsage: count } : {} },
    ...(role === "assistant" ? { status: { type: "complete", reason: "stop" } } : { attachments: [] }),
  } as ThreadMessage;
}

function conversation(): ExportedMessageRepository {
  return {
    headId: "a2",
    messages: [
      { parentId: null, message: message("u1", "user") },
      { parentId: "u1", message: message("a1", "assistant") },
      { parentId: "a1", message: message("u2", "user") },
      { parentId: "u2", message: message("a2", "assistant", legacyUsage) },
    ],
  };
}

function harness(exported: ExportedMessageRepository, failure?: Error) {
  let imported: ExportedMessageRepository | undefined;
  let persisted: MessageRecord[] | undefined;
  let reads = 0;
  const module = loadWithStubs<DeleteModule>(
    new URL("../src/features/chat/utils/delete-thread-message.ts", import.meta.url),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": { listChatMessages: async () => { reads++; return []; } },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => { reads++; },
        syncStoredChatMessages: async (_id: string, records: MessageRecord[]) => {
          if (failure) throw failure;
          persisted = JSON.parse(JSON.stringify(records));
          return persisted;
        },
      },
      "./research-message-sync": researchSync,
      "./saved-context-usage": savedUsage,
    },
  );
  const records = (repo: ExportedMessageRepository) => repo.messages.map(({ message, parentId }) =>
    module.exportedItemToRecord("thread", parentId, message),
  );
  return {
    remove: (messageId: string, remoteId: string | undefined = "thread") => module.deleteThreadMessage({
      thread: { export: () => exported, import: (next) => { imported = next; } },
      messageId, remoteId,
    }),
    imported: () => imported,
    live: () => records(imported ?? exported),
    persisted: () => persisted,
    reads: () => reads,
  };
}

function priced(records: MessageRecord[], head = "a2", model = "local-model") {
  return savedUsage.branchContextUsage(
    orderBySelectedBranch(records, head), model, model.startsWith("external::") ? null : 65_536,
    model.startsWith("external::") ? null : 65_536,
  );
}

for (const deleted of ["a1", "u1"]) {
  test(`deleting ${deleted} invalidates a later legacy count in memory and after reload`, async () => {
    const original = conversation();
    const before = structuredClone(original);
    const h = harness(original);
    assert.equal(priced(h.live()), legacyUsage);
    await h.remove(deleted);
    assert.deepEqual(original, before, "deletion mutated the original repository");
    assert.ok(h.live().some((m) => m.id === "a2"), "the later answer must survive");
    for (const records of [h.live(), h.persisted()!]) {
      const count = priced(records);
      assert.ok(!count || count.estimated, "the deleted history's count was restored as exact");
    }
    assert.equal(h.reads(), 0, "ordinary deletion added a storage read");
  });
}

test("deleting an older answer invalidates cloud legacy usage without requiring a local tokenizer", async () => {
  const original = conversation();
  const cloud = "external::provider::model";
  original.messages[3].message.metadata.custom.contextUsage = { ...legacyUsage, modelId: cloud };
  const h = harness(original);
  await h.remove("a1");
  const records = h.persisted()!;
  const count = priced(records, "a2", cloud);
  assert.ok(!count || count.estimated);
  assert.equal(savedUsage.estimateBranch(orderBySelectedBranch(records, "a2"))?.estimated, true);
});

test("deleting a hidden sibling preserves the untouched branch's exact legacy usage", async () => {
  const original = conversation();
  original.messages.push({ parentId: "u1", message: message("alternative", "assistant") });
  const h = harness(original);
  await h.remove("alternative");
  assert.equal(priced(h.live()), legacyUsage);
  assert.deepEqual(priced(h.persisted()!), legacyUsage);
  assert.equal(h.live().some((m) => m.metadata?.revision), false);
});

test("deleting a leaf leaves earlier completed usage valid", async () => {
  const original = conversation();
  original.messages[1].message.metadata.custom.contextUsage = legacyUsage;
  const h = harness(original);
  await h.remove("u2");
  assert.equal(priced(h.live(), "a1"), legacyUsage);
  assert.deepEqual(priced(h.persisted()!, "a1"), legacyUsage);
});

test("deleting a prompt with sibling replies invalidates every surviving descendant branch", async () => {
  const original = conversation();
  original.messages.push(
    { parentId: "u1", message: message("a1b", "assistant") },
    { parentId: "a1b", message: message("u2b", "user") },
    { parentId: "u2b", message: message("a2b", "assistant", legacyUsage) },
  );
  const h = harness(original);
  await h.remove("u1");
  assert.deepEqual(h.live().map((m) => m.id).sort(), ["a2", "a2b", "u2", "u2b"]);
  for (const head of ["a2", "a2b"]) {
    const count = priced(h.persisted()!, head);
    assert.ok(!count || count.estimated, `${head} retained a pre-delete exact count`);
  }
});

test("a failed deletion changes neither live revisions nor saved usage", async () => {
  const original = conversation();
  const before = structuredClone(original);
  const failure = new Error("sync failed");
  const h = harness(original, failure);
  await assert.rejects(h.remove("a1"), failure);
  assert.equal(h.imported(), undefined);
  assert.equal(h.persisted(), undefined);
  assert.deepEqual(original, before);
  assert.equal(priced(h.live()), legacyUsage);
});

test("a missing deletion target does not change revisions or usage", async () => {
  const original = conversation();
  const before = structuredClone(original);
  const h = harness(original);
  await assert.rejects(h.remove("missing"));
  assert.equal(h.imported(), undefined);
  assert.equal(h.persisted(), undefined);
  assert.deepEqual(original, before);
});
