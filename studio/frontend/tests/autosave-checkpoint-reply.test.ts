// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import * as savedUsage from "../src/features/chat/utils/saved-context-usage.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Sync = (
  remoteId: string,
  exp: Exported,
  options?: { replyOnly?: boolean },
) => Promise<void>;

function harness() {
  const saved: Record<string, unknown>[] = [];
  const synced: Record<string, unknown>[][] = [];
  const module = loadWithStubs<{ syncExportedRepositoryToBackend: Sync }>(
    new URL("../src/features/chat/utils/delete-thread-message.ts", import.meta.url),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": { listChatMessages: async () => [] },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => {},
        saveStoredChatMessage: async (record: Record<string, unknown>) => {
          saved.push(record);
          return record;
        },
        syncStoredChatMessages: async (_id: string, records: Record<string, unknown>[]) => {
          synced.push(records);
          return records;
        },
      },
      "./research-message-sync": researchSync,
      "./saved-context-usage": savedUsage,
    },
  );
  return { sync: module.syncExportedRepositoryToBackend, saved, synced };
}

const image = { type: "image", image: `data:image/png;base64,${"A".repeat(400_000)}` };

function chat(headRole: "user" | "assistant"): Exported {
  const messages: Exported["messages"] = [
    {
      parentId: null,
      message: {
        id: "u0",
        role: "user",
        content: [image, { type: "text", text: "what is this" }],
        createdAt: new Date(1000),
      },
    },
    {
      parentId: "u0",
      message: {
        id: "a0",
        role: "assistant",
        content: [{ type: "text", text: "a cat" }],
        createdAt: new Date(2000),
      },
    },
    {
      parentId: "a0",
      message: {
        id: "u1",
        role: "user",
        content: [{ type: "text", text: "and now?" }],
        createdAt: new Date(3000),
      },
    },
  ];
  if (headRole === "assistant") {
    messages.push({
      parentId: "u1",
      message: {
        id: "a1",
        role: "assistant",
        content: [{ type: "text", text: "still stream" }],
        createdAt: new Date(4000),
      },
    });
  }
  return { headId: headRole === "assistant" ? "a1" : "u1", messages };
}

test("a mid-run checkpoint saves only the reply being written, not the earlier image", async () => {
  const h = harness();
  await h.sync("remote-1", chat("assistant"), { replyOnly: true });
  assert.deepEqual(h.synced, []);
  assert.equal(h.saved.length, 1);
  assert.equal(h.saved[0].id, "a1");
  assert.equal(h.saved[0].parentId, "u1");
  assert.equal(h.saved[0].threadId, "remote-1");
  assert.deepEqual(h.saved[0].content, [{ type: "text", text: "still stream" }]);
  assert.ok(JSON.stringify(h.saved).length < 1_000);
});

test("a checkpoint with no reply yet still saves the whole thread", async () => {
  const h = harness();
  await h.sync("remote-1", chat("user"), { replyOnly: true });
  assert.deepEqual(h.saved, []);
  assert.deepEqual(h.synced[0].map((r) => r.id), ["u0", "a0", "u1"]);
});

test("a run's start and end saves still send the whole thread", async () => {
  const h = harness();
  await h.sync("remote-1", chat("assistant"));
  assert.deepEqual(h.saved, []);
  assert.deepEqual(h.synced[0].map((r) => r.id), ["u0", "a0", "u1", "a1"]);
});

test("a failed whole-chat save keeps later checkpoints whole until one lands", () => {
  const provider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /const partial = replyOnly && !fullSaveDueRef\.current\.has\(threadId\);\s*if \(!partial\) fullSaveDueRef\.current\.add\(threadId\);\s*await syncExportedRepositoryToBackend\(remoteId, exported, \{\s*replyOnly: partial,\s*\}\);\s*if \(!partial\) fullSaveDueRef\.current\.delete\(threadId\);/,
  );
});
