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
  syncExportedRepositoryToBackend: (
    remoteId: string,
    exp: Exported,
    options?: { pruneMissing?: boolean; deletedMessageIds?: string[] },
  ) => Promise<void>;
};

function message(
  id: string,
  role: "user" | "assistant",
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    role,
    content: [{ type: "text", text: "same prompt" }],
    createdAt: new Date(1000),
    status: { type: "complete", reason: "stop" },
    ...extra,
  };
}

test("autosave does not infer deleted ids from identical user siblings", async () => {
  const synced: {
    records: Record<string, unknown>[];
    deletedMessageIds?: string[];
  }[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/delete-thread-message.ts",
      import.meta.url,
    ),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": {
        listChatMessages: async () => [],
      },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => {},
        syncStoredChatMessages: async (
          _threadId: string,
          records: Record<string, unknown>[],
          options: { deletedMessageIds?: string[] } = {},
        ) => {
          synced.push({
            records,
            deletedMessageIds: options.deletedMessageIds,
          });
          return records;
        },
      },
      "./research-message-sync": researchSync,
    },
  );

  const exported: Exported = {
    headId: "a-b",
    messages: [
      { parentId: null, message: message("prior", "assistant") },
      { parentId: "prior", message: message("user-a", "user") },
      { parentId: "prior", message: message("user-b", "user") },
      {
        parentId: "user-a",
        message: message("a-a", "assistant", {
          content: [{ type: "text", text: "first" }],
        }),
      },
      {
        parentId: "user-b",
        message: message("a-b", "assistant", {
          content: [{ type: "text", text: "second" }],
        }),
      },
    ],
  };

  await module.syncExportedRepositoryToBackend("thread-1", exported);

  assert.equal(synced.length, 1);
  assert.equal(synced[0].deletedMessageIds, undefined);
  const ids = synced[0].records.map((record) => record.id);
  assert.deepEqual(ids.sort(), ["a-a", "a-b", "prior", "user-a", "user-b"].sort());
});

test("a stale tab cannot delete a sibling another tab has made distinct", async () => {
  const synced: {
    records: Record<string, unknown>[];
    deletedMessageIds?: string[];
  }[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/delete-thread-message.ts",
      import.meta.url,
    ),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": {
        listChatMessages: async () => [],
      },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => {},
        syncStoredChatMessages: async (
          _threadId: string,
          records: Record<string, unknown>[],
          options: { deletedMessageIds?: string[] } = {},
        ) => {
          synced.push({
            records,
            deletedMessageIds: options.deletedMessageIds,
          });
          return records;
        },
      },
      "./research-message-sync": researchSync,
    },
  );

  // Tab A still sees user-b as a content clone. Tab B already edited b's text
  // server-side. Forwarding b as a deleted id would wipe that distinct branch.
  const staleTab: Exported = {
    headId: "user-b",
    messages: [
      { parentId: null, message: message("user-a", "user") },
      { parentId: null, message: message("user-b", "user") },
    ],
  };

  await module.syncExportedRepositoryToBackend("thread-1", staleTab, {
    deletedMessageIds: [],
  });

  assert.equal(synced[0].deletedMessageIds?.length ?? 0, 0);
  assert.deepEqual(
    synced[0].records.map((record) => record.id).sort(),
    ["user-a", "user-b"],
  );
});
