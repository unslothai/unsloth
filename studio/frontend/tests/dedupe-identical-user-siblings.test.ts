// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { dedupeIdenticalUserSiblings } from "../src/features/chat/utils/dedupe-identical-user-siblings.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";

const threadId = "thread-1";
const parentId = "prior-assistant";

function userMessage(
  id: string,
  createdAt: number,
  attachments?: MessageRecord["attachments"],
): MessageRecord {
  return {
    id,
    threadId,
    parentId,
    role: "user",
    content: [{ type: "text", text: "Please review this document." }],
    ...(attachments ? { attachments } : {}),
    createdAt,
  };
}

function assistantMessage(
  id: string,
  parent: string,
  createdAt: number,
): MessageRecord {
  return {
    id,
    threadId,
    parentId: parent,
    role: "assistant",
    content: [{ type: "text", text: "Summary." }],
    createdAt,
  };
}

test("dedupe collapses identical user siblings and remaps assistant parents", () => {
  const attachment = [
    {
      id: "att-1",
      type: "document" as const,
      name: "doc.md",
      contentType: "text/markdown",
      content: [{ type: "text" as const, text: "[Markdown: doc.md]\nhello" }],
      status: { type: "complete" as const },
    },
  ];
  const duplicateAttachment = [
    {
      ...attachment[0],
      id: "att-2",
    },
  ];

  const canonical = userMessage("user-a", 10, attachment);
  const duplicate = userMessage("user-b", 20, duplicateAttachment);
  const firstReply = assistantMessage("assistant-1", "user-b", 30);
  const secondReply = assistantMessage("assistant-2", "user-b", 40);

  const { records, collapsedIds } = dedupeIdenticalUserSiblings([
    canonical,
    duplicate,
    firstReply,
    secondReply,
  ]);

  assert.deepEqual(collapsedIds, ["user-b"]);
  assert.deepEqual(
    records.map((record) => record.id),
    ["user-a", "assistant-1", "assistant-2"],
  );
  assert.equal(records[1].parentId, "user-a");
  assert.equal(records[2].parentId, "user-a");
});

test("dedupe keeps intentional user branches with different content", () => {
  const first = userMessage("user-a", 10);
  const second = {
    ...userMessage("user-b", 20),
    content: [{ type: "text" as const, text: "Different prompt." }],
  };

  const { records, collapsedIds } = dedupeIdenticalUserSiblings([
    first,
    second,
  ]);

  assert.deepEqual(collapsedIds, []);
  assert.equal(records.length, 2);
});
