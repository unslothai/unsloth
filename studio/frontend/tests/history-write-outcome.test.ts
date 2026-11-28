// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { MessageRecord } from "../src/features/chat/types.ts";
import { historyWriteOutcome } from "../src/features/chat/utils/history-write-outcome.ts";

function record(): MessageRecord {
  return {
    id: "message-1",
    threadId: "thread-1",
    parentId: null,
    role: "user",
    content: [{ type: "text", text: "hello" }],
    metadata: { quote: { messageId: "quote", text: "context" } },
    createdAt: 1,
  };
}

test("a written row is persisted", () => {
  const input = record();
  const saved: MessageRecord = { ...input, createdAt: 2 };
  assert.equal(historyWriteOutcome(input, saved), "persisted");
});

test("an early return by reference is delivered but not persisted", () => {
  const input = record();
  assert.equal(historyWriteOutcome(input, input), "delivered");
});
