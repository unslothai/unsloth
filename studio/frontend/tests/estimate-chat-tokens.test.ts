// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";
import {
  estimateContextUsage,
  estimateMessagesTokenCount,
} from "../src/features/chat/utils/estimate-chat-tokens.ts";

function record(
  id: string,
  role: MessageRecord["role"],
  text: string,
  createdAt: number,
  parentId?: string,
): MessageRecord {
  return {
    id,
    threadId: "t",
    role,
    content: [{ type: "text", text }] as MessageRecord["content"],
    createdAt,
    ...(parentId ? { parentId } : {}),
  };
}

test("text length over four, at least one token", () => {
  assert.equal(estimateMessagesTokenCount([record("1", "user", "a".repeat(3600), 1)]), 900);
  assert.equal(estimateMessagesTokenCount([record("1", "user", "hi", 1)]), 1);
});

test("an empty or textless chat has no estimate", () => {
  assert.equal(estimateMessagesTokenCount([]), null);
  assert.equal(estimateMessagesTokenCount(null), null);
  assert.equal(estimateContextUsage([record("1", "user", "", 1)]), null);
});

test("only the selected branch is counted", () => {
  const messages = [
    record("1", "user", "a".repeat(40), 100),
    record("2", "assistant", "b".repeat(400), 200, "1"),
    record("3", "assistant", "c".repeat(80), 300, "1"),
    record("4", "user", "d".repeat(40), 400, "3"),
  ];
  assert.equal(estimateMessagesTokenCount(messages), 40);
});

test("the estimate is flagged so a count replaces it", () => {
  assert.deepEqual(estimateContextUsage([record("1", "user", "a".repeat(400), 1)]), {
    promptTokens: 100,
    completionTokens: 0,
    totalTokens: 100,
    cachedTokens: 0,
    estimated: true,
  });
});

test("an estimate without a window reads as approximate tokens", () => {
  const state = deriveContextUsageBar({
    used: 4100,
    total: null,
    promptTokens: 4100,
    completionTokens: 0,
    estimated: true,
  });
  assert.ok(state);
  assert.equal(state.face, "~4.1k tokens");
  assert.equal(state.compactFace, "~4.1k");
  assert.equal(state.totalRowValue, "~4,100");
  assert.equal(state.percent, null);
  assert.equal(state.hasUsageDetails, false);
});

test("an estimate against a window gives a ratio but no limit advice", () => {
  const state = deriveContextUsageBar({ used: 30000, total: 32768, estimated: true });
  assert.ok(state);
  assert.equal(state.face, "~30.0k / 32.8k");
  assert.equal(state.percent, (30000 / 32768) * 100);
  assert.equal(state.advice, "none");
  const exact = deriveContextUsageBar({ used: 30000, total: 32768 });
  assert.equal(exact?.advice, "stops-at-limit");
});
