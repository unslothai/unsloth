// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  estimateMessagesTokenCount,
  extractMessageText,
  orderMessagesByBranch,
} from "../src/features/chat/utils/estimate-chat-tokens.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";

test("extractMessageText extracts string content", () => {
  assert.equal(extractMessageText("Hello world"), "Hello world");
});

test("extractMessageText extracts array text and reasoning parts", () => {
  const content = [
    { type: "text" as const, text: "Hello " },
    { type: "reasoning" as const, text: "Thinking... " },
    { type: "text" as const, text: "World!" },
  ];
  assert.equal(extractMessageText(content), "Hello Thinking... World!");
});

test("extractMessageText handles empty or non-text parts safely", () => {
  const content = [
    { type: "image", image_url: "data:..." },
    { type: "tool-call", name: "calculator" },
  ];
  assert.equal(extractMessageText(content), "");
});

test("estimateMessagesTokenCount returns null for empty messages or blank text", () => {
  assert.equal(estimateMessagesTokenCount([]), null);
  assert.equal(estimateMessagesTokenCount(null), null);
  assert.equal(estimateMessagesTokenCount(undefined), null);

  const blankMessages: MessageRecord[] = [
    {
      id: "1",
      threadId: "t1",
      role: "user",
      content: [{ type: "text", text: "" }],
      createdAt: 1000,
    },
  ];
  assert.equal(estimateMessagesTokenCount(blankMessages), null);
});

test("estimateMessagesTokenCount calculates char length / 4 rounded", () => {
  // 3600 characters => 900 tokens
  const text = "a".repeat(3600);
  const messages: MessageRecord[] = [
    {
      id: "1",
      threadId: "t1",
      role: "user",
      content: [{ type: "text", text }],
      createdAt: 1000,
    },
  ];
  assert.equal(estimateMessagesTokenCount(messages), 900);
});

test("estimateMessagesTokenCount returns minimum 1 for non-empty short text", () => {
  const messages: MessageRecord[] = [
    {
      id: "1",
      threadId: "t1",
      role: "user",
      content: [{ type: "text", text: "hi" }],
      createdAt: 1000,
    },
  ];
  assert.equal(estimateMessagesTokenCount(messages), 1);
});

test("orderMessagesByBranch traverses active branch parent chain", () => {
  const messages: MessageRecord[] = [
    {
      id: "1",
      threadId: "t1",
      role: "user",
      content: [{ type: "text", text: "Prompt 1" }],
      createdAt: 100,
    },
    {
      id: "2",
      threadId: "t1",
      role: "assistant",
      parentId: "1",
      content: [{ type: "text", text: "Reply 1A" }],
      createdAt: 200,
    },
    {
      id: "3",
      threadId: "t1",
      role: "assistant",
      parentId: "1",
      content: [{ type: "text", text: "Reply 1B (retry)" }],
      createdAt: 300,
    },
    {
      id: "4",
      threadId: "t1",
      role: "user",
      parentId: "3",
      content: [{ type: "text", text: "Prompt 2" }],
      createdAt: 400,
    },
  ];

  const branch = orderMessagesByBranch(messages);
  assert.deepEqual(
    branch.map((m) => m.id),
    ["1", "3", "4"],
  );

  // Character count should only include branch ["1", "3", "4"] and not "2"
  const tokens = estimateMessagesTokenCount(messages);
  const expectedChars =
    "Prompt 1".length + "Reply 1B (retry)".length + "Prompt 2".length;
  assert.equal(tokens, Math.max(1, Math.round(expectedChars / 4)));
});
