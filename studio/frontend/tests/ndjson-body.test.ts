// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import {
  canMergeConversationExport,
  conversationJsonlBody,
  exportFormatIncludesSiblings,
  isOpenAIMessageRecord,
  messageJsonlConversationRecord,
  ndjsonBody,
} from "../src/features/chat/utils/ndjson.ts";

const messages = [
  { role: "user", content: "Hello" },
  { role: "assistant", content: "Hi" },
];

test("training JSONL keeps one conversation per record", () => {
  assert.equal(
    conversationJsonlBody(messages, "training"),
    '{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}',
  );
});

test("message JSONL writes one message per record", () => {
  const body = conversationJsonlBody(messages, "messages");
  assert.equal(
    body,
    '{"role":"user","content":"Hello"}\n{"role":"assistant","content":"Hi"}',
  );
  assert.deepEqual(
    body.split("\n").map((line) => JSON.parse(line)),
    messages,
  );
});

test("empty conversations produce an empty body", () => {
  assert.equal(conversationJsonlBody([], "training"), '{"messages":[]}');
  assert.equal(conversationJsonlBody([], "messages"), "");
});

test("terminates a single record with a newline", () => {
  assert.equal(ndjsonBody(['{"messages":[]}']), '{"messages":[]}\n');
});

test("separates and terminates every record", () => {
  assert.equal(ndjsonBody(['{"a":1}', '{"b":2}']), '{"a":1}\n{"b":2}\n');
});

test("concatenated bodies stay parseable line by line", () => {
  const combined = ndjsonBody(['{"a":1}']) + ndjsonBody(['{"b":2}']);
  const parsed = combined
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as Record<string, number>);
  assert.deepEqual(parsed, [{ a: 1 }, { b: 2 }]);
});

test("returns an empty body when there are no records", () => {
  assert.equal(ndjsonBody([]), "");
});

test("message JSONL records form one importable conversation", () => {
  assert.equal(isOpenAIMessageRecord(messages[0]), true);
  assert.equal(
    isOpenAIMessageRecord({ role: "developer", content: "Follow policy" }),
    true,
  );
  assert.equal(isOpenAIMessageRecord({ messages }), false);
  assert.deepEqual(messageJsonlConversationRecord(messages), { messages });
});

test("message JSONL cannot merge conversations without losing boundaries", () => {
  assert.equal(canMergeConversationExport("jsonl-messages"), false);
  assert.equal(canMergeConversationExport("jsonl-raw"), true);
});

test("both JSONL layouts export only the displayed branch", () => {
  assert.equal(exportFormatIncludesSiblings("jsonl-raw"), false);
  assert.equal(exportFormatIncludesSiblings("jsonl-messages"), false);
  assert.equal(exportFormatIncludesSiblings("csv"), true);
  assert.equal(exportFormatIncludesSiblings("sharegpt"), true);
});

test("training order excludes abandoned response branches", () => {
  const branched = [
    { id: "user", parentId: null, createdAt: 1 },
    { id: "old-reply", parentId: "user", createdAt: 2 },
    { id: "new-reply", parentId: "user", createdAt: 3 },
    { id: "follow-up", parentId: "new-reply", createdAt: 4 },
  ];
  assert.deepEqual(
    orderByParentChain(branched, { includeSiblings: false }).map(
      ({ id }) => id,
    ),
    ["user", "new-reply", "follow-up"],
  );
});

test("training order follows the displayed branch ancestor chain", () => {
  const branched = [
    { id: "user-1", parentId: null, createdAt: 1, role: "user" },
    { id: "selected", parentId: "user-1", createdAt: 2, role: "assistant" },
    { id: "abandoned", parentId: "user-1", createdAt: 5, role: "assistant" },
    { id: "user-2", parentId: "selected", createdAt: 6, role: "user" },
    { id: "answer", parentId: "user-2", createdAt: 7, role: "assistant" },
  ];

  assert.deepEqual(
    orderByParentChain(branched, { includeSiblings: false }).map(
      ({ id }) => id,
    ),
    ["user-1", "selected", "user-2", "answer"],
  );
});
