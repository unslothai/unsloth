// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { conversationJsonlBody } from "../src/features/chat/utils/ndjson.ts";

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
