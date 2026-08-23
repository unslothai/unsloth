// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { mergeConsecutiveSameRoleMessages } from "../src/features/chat/utils/merge-same-role-messages.ts";

test("merges consecutive user messages left by cancel + resend", () => {
  const merged = mergeConsecutiveSameRoleMessages([
    { role: "system", content: "sys" },
    { role: "user", content: "hello" },
    { role: "user", content: "hello" },
  ]);
  assert.equal(merged.length, 2);
  assert.equal(merged[0].role, "system");
  assert.equal(merged[1].role, "user");
  assert.equal(merged[1].content, "hello\n\nhello");
});

test("merges a restored system message into the prepended one", () => {
  const merged = mergeConsecutiveSameRoleMessages([
    { role: "system", content: "one" },
    { role: "system", content: "two" },
    { role: "user", content: "hi" },
  ]);
  assert.equal(merged.length, 2);
  assert.deepEqual(merged[0], { role: "system", content: "one\n\ntwo" });
});

test("never merges assistant turns that carry tool_calls", () => {
  const toolCalls = [
    { id: "call-1", type: "function" as const, function: { name: "f", arguments: "{}" } },
  ];
  const merged = mergeConsecutiveSameRoleMessages([
    {
      role: "assistant",
      content: null,
      tool_calls: toolCalls,
    },
    { role: "assistant", content: "plain text" },
    { role: "tool", tool_call_id: "call-1", content: "result" },
    { role: "tool", tool_call_id: "call-1", content: "result2" },
  ]);
  // All four survive: the tool_call turn is kept whole so its role="tool"
  // results keep matching ids; role="tool" never merges.
  assert.equal(merged.length, 4);
  assert.deepEqual(merged[0].tool_calls, toolCalls);
});

test("assistant text turns still merge when no tool calls involved", () => {
  const merged = mergeConsecutiveSameRoleMessages([
    { role: "user", content: "q" },
    { role: "assistant", content: "a1" },
    { role: "assistant", content: "a2" },
  ]);
  assert.equal(merged.length, 2);
  assert.equal(merged[1].content, "a1\n\na2");
});

test("preserves image parts in order while joining adjacent text", () => {
  const image = { type: "image_url" as const, image_url: { url: "data:image/png;base64,x" } };
  const merged = mergeConsecutiveSameRoleMessages([
    { role: "user", content: [image, { type: "text", text: "a" }] },
    { role: "user", content: [{ type: "text", text: "b" }, image] },
  ]);
  assert.equal(merged.length, 1);
  const parts = merged[0].content as Array<{ type: string }>;
  assert.deepEqual(parts.map((p) => p.type), ["image_url", "text", "image_url"]);
  assert.equal((parts[1] as { text: string }).text, "a\n\nb");
});

test("collapses empty-string neighbors without leaving stray separators", () => {
  const merged = mergeConsecutiveSameRoleMessages([
    { role: "user", content: "" },
    { role: "user", content: "real" },
    { role: "user", content: "" },
  ]);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].content, "real");
});
