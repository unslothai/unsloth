// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recovery follower replays a run's stored chunk events. It used to fold them into ONE string and
// re-parse that string on every publish, which had two consequences: the work was quadratic in the
// length of the answer, and a tool-heavy reply had no string form at all -- its calls are PARTS, so
// the replay dropped them and a reopened message lost its pills and their output. These pin the
// accumulator that replaced the string: the same parts the live stream builds, in the order the run
// produced them, extended one event at a time.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createRecoveryReplay, seededReplayState } = await import(
  "../src/features/chat/utils/chat-generation-replay.ts"
);

const text = (t: string) => ({ type: "text", text: t });
const reasoning = (t: string) => ({ type: "reasoning", text: t });
const tool = (toolCallId: string, toolName: string, extra: object = {}) => ({
  type: "tool-call",
  toolCallId,
  toolName,
  args: {},
  argsText: "{}",
  ...extra,
});

test("a seeded reply keeps the call between its two paragraphs", () => {
  // The flattened string cannot say this. `generationRawContent` could only ever hand back the two
  // runs of prose, and `parseAssistantContent` put them side by side.
  const replay = createRecoveryReplay([
    reasoning("It should sit between the paragraphs."),
    tool("call_0:uuid", "read_file", { result: "the file body" }),
    text("and here is what it said."),
  ]);
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    parts.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
  );
  assert.equal(parts[1].toolCallId, "call_0:uuid");
  assert.equal(parts[1].result, "the file body");
});

test("a replayed tool_start lands where the call happened, not at the end", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "before " } }] });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_start",
      tool_call_id: "call_0",
      tool_name: "read_file",
      arguments: { path: "a.txt" },
      arguments_text: '{"path":"a.txt"}',
    },
  });
  replay.applyChunk({ choices: [{ delta: { content: "after" } }] });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["text", "tool-call", "text"]);
  assert.equal(parts[1].toolName, "read_file");
  assert.deepEqual(parts[1].args, { path: "a.txt" });
  assert.equal(
    (parts[0] as { text: string }).text + (parts[2] as { text: string }).text,
    "before after",
    "the prose around a call must survive in order",
  );
});

test("tool_end carries the result onto the card its id names", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "x" } }] });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_0", tool_name: "bash", arguments: {} },
  });
  replay.applyChunk({ _toolEvent: { type: "tool_output", tool_call_id: "call_0", text: "line one\n" } });
  replay.applyChunk({
    _toolEvent: { type: "tool_end", tool_call_id: "call_0", result: "exit 0" },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const call = parts.find((part) => part.type === "tool-call")!;
  assert.equal(call.toolName, "bash");
  // The longer captured stream beats the status line, exactly as the live path decides.
  assert.match(String(call.result), /line one/);
  assert.match(String(call.result), /exit 0/);
});

test("index-keyed tool_call deltas accumulate into the ONE card they belong to", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, function: { name: "read_file", arguments: "{\"a\":" } },
          ],
        },
      },
    ],
  });
  replay.applyChunk({
    choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: " 1}" } }] } }],
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const calls = parts.filter((part) => part.type === "tool-call");
  assert.equal(calls.length, 1, "a continuing call must not open a second card");
  assert.deepEqual(calls[0].args, { a: 1 });
});

test("the backend's own spelling of a minted id reaches the SAME card", () => {
  // Providers stream `tool_calls` with no id; the backend mints `tool_call_<index>` for the slot
  // under the same rule the replay mints it, so its `tool_start` lands on the card the fragments
  // opened instead of opening a second one. A DIFFERENT id is a different call, and gets its own.
  const replay = createRecoveryReplay([]);
  replay.applyChunk({
    choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "grep", arguments: "{}" } }] } }],
  });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "tool_call_0", tool_name: "grep", arguments: {} },
  });
  let parts = replay.content() as Array<Record<string, unknown>>;
  assert.equal(parts.filter((part) => part.type === "tool-call").length, 1);
  assert.equal(parts.find((part) => part.type === "tool-call")?.toolCallId, "tool_call_0");

  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_1", tool_name: "bash", arguments: {} },
  });
  parts = replay.content() as Array<Record<string, unknown>>;
  const calls = parts.filter((part) => part.type === "tool-call");
  assert.equal(calls.length, 2, "a second call is a second card, not a rename of the first");
  assert.equal(calls[1].toolName, "bash");
});

test("reasoning reopens only when the seed left it open", () => {
  const open = createRecoveryReplay([reasoning("the thought")]);
  open.applyChunk({ choices: [{ delta: { content: "the answer" } }] });
  const parts = open.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["reasoning", "text"]);
  assert.equal((parts[0] as { text: string }).text, "the thought");
  assert.equal((parts[1] as { text: string }).text, "the answer");

  const closed = createRecoveryReplay([text("plain")]);
  closed.applyChunk({ choices: [{ delta: { reasoning_content: "fresh thought" } }] });
  assert.deepEqual(
    (closed.content() as Array<{ type: string }>).map((part) => part.type),
    ["text", "reasoning"],
  );
});

test("a reasoning_summary frame is not content, and does not extend the reply", () => {
  const replay = createRecoveryReplay("answer");
  assert.equal(replay.applyChunk({ _reasoningDurationMs: 1200 }), false);
  assert.equal(replay.rawText(), "answer");
});
