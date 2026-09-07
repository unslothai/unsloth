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

const { createRecoveryReplay } = await import(
  "../src/features/chat/utils/chat-generation-replay.ts"
);

const { parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
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

// The live stream closes an open thought before it records a call, so the block's close tag sits at the
// END of the thought run. A replay seeded from parts re-creates those tags from what storage holds, and
// a flag carried across the whole reply keeps closing a block a boundary already closed -- the stray
// `< /think>` then survives as literal text in the part after it. What the reader saw was an answer
// whose first word was a tag, doubled on every reopen.
test("a call landing mid-thought closes the block where the live stream closed it", () => {
  const replay = createRecoveryReplay([reasoning("the thought")]);
  replay.applyChunk({
    choices: [
      { delta: { tool_calls: [{ index: 0, function: { name: "bash", arguments: "{}" } }] } },
    ],
  });
  replay.applyChunk({ choices: [{ delta: { content: "the answer" } }] });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    parts.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
    "the call sits between the thought and the answer it produced",
  );
  assert.equal(
    (parts[0] as { text: string }).text,
    "the thought",
    "the close belongs to the end of the thought, never to the start of the answer",
  );
  assert.equal((parts[2] as { text: string }).text, "the answer");
});

test("reopening at any point of a run shows what a stream that never left shows", () => {
  // The reader must not be able to tell the tab was shut: replaying from what storage holds has to
  // yield exactly what the same frames would have produced had they arrived in one sitting.
  const frames = [
    { choices: [{ delta: { reasoning_content: "think A" } }] },
    {
      choices: [
        { delta: { tool_calls: [{ index: 0, function: { name: "read_file", arguments: "{}" } }] } },
      ],
    },
    { choices: [{ delta: { content: "the answer" } }] },
    { choices: [{ delta: { reasoning_content: "a second thought" } }] },
    { choices: [{ delta: { content: " and its tail" } }] },
  ];
  const neverLeft = createRecoveryReplay("");
  for (const frame of frames) neverLeft.applyChunk(frame);
  const expected = neverLeft.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    expected.map((part) => part.type),
    ["reasoning", "tool-call", "text", "reasoning", "text"],
  );
  for (let cut = 1; cut <= frames.length; cut++) {
    const seed = createRecoveryReplay("");
    for (const frame of frames.slice(0, cut)) seed.applyChunk(frame);
    const reopened = createRecoveryReplay(seed.content());
    for (const frame of frames.slice(cut)) reopened.applyChunk(frame);
    assert.deepEqual(
      reopened.content() as Array<Record<string, unknown>>,
      expected,
      `reopening after ${cut} of ${frames.length} frames must show the same reply`,
    );
  }
});

// A model that carries its OWN tags inside delta.content already classifies its own thought: the live
// stream appends such a delta verbatim and lets the tags inside it do the work. The replay closed the
// block it happened to sit in regardless of who opened it, so the first answer delta of a tagged run cut
// the thought mid-sentence and the model's own close tag survived as literal text in the answer part.
test("a run whose own tags carry the thought replays as the live stream read it", () => {
  const frames = [
    { choices: [{ delta: { content: "<think>the thought " } }] },
    { choices: [{ delta: { content: "still thinking</think>" } }] },
    { choices: [{ delta: { content: "the answer" } }] },
  ];
  // What the tab that never left renders: the buffer the live adapter appends to, parsed once.
  const live = parseAssistantContent(
    frames.map((frame) => frame.choices[0].delta.content as string).join(""),
  );
  const replay = createRecoveryReplay("");
  for (const frame of frames) replay.applyChunk(frame, 1000);
  assert.deepEqual(
    replay.content() as Array<Record<string, unknown>>,
    live,
    "the whole thought stays reasoning and only the answer becomes a text part",
  );
});

test("reopening at any point of a tagged run shows what a stream that never left shows", () => {
  const frames = [
    { choices: [{ delta: { content: "<think>the thought " } }] },
    { choices: [{ delta: { content: "still thinking</think>" } }] },
    {
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, id: "call_0", function: { name: "bash", arguments: "{}" } }],
          },
        },
      ],
    },
    { choices: [{ delta: { content: "the answer" } }] },
  ];
  const neverLeft = createRecoveryReplay("");
  for (const frame of frames) neverLeft.applyChunk(frame);
  const expected = neverLeft.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    expected.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
    "the model's own tags classify the reply exactly once, and a reopen does not add a second pair",
  );
  for (let cut = 1; cut <= frames.length; cut++) {
    const seed = createRecoveryReplay("");
    for (const frame of frames.slice(0, cut)) seed.applyChunk(frame);
    const reopened = createRecoveryReplay(seed.content());
    for (const frame of frames.slice(cut)) reopened.applyChunk(frame);
    assert.deepEqual(
      reopened.content() as Array<Record<string, unknown>>,
      expected,
      `reopening after ${cut} of ${frames.length} frames must show the same reply`,
    );
  }
});

// A structured `thinking` part arrives inside delta.content already wrapped in a full pair of tags by
// extractDeltaText. When the block was opened by THIS replay (a reasoning_content frame before it), that
// second open tag is not a no-op for the parser: it survives as literal text inside the thought. A chunk
// carrying only a close tag is the opposite case -- that tag ends whichever block is open either way.
test("a thinking part with its own tags lands in the thought the replay opened", () => {
  const frames = [
    { choices: [{ delta: { reasoning_content: "first thought " } }] },
    { choices: [{ delta: { content: [{ type: "thinking", text: "structured thought" }] } }] },
    { choices: [{ delta: { content: [{ type: "text", text: "the answer" }] } }] },
  ];
  // What the live path appends, in order: it wraps reasoning_content itself and appends extractDeltaText's
  // output verbatim, so parsing THAT string is what a reopened tab has to reproduce exactly.
  const live = parseAssistantContent("<think>first thought </think><think>structured thought</think>the answer");
  const replay = createRecoveryReplay("");
  for (const frame of frames) replay.applyChunk(frame, 1000);
  assert.deepEqual(
    replay.content() as Array<Record<string, unknown>>,
    live,
    "a second open tag must not survive as literal text inside the thought",
  );
});
