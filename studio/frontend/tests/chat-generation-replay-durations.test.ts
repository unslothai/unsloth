// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A card reads its seconds from `message.metadata.custom.reasoningDurations[groupIndex]`, and the live
// stream fills that array from `Date.now()` between two chunks. A follower has no such clock: every
// frame it folds was written before this tab existed, so timing them against now collapses every
// thought to "Thought for 0 seconds". These pin the fix -- the frames ARE the clock, the groups are
// counted on the same parts array the renderer indexes by, and a duration the closing tab already
// measured survives the replay that could never have measured it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

const read = (relative: string) =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

registerBundlerResolver();

const { createRecoveryReplay } = await import(
  "../src/features/chat/utils/chat-generation-replay.ts"
);
const { resolveReasoningGroupDuration } = await import(
  "../src/features/chat/utils/reasoning-duration.ts"
);

const thought = (text: string) => ({
  choices: [{ delta: { reasoning_content: text } }],
});
const answer = (text: string) => ({ choices: [{ delta: { content: text } }] });
const call = (toolCallId: string) => ({
  _toolEvent: {
    type: "tool_start",
    tool_call_id: toolCallId,
    tool_name: "bash",
    arguments: {},
    arguments_text: "{}",
  },
});

test("a thought that ran while the tab was shut keeps the seconds it took", () => {
  // The frames carry the instant the RUN wrote them, which is the only clock a replay has.
  const replay = createRecoveryReplay("");
  replay.applyChunk(thought("Thinking..."), 10_000);
  replay.applyChunk(thought(" still thinking"), 12_000);
  replay.applyChunk(answer("the answer"), 14_000);
  // Four seconds, not zero, and not fourteen: the timer stops where the block closed.
  assert.deepEqual(replay.durations().reasoningDurations, [4]);
  replay.applyChunk(answer(" keeps going long after"), 60_000);
  assert.deepEqual(
    replay.durations().reasoningDurations,
    [4],
    "the answer running on must not run the thought's clock on with it",
  );
});

test("a call between two thoughts ends the first one's clock and starts the next", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk(thought("first thought"), 1_000);
  // A call landing on the reply is a boundary: the thought before it ended at this frame.
  replay.applyChunk(call("call_0"), 5_000);
  replay.applyChunk(thought("second thought"), 8_000);
  replay.applyChunk(answer("done"), 15_000);
  assert.deepEqual(replay.durations().reasoningDurations, [4, 7]);
  // The card renders the LAST group's number, which is what `metadata()` hands back.
  assert.equal(replay.durations().reasoningDuration, 7);
});

test("a server summary lands in its own group instead of shifting the later ones", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk(thought("first thought"), 1_000);
  // The backend emits one summary at the end of a visible pass. Appending it would have read [4, 2, 7].
  replay.recordServerDuration(2_000);
  replay.applyChunk(call("call_0"), 5_000);
  replay.applyChunk(thought("second thought"), 8_000);
  replay.applyChunk(answer("done"), 15_000);
  assert.deepEqual(
    replay.durations().reasoningDurations,
    [2, 7],
    "the server's number wins for the group it describes, and only that one",
  );
});

test("what a closing tab already measured survives the replay that could not measure it", () => {
  // The frames below this reader's cursor were never folded here, so group 0 keeps the value it
  // arrived with, and the group this reader DID watch takes the next slot.
  const replay = createRecoveryReplay(
    [{ type: "reasoning", text: "thought one" }],
    [4],
  );
  replay.applyChunk(call("call_0"), 5_000);
  replay.applyChunk(thought("thought two"), 8_000);
  replay.applyChunk(answer("the answer"), 15_000);
  assert.deepEqual(replay.durations().reasoningDurations, [4, 7]);
});

test("nothing measured publishes nothing", () => {
  // A replay that measured no group must not overwrite a stored array with an empty one.
  const replay = createRecoveryReplay("");
  replay.applyChunk(answer("no thought at all"), 1_000);
  assert.deepEqual(replay.durations(), {});
});

test("what the reader reads off the finished message is the seconds it took", () => {
  // The contract is not `durations()`, it is what `resolveReasoningGroupDuration` answers for a part
  // index -- that is the number the card renders. Fold a whole run the way the follower does, publish
  // the way the follower publishes, and read the card back out.
  const replay = createRecoveryReplay([]);
  replay.applyChunk(thought("first thought"), 1_000);
  replay.applyChunk(call("call_0"), 5_000);
  replay.applyChunk(thought("second thought"), 8_000);
  replay.applyChunk(answer("the answer"), 15_000);
  const parts = replay.content() as Array<{ type: string; text?: string }>;
  assert.deepEqual(
    parts.map((part) => part.type),
    ["reasoning", "tool-call", "reasoning", "text"],
  );
  const custom = { ...replay.durations() };
  const reasoningIndexes = parts.reduce<number[]>(
    (acc, part, i) => (part.type === "reasoning" ? [...acc, i] : acc),
    [],
  );
  assert.equal(
    resolveReasoningGroupDuration(parts, reasoningIndexes[0], custom),
    4,
  );
  assert.equal(
    resolveReasoningGroupDuration(parts, reasoningIndexes[1], custom),
    7,
  );
});

test("the follower feeds the clock, the seed and the publish", () => {
  // The rules above live in the accumulator; they only reach a card through three call sites in the
  // follower. Pin them, the way the gate is pinned: deleting one restores "0 seconds" silently.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.ok(
    /createRecoveryReplay\(\s*storedMessage\.content,\s*\n?\s*Array\.isArray\(metadata\.reasoningDurations\)/.test(
      provider,
    ),
    "the accumulator must inherit what the closing tab already measured",
  );
  assert.ok(
    provider.includes("update.event.createdAt,"),
    "groups must be timed on the run's own frame timestamps, not on this tab's clock",
  );
  assert.ok(
    provider.includes("current: { ...currentMetadata, ...replay.durations() }"),
    "what was measured has to be published with the reply, or nothing reads it",
  );
  assert.ok(
    provider.includes("replay.recordServerDuration(chunk._reasoningDurationMs)"),
    "a server summary must land in its own group instead of shifting the later ones along",
  );
  assert.equal(
    provider.includes("recoveredReasoningSummaryMetadata"),
    false,
    "the append-only helper is back: it shifts every group after the summary by one",
  );
});
