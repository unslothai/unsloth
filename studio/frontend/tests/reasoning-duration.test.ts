// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  countReasoningGroups,
  createReasoningDurationTracker,
  lastReasoningGroupTextLength,
  resolveReasoningGroupDuration,
} from "../src/features/chat/utils/reasoning-duration.ts";
import { extractDeltaText } from "../src/features/chat/utils/parse-assistant-content.ts";

const separatedReasoning = [
  { type: "reasoning" },
  { type: "tool-call" },
  { type: "reasoning" },
  { type: "text" },
];

test("selects per-group durations while preserving legacy messages", () => {
  const current = {
    reasoningDuration: 5,
    reasoningDurations: [2, 5],
  };
  assert.equal(resolveReasoningGroupDuration(separatedReasoning, 0, current), 2);
  assert.equal(resolveReasoningGroupDuration(separatedReasoning, 2, current), 5);
  assert.equal(countReasoningGroups(separatedReasoning), 2);

  const legacy = { reasoningDuration: 5 };
  assert.equal(
    resolveReasoningGroupDuration(separatedReasoning, 0, legacy),
    undefined,
  );
  assert.equal(resolveReasoningGroupDuration(separatedReasoning, 2, legacy), 5);

  const contiguous = [
    { type: "reasoning" },
    { type: "reasoning" },
    { type: "text" },
  ];
  assert.equal(countReasoningGroups(contiguous), 1);
  assert.equal(
    resolveReasoningGroupDuration(contiguous, 0, {
      reasoningDurations: [3],
    }),
    3,
  );
});

test("tracks the exact reasoning, tool, reasoning sequence", () => {
  let now = 0;
  const tracker = createReasoningDurationTracker(() => now);

  tracker.startGroup();
  now = 1_200;
  tracker.recordServerDuration(2_000);
  tracker.finishGroup();

  tracker.startGroup();
  now = 5_600;
  tracker.recordServerDuration(5_000);
  tracker.finishGroup();

  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 5,
    reasoningDurations: [2, 5],
  });
});

test("keeps groups aligned when summaries are missing or orphaned", () => {
  let now = 0;
  const tracker = createReasoningDurationTracker(() => now);

  tracker.startGroup();
  now = 2_000;
  tracker.finishGroup();

  tracker.startGroup();
  now = 7_000;
  tracker.recordServerDuration(5_000);
  tracker.finishGroup();
  tracker.recordServerDuration(9_000);

  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 5,
    reasoningDurations: [2, 5],
  });
});

test("accepts zero after closure and rejects malformed server timing", () => {
  let now = 0;
  const tracker = createReasoningDurationTracker(() => now);

  tracker.startGroup();
  now = 1_000;
  tracker.finishGroup();
  assert.equal(tracker.recordServerDuration(0), true);
  assert.equal(tracker.recordServerDuration(-1), false);

  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 0,
    reasoningDurations: [0],
  });
});

test("omits unknown timing and falls back to elapsed time", () => {
  let now = 0;
  const tracker = createReasoningDurationTracker(() => now);

  tracker.startGroup();
  assert.deepEqual(tracker.metadata(), {});

  now = 3_200;
  tracker.finishGroup();
  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 3,
    reasoningDurations: [3],
  });
});

test("keeps structured reasoning active only when it is the final content", () => {
  assert.deepEqual(
    extractDeltaText([{ type: "reasoning", text: "First" }]),
    {
      text: "<think>First</think>",
      structuredReasoningContinues: true,
    },
  );
  assert.deepEqual(
    extractDeltaText([
      { type: "reasoning", text: "Last thought" },
      { type: "text", text: "Answer" },
    ]),
    {
      text: "<think>Last thought</think>Answer",
      structuredReasoningContinues: false,
    },
  );
  assert.deepEqual(
    extractDeltaText([
      { type: "text", text: "Preface" },
      { type: "reasoning", text: "First thought" },
    ]),
    {
      text: "Preface<think>First thought</think>",
      structuredReasoningContinues: true,
    },
  );
});

test("keeps a coalesced reasoning group growing across atomic blocks", () => {
  let now = 1_770_000_000_000;
  const tracker = createReasoningDurationTracker(() => now);

  // A provider that closes every reasoning block in its own chunk still
  // belongs to ONE rendered group, so the timer must span all of them.
  tracker.startGroup();
  tracker.resumeGroup(0, "first block".length);
  tracker.finishGroup();

  now += 3_000;
  tracker.resumeGroup(0, "first blocksecond block".length);
  tracker.finishGroup();

  // The answer that follows adds no reasoning text, so the timer stops here.
  now += 3_000;
  tracker.resumeGroup(0, "first blocksecond block".length);
  tracker.finishGroup();

  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 3,
    reasoningDurations: [3],
  });
});

test("never persists a hole when one delta reveals several groups", () => {
  let now = 1_770_000_000_000;
  const tracker = createReasoningDurationTracker(() => now);

  // Index 0 was never started explicitly: it became visible and closed inside
  // the same chunk that revealed index 1.
  tracker.startGroup(1);
  now += 4_000;
  tracker.finishGroup();

  const metadata = tracker.metadata();
  const durations = metadata.reasoningDurations as number[];
  assert.equal(durations.length, 2);
  assert.ok(durations.every((value) => typeof value === "number"));
  assert.deepEqual(JSON.parse(JSON.stringify(durations)), [0, 4]);
});

test("a server duration is never overwritten by local timing", () => {
  let now = 1_770_000_000_000;
  const tracker = createReasoningDurationTracker(() => now);

  tracker.startGroup();
  tracker.recordServerDuration(2_000);
  now += 30_000;
  tracker.resumeGroup(0, 99);
  tracker.finishGroup();

  assert.deepEqual(tracker.metadata(), {
    reasoningDuration: 2,
    reasoningDurations: [2],
  });
});

test("lastReasoningGroupTextLength measures only the last reasoning group", () => {
  assert.equal(
    lastReasoningGroupTextLength([
      { type: "reasoning", text: "aaaa" },
      { type: "tool-call" },
      { type: "reasoning", text: "bb" },
      { type: "reasoning", text: "c" },
    ]),
    3,
  );
  // The answer that follows is not reasoning, so it does not count -- but the
  // group itself is still measured, which is what lets resumeGroup see that the
  // reasoning has stopped growing.
  assert.equal(
    lastReasoningGroupTextLength([
      { type: "reasoning", text: "aaaa" },
      { type: "text", text: "answer" },
    ]),
    4,
  );
  assert.equal(
    lastReasoningGroupTextLength([{ type: "text", text: "answer only" }]),
    0,
  );
  assert.equal(lastReasoningGroupTextLength([]), 0);
});
