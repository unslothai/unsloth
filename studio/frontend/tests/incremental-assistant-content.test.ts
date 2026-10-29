// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createSegmentedAssistantText } from "../src/features/chat/utils/incremental-assistant-content.ts";
import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";

/** Oracle: the original full reparse at the tool cursors. Must stay a plain rewrite of it. */
function referenceRuns(
  rawText: string,
  boundaries: readonly number[],
): ReturnType<typeof parseAssistantContent>[] {
  const out: ReturnType<typeof parseAssistantContent>[] = [];
  let from = 0;
  for (const boundary of boundaries) {
    out.push(parseAssistantContent(rawText.slice(from, boundary)));
    from = boundary;
  }
  out.push(parseAssistantContent(rawText.slice(from)));
  return out;
}

function makeRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

// Pieces split tags at every offset and often produce tag lookalikes ("<thi", "<<think>").
const PIECES = [
  "a",
  " ",
  "\n",
  "<",
  "</",
  "<t",
  "<th",
  "<thi",
  "<thin",
  "<think",
  "<think>",
  "hink>",
  "think>",
  "k>",
  ">",
  "/think>",
  "</think>",
  "</thi",
  "word ",
  "<think></think>",
  "<think>x",
  "x</think>",
  "",
];

function randomReply(random: () => number, pieces: number): string {
  let out = "";
  for (let index = 0; index < pieces; index += 1) {
    out += PIECES[Math.floor(random() * PIECES.length)];
  }
  return out;
}

function randomArrivals(random: () => number, text: string): string[] {
  const out: string[] = [];
  let at = 0;
  while (at < text.length) {
    const size = 1 + Math.floor(random() * 5);
    out.push(text.slice(at, at + size));
    at += size;
  }
  return out;
}

test("the incremental parse matches a full reparse at every arrival", () => {
  const CASES = 3000;
  let states = 0;
  for (let seed = 1; seed <= CASES; seed += 1) {
    const random = makeRandom(seed);
    const text = randomReply(random, 1 + Math.floor(random() * 20));
    const arrivals = randomArrivals(random, text);
    const segmented = createSegmentedAssistantText();
    let accumulated = "";
    for (const arrival of arrivals) {
      accumulated += arrival;
      segmented.appendText(arrival);
      states += 1;
      assert.deepEqual(
        segmented.runs(accumulated, []),
        referenceRuns(accumulated, []),
        `seed ${seed}: diverged at ${JSON.stringify(accumulated)}`,
      );
    }
  }
  assert.equal(states > 10_000, true, `only ${states} states exercised`);
});

test("the incremental parse matches a full reparse across tool boundaries", () => {
  const CASES = 3000;
  let boundaryStates = 0;
  for (let seed = 1; seed <= CASES; seed += 1) {
    const random = makeRandom(seed + 500_000);
    const text = randomReply(random, 1 + Math.floor(random() * 20));
    const arrivals = randomArrivals(random, text);
    const segmented = createSegmentedAssistantText();
    let accumulated = "";
    const boundaries: number[] = [];
    for (const arrival of arrivals) {
      accumulated += arrival;
      segmented.appendText(arrival);
      if (random() < 0.2) {
        if (boundaries[boundaries.length - 1] !== accumulated.length) {
          boundaries.push(accumulated.length);
          boundaryStates += 1;
        }
      }
      assert.deepEqual(
        segmented.runs(accumulated, boundaries),
        referenceRuns(accumulated, boundaries),
        `seed ${seed}: diverged at ${JSON.stringify(accumulated)} with boundaries ${boundaries.join(",")}`,
      );
    }
  }
  assert.equal(
    boundaryStates > 1000,
    true,
    `only ${boundaryStates} boundaries exercised`,
  );
});

test("the incremental parse recovers when a suffix is removed", () => {
  const CASES = 1500;
  let truncations = 0;
  for (let seed = 1; seed <= CASES; seed += 1) {
    const random = makeRandom(seed + 900_000);
    const text = randomReply(random, 2 + Math.floor(random() * 20));
    const arrivals = randomArrivals(random, text);
    const segmented = createSegmentedAssistantText();
    let accumulated = "";
    for (const arrival of arrivals) {
      accumulated += arrival;
      segmented.appendText(arrival);
      if (random() < 0.15 && accumulated.length > 1) {
        accumulated = accumulated.slice(
          0,
          Math.floor(random() * accumulated.length),
        );
        truncations += 1;
      }
      assert.deepEqual(
        segmented.runs(accumulated, []),
        referenceRuns(accumulated, []),
        `seed ${seed}: diverged after truncation at ${JSON.stringify(accumulated)}`,
      );
    }
  }
  assert.equal(
    truncations > 500,
    true,
    `only ${truncations} truncations exercised`,
  );
});

test("a rewritten prefix is reparsed rather than extended", () => {
  // What `mergeContinuation` does; the cache is built with the fast path off for that case.
  const segmented = createSegmentedAssistantText({ trustAppends: false });
  segmented.appendText("<think>one</think>two");
  assert.deepEqual(
    segmented.runs("<think>ONE</think>two", []),
    referenceRuns("<think>ONE</think>two", []),
  );
  assert.deepEqual(
    segmented.runs("<think>xxx</think>two", []),
    referenceRuns("<think>xxx</think>two", []),
  );
});

test("held-back characters are reclassified when the tag completes", () => {
  const segmented = createSegmentedAssistantText();
  segmented.appendText("hello<thi");
  assert.deepEqual(segmented.runs("hello<thi", []), [
    [{ type: "text", text: "hello<thi" }],
  ]);
  segmented.appendText("nk>secret");
  assert.deepEqual(segmented.runs("hello<think>secret", []), [
    [
      { type: "text", text: "hello" },
      { type: "reasoning", text: "secret" },
    ],
  ]);
});

test("the parts a run hands out are not shared with its retained state", () => {
  const segmented = createSegmentedAssistantText();
  segmented.appendText("one");
  const first = segmented.runs("one", []);
  first[0][0] = { type: "text", text: "clobbered" };
  segmented.appendText(" two");
  assert.deepEqual(segmented.runs("one two", []), [
    [{ type: "text", text: "one two" }],
  ]);
});

test("a tool call before any text leaves an empty run in front of it", () => {
  // A tool call before any text sits at cursor 0; the empty run must contribute no parts.
  const segmented = createSegmentedAssistantText();
  assert.deepEqual(segmented.runs("", [0]), referenceRuns("", [0]));
  segmented.appendText("after the tool");
  assert.deepEqual(
    segmented.runs("after the tool", [0]),
    referenceRuns("after the tool", [0]),
  );
  assert.deepEqual(segmented.runs("after the tool", [0]), [
    [],
    [{ type: "text", text: "after the tool" }],
  ]);
});

test("a think block split by a tool boundary parses as the adapter parses it", () => {
  // Odd but existing reference behaviour, which the incremental parse must reproduce.
  const segmented = createSegmentedAssistantText();
  segmented.appendText("<think>before");
  segmented.appendText("after</think> done");
  const text = "<think>beforeafter</think> done";
  assert.deepEqual(segmented.runs(text, [13]), referenceRuns(text, [13]));
  assert.deepEqual(segmented.runs(text, [13]), [
    [{ type: "reasoning", text: "before" }],
    [{ type: "text", text: "after</think> done" }],
  ]);
});
