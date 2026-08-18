// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createSegmentedAssistantText } from "../src/features/chat/utils/incremental-assistant-content.ts";
import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";

/**
 * What the adapter did before: cut the whole reply at the tool cursors and
 * parse each run from scratch. The incremental parse has to agree with this
 * for every state a stream passes through, which is what the sweep below
 * checks, so this is the oracle and must stay a plain rewrite of the original.
 */
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

// ---------------------------------------------------------------- random ---

function makeRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

// Pieces chosen so a tag can be split at every offset, and so text that only
// looks like a tag ("<thi", "</thin", "<<think>") is generated often.
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

// ------------------------------------------------------------------ tests ---

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
      // A tool call lands at the end of the reply as it stands, which is the
      // only place the adapter puts one.
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
        // The trailing placeholder strip is the only thing that shortens the
        // buffer, and it always takes a suffix.
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
  // What `mergeContinuation` does to an external continuation. The cache is
  // built with the fast path off for that case, so it must still be right.
  const segmented = createSegmentedAssistantText({ trustAppends: false });
  segmented.appendText("<think>one</think>two");
  assert.deepEqual(
    segmented.runs("<think>ONE</think>two", []),
    referenceRuns("<think>ONE</think>two", []),
  );
  // Same length, different characters: the length check alone cannot see this,
  // which is why that path does not rely on it.
  assert.deepEqual(
    segmented.runs("<think>xxx</think>two", []),
    referenceRuns("<think>xxx</think>two", []),
  );
});

test("held-back characters are reclassified when the tag completes", () => {
  const segmented = createSegmentedAssistantText();
  segmented.appendText("hello<thi");
  // Nothing has said this is a tag yet, so it reads as text.
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
  // The adapter gives a tool part the reply's length as its cursor, so a tool
  // call that arrives before the model has written anything sits at 0. The run
  // in front of it is empty and must contribute no parts at all, not an empty
  // text part.
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
  // The reference cuts the text at the cursor and parses each side on its own,
  // so the opening tag on the near side leaves an unclosed reasoning part and
  // the far side starts fresh, as text. That is the existing behaviour, odd as
  // it looks, and the incremental parse has to reproduce it rather than fix it.
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
