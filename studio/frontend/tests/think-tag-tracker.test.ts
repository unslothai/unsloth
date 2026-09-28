// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createThinkTagTracker,
  hasUnclosedThinkTag,
} from "../src/features/chat/utils/parse-assistant-content.ts";

/**
 * The tracker takes deltas, because reading the accumulated buffer at all
 * costs O(reply) per arrival. The cases below are written against the buffer,
 * so this maps a buffer to the calls the adapter makes for it: an arrival
 * appends what it added, and the trailing-fragment strip retracts to what it
 * left behind.
 */
function bufferTracker(): { update(text: string): boolean } {
  const tracker = createThinkTagTracker();
  let seen = "";
  return {
    update(text: string): boolean {
      if (text.length >= seen.length) {
        tracker.append(text.slice(seen.length));
      } else {
        tracker.retract(text);
      }
      seen = text;
      return tracker.endsInsideThink();
    },
  };
}

/**
 * Feed `chunks` in order and assert the tracker agrees with a full rescan
 * after every one of them. `hasUnclosedThinkTag` is the reference: the
 * tracker exists only to give the same answer without rereading the buffer.
 */
function assertAgreesOnStream(
  chunks: string[],
  label: string,
): { unclosed: number; closed: number } {
  const tracker = bufferTracker();
  let text = "";
  let unclosed = 0;
  let closed = 0;
  for (const [index, chunk] of chunks.entries()) {
    text += chunk;
    const answer = hasUnclosedThinkTag(text);
    assert.equal(
      tracker.update(text),
      answer,
      `${label}: disagreed after chunk ${index} (${JSON.stringify(chunk)}), buffer ${JSON.stringify(text)}`,
    );
    if (answer) unclosed += 1;
    else closed += 1;
  }
  return { unclosed, closed };
}

/** Every way of cutting `tag` into `parts` non-empty pieces. */
function splits(tag: string, parts: number): string[][] {
  if (parts === 1) return [[tag]];
  const out: string[][] = [];
  for (let cut = 1; cut <= tag.length - (parts - 1); cut += 1) {
    for (const rest of splits(tag.slice(cut), parts - 1)) {
      out.push([tag.slice(0, cut), ...rest]);
    }
  }
  return out;
}

test("an opening tag split across two, three and four arrivals is seen", () => {
  for (const parts of [2, 3, 4]) {
    for (const pieces of splits("<think>", parts)) {
      assertAgreesOnStream(
        ["before ", ...pieces, "reasoning"],
        `<think> in ${parts} parts as ${JSON.stringify(pieces)}`,
      );
    }
  }
});

test("a closing tag split across two, three and four arrivals is seen", () => {
  for (const parts of [2, 3, 4]) {
    for (const pieces of splits("</think>", parts)) {
      assertAgreesOnStream(
        ["<think>reasoning", ...pieces, " answer"],
        `</think> in ${parts} parts as ${JSON.stringify(pieces)}`,
      );
    }
  }
});

test("tags delivered one character per arrival are seen", () => {
  assertAgreesOnStream(
    [..."prefix <think>thought</think> answer"],
    "one character per arrival",
  );
});

test("a tag split across arrivals that carry nothing else is seen", () => {
  // Worst case for too short an overlap: one tag character per arrival, with
  // empty arrivals in between.
  assertAgreesOnStream(
    ["a", "<", "", "t", "", "h", "i", "", "n", "k", "", ">", "b"],
    "empty arrivals between tag characters",
  );
  assertAgreesOnStream(
    ["<think>a", "<", "/", "t", "h", "i", "n", "k", ">", "b"],
    "</think> one character at a time",
  );
});

test("repeated reasoning blocks track the last tag, not the first", () => {
  assertAgreesOnStream(
    [
      "<think>",
      "one",
      "</think>",
      "answer ",
      "<think>",
      "two",
      "</think>",
      "more ",
      "<think>",
      "three",
    ],
    "three reasoning blocks",
  );
});

test("a near miss is not mistaken for a tag", () => {
  assertAgreesOnStream(
    [..."<thin> <think > </thin> <thinks> <think>x</think >"],
    "near misses",
  );
});

test("a stream seeded with a continuation partial is tracked", () => {
  const tracker = bufferTracker();
  let text = "<think>resumed reasoning";
  assert.equal(tracker.update(text), hasUnclosedThinkTag(text));
  text += "</think> answer";
  assert.equal(tracker.update(text), hasUnclosedThinkTag(text));
});

test("a suffix removed from the buffer is accounted for", () => {
  // The adapter strips a trailing ${...} fragment after appending, so its
  // buffer can be shorter than the one from the arrival before.
  const cases: Array<{ steps: string[]; label: string }> = [
    { steps: ["<think>a", "<think>a ${x}", "<think>a"], label: "plain strip" },
    {
      steps: ["<think>a</think>", "<think>a</think> ${x}", "<think>a</think>"],
      label: "strip after a close",
    },
    {
      // The suffix takes the closing tag with it, so the buffer is unclosed
      // again and the tracker has to find the earlier close.
      steps: ["<think>a</think>b</think>", "<think>a</think>b"],
      label: "strip removes the last close tag",
    },
    {
      // The removed suffix takes the opening tag with it.
      steps: ["<think>a</think>b<think>c", "<think>a</think>b"],
      label: "strip removes the last open tag",
    },
    {
      // Removed back past a tag that was never scanned on its own.
      steps: ["a<think>b</think>c", "a"],
      label: "strip removes both tags",
    },
  ];
  for (const { steps, label } of cases) {
    const tracker = bufferTracker();
    for (const [index, text] of steps.entries()) {
      assert.equal(
        tracker.update(text),
        hasUnclosedThinkTag(text),
        `${label}: disagreed at step ${index} on ${JSON.stringify(text)}`,
      );
    }
  }
});

test("append then strip within one arrival is tracked", () => {
  // What the adapter actually does: append the delta, strip a trailing
  // fragment, then ask once. Only the post-strip buffer is ever seen.
  const tracker = bufferTracker();
  const deltas = [
    "<think>",
    "reasoning ",
    "</think>",
    " answer ${x}",
    " continues",
    "<think>more ${y}",
    "</think>",
  ];
  const strip = (text: string): string => text.replace(/\s*\$\{[^}]*\}\s*$/, "");
  let text = "";
  for (const [index, delta] of deltas.entries()) {
    text = strip(text + delta);
    assert.equal(
      tracker.update(text),
      hasUnclosedThinkTag(text),
      `disagreed after delta ${index} on ${JSON.stringify(text)}`,
    );
  }
});

/**
 * Deterministic pseudo-random generator. A plain `seed * 1103515245` loop is not
 * usable here: the product runs past 2^53, so the sampled low bits come out
 * constant and half of any alphabet is never drawn.
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

test("tracker agrees with a full rescan on random streams", () => {
  const alphabet = [
    "<",
    ">",
    "/",
    "t",
    "h",
    "i",
    "n",
    "k",
    "<think>",
    "</think>",
    "x",
    " ",
    "<thin",
    "k>",
    "</thi",
    "nk>",
  ];
  const next = mulberry32(24680);
  let unclosed = 0;
  let closed = 0;
  for (let run = 0; run < 2_000; run += 1) {
    const chunks: string[] = [];
    const count = 1 + (next() % 24);
    for (let i = 0; i < count; i += 1) {
      chunks.push(alphabet[next() % alphabet.length]);
    }
    const seen = assertAgreesOnStream(chunks, `random run ${run}`);
    unclosed += seen.unclosed;
    closed += seen.closed;
  }
  // Both answers have to occur, or the comparison above would hold for a
  // constant tracker.
  assert.equal(unclosed > 1_000, true, `only ${unclosed} unclosed states seen`);
  assert.equal(closed > 1_000, true, `only ${closed} closed states seen`);
});

test("tracker agrees with a full rescan on random streams with strips", () => {
  const alphabet = ["<think>", "</think>", "<", "/", "t", "h", "i", "n", "k", ">", "x"];
  const next = mulberry32(13579);
  let unclosed = 0;
  let closed = 0;
  let strips = 0;
  for (let run = 0; run < 2_000; run += 1) {
    const tracker = bufferTracker();
    let text = "";
    const count = 1 + (next() % 16);
    for (let i = 0; i < count; i += 1) {
      text += alphabet[next() % alphabet.length];
      // Cut a suffix off, as the trailing-fragment strip does.
      if (next() % 3 === 0) {
        const shorter = text.slice(0, Math.max(0, text.length - (next() % 12)));
        if (shorter.length < text.length) strips += 1;
        text = shorter;
      }
      const answer = hasUnclosedThinkTag(text);
      assert.equal(
        tracker.update(text),
        answer,
        `random strip run ${run}: disagreed at step ${i} on ${JSON.stringify(text)}`,
      );
      if (answer) unclosed += 1;
      else closed += 1;
    }
  }
  assert.equal(strips > 1_000, true, `only ${strips} suffixes were removed`);
  assert.equal(unclosed > 500, true, `only ${unclosed} unclosed states seen`);
  assert.equal(closed > 500, true, `only ${closed} closed states seen`);
});

test("the adapter's own order, append the delta then retract the strip, is tracked", () => {
  // The wrapper above only ever sees the buffer a strip left behind, so it
  // hides the two-call sequence the adapter really makes. This drives the
  // tracker directly: every delta is appended whole, and only then is the
  // fragment taken back off.
  const random = mulberry32(20260817);
  const alphabet = ["<think>", "</think>", "x", " ", "<thi", "nk>", "}", "${a}"];
  for (let seed = 0; seed < 400; seed += 1) {
    const tracker = createThinkTagTracker();
    let text = "";
    for (let step = 0; step < 24; step += 1) {
      const delta = alphabet[random() % alphabet.length];
      text += delta;
      tracker.append(delta);
      const stripped = text.replace(/\s*\$\{[^}]*\}\s*$/, "");
      if (stripped.length !== text.length) {
        text = stripped;
        tracker.retract(text);
      }
      assert.equal(
        tracker.endsInsideThink(),
        hasUnclosedThinkTag(text),
        `seed ${seed} step ${step}: disagreed on ${JSON.stringify(text)}`,
      );
    }
  }
});
