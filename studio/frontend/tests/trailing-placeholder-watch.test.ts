// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createTrailingPlaceholderWatch,
  stripTrailingTemplatePlaceholder,
} from "../src/features/chat/utils/trailing-template-placeholder.ts";

/**
 * The one property that matters: the watch never says no when the strip would
 * cut. Saying yes when it would not is allowed, and costs one scan.
 *
 * Everything else in the adapter is downstream of that, because the strip only
 * runs on an arrival the watch admits.
 */
function assertSound(text: string, candidate: boolean, label: string): void {
  const stripped = stripTrailingTemplatePlaceholder(text);
  if (stripped.length !== text.length) {
    assert.equal(
      candidate,
      true,
      `${label}: the strip cut ${JSON.stringify(text)} down to ${JSON.stringify(stripped)}, but the watch said there was nothing to do`,
    );
  }
}

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

test("the watch admits every fragment the strip would cut", () => {
  const alphabet = [
    "$",
    "{",
    "}",
    "${",
    "${x}",
    "a",
    " ",
    "\n",
    "x}",
    "${a",
    "b}c",
    "}}",
    "${}",
  ];
  const random = mulberry32(20260817);
  let cuts = 0;
  let admitted = 0;
  let states = 0;
  for (let seed = 0; seed < 2000; seed += 1) {
    const watch = createTrailingPlaceholderWatch();
    let text = "";
    for (let step = 0; step < 20; step += 1) {
      const delta = alphabet[random() % alphabet.length];
      text += delta;
      watch.append(delta);
      const candidate = watch.isCandidate();
      states += 1;
      if (candidate) admitted += 1;
      assertSound(text, candidate, `seed ${seed} step ${step}`);
      // What the adapter does with a yes: run the strip, and if it cut, tell
      // the watch what is left.
      if (candidate) {
        const stripped = stripTrailingTemplatePlaceholder(text);
        if (stripped.length !== text.length) {
          cuts += 1;
          text = stripped;
          watch.retract(text);
        }
      }
    }
  }
  assert.equal(cuts > 500, true, `only ${cuts} strips exercised`);
  assert.equal(states > 10_000, true, `only ${states} states exercised`);
  // The point of the watch is that most arrivals never reach the strip. If
  // this ever admitted everything it would be sound and useless.
  assert.equal(
    admitted < states / 2,
    true,
    `the watch admitted ${admitted} of ${states} states; it is not rejecting anything`,
  );
});

test("the watch admits every fragment the strip would cut, on prose", () => {
  // Brace-heavy prose, which is what a code fence looks like: plenty of lines
  // ending in `}` with no `${` in front of them.
  const lines = [
    "function step(input) {\n",
    "  return { ...input };\n",
    "}\n",
    "\n",
    "text and more text ",
    "a `${x}` in a sentence ",
    "closing }\n",
  ];
  const random = mulberry32(7);
  let admitted = 0;
  let states = 0;
  for (let seed = 0; seed < 200; seed += 1) {
    const watch = createTrailingPlaceholderWatch();
    let text = "";
    for (let step = 0; step < 40; step += 1) {
      const piece = lines[random() % lines.length];
      // Split each piece into small arrivals, the way tokens land.
      for (let at = 0; at < piece.length; at += 3) {
        const delta = piece.slice(at, at + 3);
        text += delta;
        watch.append(delta);
        const candidate = watch.isCandidate();
        states += 1;
        if (candidate) admitted += 1;
        assertSound(text, candidate, `seed ${seed} step ${step}`);
        if (candidate) {
          const stripped = stripTrailingTemplatePlaceholder(text);
          if (stripped.length !== text.length) {
            text = stripped;
            watch.retract(text);
          }
        }
      }
    }
  }
  assert.equal(states > 10_000, true, `only ${states} states exercised`);
  assert.equal(
    admitted * 20 < states,
    true,
    `the watch admitted ${admitted} of ${states} prose states; a code fence must not keep waking the strip`,
  );
});

test("a fragment split across arrivals is still admitted", () => {
  for (const pieces of [
    ["a ", "$", "{", "x", "}"],
    ["a ", "${", "x}"],
    ["a ", "${x", "}"],
    ["a ", "${x}"],
    ["a ", "$", "{x}", "  "],
  ]) {
    const watch = createTrailingPlaceholderWatch();
    let text = "";
    for (const piece of pieces) {
      text += piece;
      watch.append(piece);
    }
    assert.equal(
      watch.isCandidate(),
      true,
      `${JSON.stringify(pieces)} was not admitted, so the strip would never run on it`,
    );
    assert.equal(stripTrailingTemplatePlaceholder(text), "a");
  }
});

test("ordinary prose is rejected without the buffer being read", () => {
  const watch = createTrailingPlaceholderWatch();
  for (const delta of ["the answer", " is ", "forty two", ".\n\n", "done"]) {
    watch.append(delta);
    assert.equal(watch.isCandidate(), false);
  }
});
