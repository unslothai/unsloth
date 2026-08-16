// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRAILING_PLACEHOLDER_WINDOW,
  stripTrailingTemplatePlaceholder,
} from "../src/features/chat/utils/trailing-template-placeholder.ts";

/**
 * What the adapter used to run over the whole buffer on every arrival. The
 * bounded scan has to agree with it character for character on anything that
 * fits in the window, or the strip has changed what reaches the bubble.
 */
const UNBOUNDED = /\s*\$\{[^}]*\}\s*$/;
function stripUnbounded(text: string): string {
  return text.replace(UNBOUNDED, "");
}

test("bounded strip matches the unbounded pattern on fixed cases", () => {
  const cases = [
    "",
    " ",
    "no placeholder here",
    "trailing brace }",
    "${}",
    "${a}",
    "hello ${name}",
    "hello ${name}   ",
    "hello ${name}\n\n",
    "hello\n\n${name}\n",
    "hello ${name} world",
    "hello ${name} world}",
    "a}${b}",
    "${x${y}",
    "prefix ${a}${b}",
    "prefix ${a} ${b}",
    "${a\nb}",
    "unterminated ${a",
    "closed but empty ${ }",
    "dollar $ alone }",
    "brace {a}",
    "$}{",
    "text ${x} ",
    "```ts\nconst a = { b: 1 };\n```",
    "json {\"a\": {\"b\": 1}}",
    "json {\"a\": {\"b\": 1}} ${c}",
  ];
  for (const input of cases) {
    assert.equal(
      stripTrailingTemplatePlaceholder(input),
      stripUnbounded(input),
      `disagreed on ${JSON.stringify(input)}`,
    );
  }
});

/**
 * Deterministic pseudo-random generator. A plain `seed * 1103515245` loop is
 * not usable here: the product runs past 2^53, so the low bits it is sampled
 * on come out constant and half of any alphabet is never drawn, which is how
 * the first draft of this test came to compare 20,000 strings that the pattern
 * could not match.
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

/**
 * Random strings over the characters that matter here, half of them ending in
 * something the pattern has a real chance of matching.
 */
function* randomInputs(count: number, seed = 20260816): Generator<string> {
  const alphabet = ["$", "{", "}", " ", "\n", "\t", "a", "b", "x", "${", "a}"];
  const tails = [
    "",
    "${x}",
    " ${x}",
    "${}",
    "\n${a b}\n",
    "${a",
    "}${x}",
    "a}${b}",
    "${x${y}",
    " ",
    "}",
    "${x} ",
  ];
  const next = mulberry32(seed);
  for (let i = 0; i < count; i += 1) {
    const length = next() % 40;
    let out = "";
    for (let j = 0; j < length; j += 1) {
      out += alphabet[next() % alphabet.length];
    }
    yield out + tails[next() % tails.length];
  }
}

test("bounded strip matches the unbounded pattern on random inputs", () => {
  let checked = 0;
  let stripped = 0;
  for (const input of randomInputs(20_000)) {
    const reference = stripUnbounded(input);
    assert.equal(
      stripTrailingTemplatePlaceholder(input),
      reference,
      `disagreed on ${JSON.stringify(input)}`,
    );
    checked += 1;
    if (reference !== input) stripped += 1;
  }
  assert.equal(checked, 20_000);
  // Without this the comparison above holds for an implementation that returns
  // its argument, which is exactly what the first draft of this test did.
  assert.equal(
    stripped > 2_000,
    true,
    `only ${stripped} of ${checked} random inputs had anything to strip`,
  );
});

test("bounded strip matches the unbounded pattern behind a long reply", () => {
  // The window is what makes the scan cheap; a placeholder at the end of a
  // long reply still has to be stripped exactly, whatever precedes it.
  const prefixes = [
    "word ".repeat(20_000),
    "}".repeat(5_000),
    "{".repeat(5_000),
    `${"a".repeat(50_000)}}`,
    "$".repeat(5_000),
  ];
  for (const prefix of prefixes) {
    for (const suffix of ["", " ${x}", "\n${x.y}\n", " ${}", " ${a"]) {
      const input = prefix + suffix;
      assert.equal(
        stripTrailingTemplatePlaceholder(input),
        stripUnbounded(input),
        `disagreed on ${JSON.stringify(suffix)} after ${prefix.length} chars`,
      );
    }
  }
});

test("a placeholder past the window is left whole, never cut in half", () => {
  const inner = "a".repeat(TRAILING_PLACEHOLDER_WINDOW + 100);
  const input = `answer \${${inner}}`;
  // The unbounded pattern would take it; this is the documented bound.
  assert.equal(stripUnbounded(input), "answer");
  const stripped = stripTrailingTemplatePlaceholder(input);
  assert.equal(stripped, input, "an oversized fragment must be left alone");
  // What must never happen: a result that keeps half a fragment, or that is
  // not a prefix of the input at all.
  assert.equal(input.startsWith(stripped), true);
});

test("a whitespace run past the window cannot restart the whole-buffer scan", () => {
  const input = `answer \${x}${" ".repeat(TRAILING_PLACEHOLDER_WINDOW + 100)}`;
  const stripped = stripTrailingTemplatePlaceholder(input);
  assert.equal(input.startsWith(stripped), true);
  assert.equal(stripped, input, "an oversized whitespace run must be left alone");
});

test("a narrow window only ever strips less than the unbounded pattern", () => {
  // Whatever the window, the scan may keep text the unbounded pattern would
  // have removed, but it may never remove text the unbounded pattern kept.
  let differed = 0;
  for (const window of [1, 2, 4, 8, 16]) {
    for (const input of randomInputs(4_000)) {
      const bounded = stripTrailingTemplatePlaceholder(input, window);
      const unbounded = stripUnbounded(input);
      assert.equal(
        input.startsWith(bounded),
        true,
        `not a prefix for window ${window} on ${JSON.stringify(input)}`,
      );
      assert.equal(
        bounded.length >= unbounded.length,
        true,
        `stripped more than the pattern for window ${window} on ${JSON.stringify(input)}`,
      );
      assert.equal(
        bounded.startsWith(unbounded),
        true,
        `diverged before the strip for window ${window} on ${JSON.stringify(input)}`,
      );
      if (bounded !== input) differed += 1;
    }
  }
  // The property is trivially true of a scan that never strips anything, so
  // the inputs have to reach the strip for this to mean something.
  assert.equal(
    differed > 1_000,
    true,
    `only ${differed} inputs were stripped at all across the narrow windows`,
  );
});
