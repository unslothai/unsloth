// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRAILING_PLACEHOLDER_WINDOW,
  stripTrailingTemplatePlaceholder,
} from "../src/features/chat/utils/trailing-template-placeholder.ts";

/** The old unbounded scan; the bounded one must agree on anything that fits the window. */
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

/** `seed * 1103515245` overflows 2^53, leaving constant low bits, so use mulberry32. */
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
  // Without this, an identity implementation would also pass.
  assert.equal(
    stripped > 2_000,
    true,
    `only ${stripped} of ${checked} random inputs had anything to strip`,
  );
});

test("bounded strip matches the unbounded pattern behind a long reply", () => {
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
  assert.equal(stripUnbounded(input), "answer");
  const stripped = stripTrailingTemplatePlaceholder(input);
  assert.equal(stripped, input, "an oversized fragment must be left alone");
  assert.equal(input.startsWith(stripped), true);
});

test("a whitespace run past the window cannot restart the whole-buffer scan", () => {
  const input = `answer \${x}${" ".repeat(TRAILING_PLACEHOLDER_WINDOW + 100)}`;
  const stripped = stripTrailingTemplatePlaceholder(input);
  assert.equal(input.startsWith(stripped), true);
  assert.equal(stripped, input, "an oversized whitespace run must be left alone");
});

test("a narrow window only ever strips less than the unbounded pattern", () => {
  // The scan may keep text the unbounded pattern removes, never remove text it keeps.
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
  assert.equal(
    differed > 1_000,
    true,
    `only ${differed} inputs were stripped at all across the narrow windows`,
  );
});

test("an out-of-window opener strips the nested placeholder, not the outer one", () => {
  // The outer `${` sits past the window, so stripping only the inner `${nested}` is correct.
  const input = `answer \${${"a".repeat(4196)}\${nested}`;
  const bounded = stripTrailingTemplatePlaceholder(input);
  const unbounded = stripUnbounded(input);

  assert.equal(
    bounded,
    input.slice(0, input.length - "${nested}".length),
    "the inner placeholder, and only it, should be removed",
  );
  assert.equal(input.length - unbounded.length, 4208);
  assert.equal(input.length - bounded.length, 9);
  assert.equal(
    input.startsWith(bounded),
    true,
    "the result must stay a prefix of the input",
  );
});
