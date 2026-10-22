// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * hasPrefix replaces startsWith purely for speed; any behavioural difference silently
 * breaks cache resets. Surrogate pairs pin cuts inside astral characters.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { hasPrefix } from "../src/components/assistant-ui/streaming-render-schedule.ts";

import { readSrc } from "./helpers/kit.ts";

const CORPUS = [
  "",
  "a",
  "ab",
  "abc",
  "\n",
  "  ",
  "a\nb\nc",
  "$5 and \\(x\\)",
  "```ts\nconst a = 1;\n```",
  "é",
  "\u{1F600}",
  "x\u{1F600}y",
  "\u{1F600}\u{1F600}",
  "aaaaaaaaaa",
  "aaaaaaaaab",
  "A",
  "aB",
  "AAAAAAAAAA",
];

test("hasPrefix agrees with startsWith on a fixed corpus", () => {
  for (const a of CORPUS) {
    for (const b of CORPUS) {
      assert.equal(
        hasPrefix(a, b),
        a.startsWith(b),
        `disagreed on a=${JSON.stringify(a)} b=${JSON.stringify(b)}`,
      );
    }
  }
});

test("hasPrefix agrees with startsWith on split points of astral text", () => {
  const s = "a\u{1F600}b\u{1F601}c";
  for (let i = 0; i <= s.length; i += 1) {
    const b = s.slice(0, i);
    assert.equal(hasPrefix(s, b), s.startsWith(b), `cut at ${i}`);
    assert.equal(hasPrefix(b, s), b.startsWith(s), `reversed cut at ${i}`);
  }
});

test("hasPrefix agrees with startsWith under randomised growth", () => {
  // mulberry32: low bits are usable, unlike multiply-and-mask whose product exceeds 2^53.
  let seed = 0x9e3779b9;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const alphabet = [..."ab\n $\\`", "\u{1F600}", "́"];

  let agreed = 0;
  let bothTrue = 0;
  for (let trial = 0; trial < 20_000; trial += 1) {
    const len = Math.floor(rand() * 12);
    let a = "";
    for (let i = 0; i < len; i += 1) {
      a += alphabet[Math.floor(rand() * alphabet.length)];
    }
    const b =
      rand() < 0.5
        ? a.slice(0, Math.floor(rand() * (a.length + 1)))
        : (() => {
            let x = "";
            const n = Math.floor(rand() * 12);
            for (let i = 0; i < n; i += 1) {
              x += alphabet[Math.floor(rand() * alphabet.length)];
            }
            return x;
          })();
    const expected = a.startsWith(b);
    assert.equal(
      hasPrefix(a, b),
      expected,
      `a=${JSON.stringify(a)} b=${JSON.stringify(b)}`,
    );
    agreed += 1;
    if (expected) bothTrue += 1;
  }
  assert.equal(agreed, 20_000);
  assert.ok(
    bothTrue > 4_000,
    `only ${bothTrue} of 20,000 cases were real prefixes; the true branch is barely covered`,
  );
});

// hasPrefix and startsWith agree, so only a source check catches a revert.
test("the incremental cache tests prefixes without scanning the reply", () => {
  const source = readSrc(
    "components/assistant-ui/streaming-render-schedule.ts",
  );
  const bare = [...source.matchAll(/\.startsWith\(([^)]*)\)/g)].filter(
    (match) => !match[1].includes(","),
  );
  assert.deepEqual(
    bare.map((match) => match[0]),
    [],
    "a one-argument startsWith on this path scans the whole reply; use hasPrefix",
  );
  assert.ok(
    source.split("hasPrefix(").length - 1 >= 3,
    "hasPrefix should be the spelling every prefix test on this path uses",
  );
});
