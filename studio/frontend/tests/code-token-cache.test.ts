// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Shiki token cache has to stay bounded WHILE A FENCE IS STREAMING, not just eventually.
 *
 * `@streamdown/code` 1.1.1 keeps every tokenisation it ever made in a module-level Map with no
 * size cap and no clear, keyed on the source's length plus its first and last 100 characters. A
 * streamed fence reaches the highlighter once per refresh window with a longer prefix each time,
 * so one reply mints one full dual-theme tokenisation per window and keeps them all. Measured on
 * origin/main with a 32 KB python fence streamed over about five seconds: +10.8 MB of retained V8
 * heap per fence, still present after the reply is unmounted and the heap is force-collected,
 * against +0.64 MB for the same fence delivered in a single update.
 *
 * The replacement cache lives in src/components/assistant-ui/code-token-cache.ts. Each property
 * below is what stops one specific way of leaking, so each has its own test:
 *
 *   - prefix eviction, which is what collapses a streaming reply onto one entry
 *   - prefix eviction scoped to the group, so a different language does not lose its tokens
 *   - the character budget, which is the one that bounds MEGABYTES
 *   - the entry count cap, which bounds a thread of tiny fences
 *   - LRU ordering, so the fence on screen is not the one evicted
 */

import assert from "node:assert/strict";
import test from "node:test";
import { createTokenCache } from "../src/components/assistant-ui/code-token-cache.ts";

const GROUP = "python unsloth-light unsloth-dark";
const OTHER_GROUP = "typescript unsloth-light unsloth-dark";

// A streamed fence, as the highlighter sees it: the same source, one refresh window longer each
// time.
const prefixes = (source: string, windows: number): string[] =>
  Array.from({ length: windows }, (_unused, i) =>
    source.slice(0, Math.round((source.length * (i + 1)) / windows)),
  );

const roomy = () => createTokenCache<string>({ maxChars: 1_000_000, maxEntries: 1000 });

test("a stored tokenisation is served back for the same source", () => {
  const cache = roomy();
  cache.set(GROUP, "print(1)", "tokens-a");
  assert.equal(cache.get(GROUP, "print(1)"), "tokens-a");
  assert.equal(cache.get(GROUP, "print(2)"), null);
  assert.equal(cache.get(OTHER_GROUP, "print(1)"), null);
});

test("a streamed fence occupies one entry, not one per refresh window", () => {
  const cache = roomy();
  const source = "def f():\n    return 1\n".repeat(200);
  const windows = prefixes(source, 40);
  for (const window of windows) cache.set(GROUP, window, `tokens-${window.length}`);

  // The budget is nowhere near reached, so anything above 1 here is prefixes being kept.
  assert.equal(cache.stats().entries, 1);
  assert.equal(cache.stats().chars, source.length);
  assert.equal(cache.get(GROUP, source), `tokens-${source.length}`);
  for (const window of windows.slice(0, -1)) {
    assert.equal(cache.get(GROUP, window), null);
  }
});

test("two fences on screen at once, one extending the other, both stay cached", () => {
  // The live-lock this exists to stop. A miss returns null to the renderer, which schedules an
  // asynchronous tokenisation, whose callback re-renders and asks again. If storing the longer
  // fence evicted the shorter one AND storing the shorter one evicted the longer, two fences on
  // screen would miss forever and the page would never go idle. That hung a CI job for thirty
  // minutes before eviction was made one-directional.
  const cache = roomy();
  const short = "def f():\n    return 1\n";
  const long = `${short}def g():\n    return 2\n`;
  let misses = 0;
  for (let round = 0; round < 8; round += 1) {
    for (const code of [short, long]) {
      if (cache.get(GROUP, code) === null) {
        misses += 1;
        cache.set(GROUP, code, `tokens-${code.length}`);
      }
    }
  }
  // Three at most: each fence missing once, plus one re-fetch if the first store evicted it.
  assert.ok(misses <= 3, `the cache never settled: ${misses} misses over 8 rounds`);
  assert.equal(cache.stats().entries, 2);
  assert.equal(cache.get(GROUP, short), `tokens-${short.length}`);
  assert.equal(cache.get(GROUP, long), `tokens-${long.length}`);
});

test("prefix eviction does not reach across groups", () => {
  const cache = roomy();
  cache.set(OTHER_GROUP, "const a", "ts-tokens");
  cache.set(GROUP, "const a = 1", "py-tokens");
  assert.equal(cache.get(OTHER_GROUP, "const a"), "ts-tokens");
  assert.equal(cache.stats().entries, 2);
});

test("the character budget bounds what the cache holds", () => {
  const cache = createTokenCache<string>({ maxChars: 1000, maxEntries: 1000 });
  // Distinct first characters, so none of these is a prefix of another and only the budget can
  // evict them.
  for (let i = 0; i < 20; i += 1) {
    cache.set(GROUP, `${String.fromCharCode(97 + i)}${"x".repeat(299)}`, `t${i}`);
  }
  assert.ok(
    cache.stats().chars <= 1000,
    `held ${cache.stats().chars} characters against a 1000 budget`,
  );
  assert.ok(cache.stats().entries <= 4, `held ${cache.stats().entries} entries`);
  // The most recent write is the one that must survive.
  assert.equal(cache.get(GROUP, `t${"x".repeat(299)}`), "t19");
});

test("the entry count is capped for a thread of tiny fences", () => {
  const cache = createTokenCache<string>({ maxChars: 1_000_000, maxEntries: 8 });
  for (let i = 0; i < 50; i += 1) cache.set(GROUP, `${i}-tiny`, `t${i}`);
  assert.equal(cache.stats().entries, 8);
  assert.equal(cache.get(GROUP, "49-tiny"), "t49");
});

test("reading an entry protects it from the next eviction", () => {
  const cache = createTokenCache<string>({ maxChars: 1_000_000, maxEntries: 3 });
  cache.set(GROUP, "a-one", "t1");
  cache.set(GROUP, "b-two", "t2");
  cache.set(GROUP, "c-three", "t3");
  // Touch the oldest, so the next insert must drop "b-two" instead.
  assert.equal(cache.get(GROUP, "a-one"), "t1");
  cache.set(GROUP, "d-four", "t4");
  assert.equal(cache.get(GROUP, "a-one"), "t1");
  assert.equal(cache.get(GROUP, "b-two"), null);
});

test("the accounted character total tracks what is actually held", () => {
  const cache = roomy();
  cache.set(GROUP, "aaaa", "t1");
  cache.set(GROUP, "bbbbbb", "t2");
  assert.equal(cache.stats().chars, 10);
  // Re-storing the same source must not double count it.
  cache.set(GROUP, "aaaa", "t1b");
  assert.equal(cache.stats().chars, 10);
  assert.equal(cache.stats().entries, 2);
});
