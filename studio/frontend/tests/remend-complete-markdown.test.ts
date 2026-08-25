// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import remend from "remend";

/**
 * WHY THIS FILE EXISTS. `remend` is the incomplete-markdown repair Streamdown runs over a message
 * before parsing it, and Studio runs it on every SETTLED body: `markdown-text.tsx` passes
 * `parseIncompleteMarkdown={!incrementalRender}` and `incrementalRender` is null exactly when the
 * message is not streaming. So this dependency decides what a finished message LOOKS like, not
 * only what a half-written one looks like, and a version bump is a rendering change.
 *
 * These are RUN, not scraped. Every assertion below is a call into the resolved package, so the
 * file fails if the dependency is downgraded, if the override that forces one copy of it is
 * dropped, or if somebody switches the repair off to make a benchmark faster.
 *
 * The first test fails on remend 1.3.0, which is the point of the bump.
 */

test("a complete document with LaTeX subscripts comes back untouched", () => {
  // remend 1.3.0 does not recognise `\( ... \)` or `\[ ... \]` as math, so it counts the `_` of a
  // subscript as an unmatched emphasis marker and "completes" it by appending another one. The
  // document is COMPLETE, so there is nothing to complete, and Studio renders the extra character
  // literally at the end of the message.
  for (const complete of [
    String.raw`where \( \delta_{r} = 1 \) holds.`,
    String.raw`\[ \delta_{r} = 1 \]` + "\n",
    String.raw`where $ \delta_{r} = 1 $ holds.`,
    String.raw`where \( \delta_{r} = \beta_{k} \) holds.`,
  ]) {
    assert.equal(
      remend(complete, {}),
      complete,
      `remend rewrote a complete document: ${JSON.stringify(complete)}`,
    );
  }
});

test("ordinary complete markdown is returned unchanged", () => {
  for (const complete of [
    "see [the docs](https://example.com) for more",
    "this is **bold** text",
    "call `foo()` now",
    "```js\nconst a = 1;\n```\n",
    "the value $x + y$ holds",
    "an array literal like [1, 2, 3] in prose",
    "the pattern [^a-z] matches",
    "a snake_case identifier in prose",
    "```js\nconst r = /[^a-z]/;\n```\n",
  ]) {
    assert.equal(remend(complete, {}), complete, `remend rewrote ${JSON.stringify(complete)}`);
  }
});

test("a truncated stream is still repaired", () => {
  // The other half of the bump: the repair must still DO something. Without this, dropping the
  // dependency entirely, or passing `parseIncompleteMarkdown={false}`, would pass the tests above
  // and silently take the streaming repair with it.
  const repairs: [string, string][] = [
    ["see [the docs](https://exa", "]("],
    ["this is **bol", "**"],
    ["call `foo(", "`"],
  ];
  for (const [truncated, marker] of repairs) {
    const repaired = remend(truncated, {});
    assert.notEqual(
      repaired,
      truncated,
      `remend left a truncated ${marker} unrepaired: ${JSON.stringify(truncated)}`,
    );
    assert.ok(
      repaired.length >= truncated.length - marker.length,
      `the repair of ${JSON.stringify(truncated)} lost the document`,
    );
  }
});

/*
 * NO TIMING ASSERTION HERE, deliberately.
 *
 * The bump is a performance change as well as a rendering one: 1.3.0 answers "is this offset
 * inside a code block" by scanning from the start of the document every time it is asked, once per
 * candidate marker, so the repair grows faster than the length of a body. Timed on the frozen
 * studiobench corpus with no browser and no profiler attached, ten bodies cost 697.6 ms under
 * 1.3.0 and 111.0 ms under 1.3.1, and the largest body alone reads 2.95 / 10.93 / 158.40 / 539.92
 * ms at 13,347 / 26,694 / 53,388 / 106,776 characters against 3.30 / 11.21 / 29.39 / 69.42.
 *
 * A doubling-factor assertion over that shape was written, measured, and DELETED. On this host it
 * separates the two versions by 3.10 against 2.61, which is a 15 percent margin on a quantity that
 * only moves one way under load: it would fail on a busy runner for a reason that has nothing to
 * do with the property, and a test whose failure mode is "the machine was busy" gets re-run rather
 * than read. The performance claim belongs in the pull request beside the rungs it was measured
 * at, not in a unit test that cannot hold it.
 */
