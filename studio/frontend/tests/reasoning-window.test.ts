// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  REASONING_WINDOW_CHARS,
  alignWindowStart,
  isOutsideFence,
  nextReasoningWindowStart,
} from "../src/features/chat/utils/reasoning-window.ts";

const para = (n: number) => `${"word ".repeat(40)}para${n}`;

/** A body of `units` paragraphs with a closed fence after every third one. */
function body(units: number): string {
  const out: string[] = [];
  for (let i = 0; i < units; i += 1) {
    out.push(para(i));
    if (i % 3 === 2) {
      out.push(["```python", `x = ${i}`, "y = x * 2", "```"].join("\n"));
    }
  }
  return out.join("\n\n");
}

test("inline triple backticks do not open a fence", () => {
  const text = "a ``` inline b\n\nc";
  assert.equal(isOutsideFence(text, text.length), true);
});

test("offsets between an opening and a closing fence are inside it", () => {
  const text = "intro\n\n```py\nx = 1\n```\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("x = 1")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("an unclosed fence is still open at the end of the text", () => {
  const text = "intro\n\n```py\nx = 1\n";
  assert.equal(isOutsideFence(text, text.length), false);
});

test("the window start lands on a block boundary at or after the target", () => {
  const text = body(30);
  const start = alignWindowStart(text, 400);
  assert.ok(start >= 400);
  assert.equal(text.slice(start - 2, start), "\n\n");
});

test("the window start never cuts into a fence", () => {
  const text = body(60);
  for (let target = 0; target < text.length; target += 137) {
    assert.equal(isOutsideFence(text, alignWindowStart(text, target)), true);
  }
});

test("with no safe boundary the whole body renders rather than a cut fence", () => {
  // Showing everything is correct and merely slow; cutting into a fence is wrong.
  const text = "intro\n\n```py\nno closing fence here\nmore\nmore\n";
  assert.equal(alignWindowStart(text, 10), 0);
});

test("the start does not move until the body grows a whole slack past the window", () => {
  const text = body(400).slice(0, Math.floor(REASONING_WINDOW_CHARS * 1.4));
  assert.equal(nextReasoningWindowStart(text, 0), 0);
});

test("the start moves once the body is far enough past the window", () => {
  const text = body(2000);
  assert.ok(text.length > REASONING_WINDOW_CHARS * 2);
  assert.ok(nextReasoningWindowStart(text, 0) > 0);
});

test("the start is monotone, so the body never grows backwards mid stream", () => {
  // A start that moved back would hand the renderer the string it just rendered plus a prefix.
  const text = body(4000);
  let start = 0;
  for (let end = 2000; end <= text.length; end += 2000) {
    const next = nextReasoningWindowStart(text.slice(0, end), start);
    assert.ok(next >= start);
    start = next;
  }
});

test("the rendered body stays bounded as the thinking text grows without limit", () => {
  const text = body(6000);
  let start = 0;
  let widest = 0;
  for (let end = 2000; end <= text.length; end += 2000) {
    start = nextReasoningWindowStart(text.slice(0, end), start);
    widest = Math.max(widest, end - start);
  }
  assert.ok(text.length > REASONING_WINDOW_CHARS * 4);
  assert.ok(widest < REASONING_WINDOW_CHARS * 2.2);
});

test("the start moves rarely, because each move remounts the rendered body", () => {
  const text = body(6000);
  let start = 0;
  let moves = 0;
  const chunk = 24;
  for (let end = chunk; end <= text.length; end += chunk) {
    const next = nextReasoningWindowStart(text.slice(0, end), start);
    if (next !== start) moves += 1;
    start = next;
  }
  // Order tens of moves over thousands of chunks, not one per chunk. Moving per chunk would drop
  // the incremental Markdown cache's retained blocks every time, which is worse than the problem.
  assert.ok(moves < Math.floor(text.length / chunk) / 100);
});

// ── what the reader can reach ───────────────────────────────────────
//
// The window is only ever a SUFFIX of the body, and the component restores the whole body the
// moment the reader scrolls back or the round ends. So reachability reduces to one property
// here: whatever is mounted is a suffix of the real text, never a rewrite of it. Nothing is
// paraphrased, reordered or dropped from the middle, so restoring is always just "show all of
// it". The restore itself is a component behaviour and is measured by
// tests/studio/probe_reasoning_window.py rather than asserted here.

test("what is mounted is always a suffix of the real text", () => {
  const text = body(4000);
  let start = 0;
  for (let end = 2000; end <= text.length; end += 2000) {
    const slice = text.slice(0, end);
    start = nextReasoningWindowStart(slice, start);
    assert.ok(slice.endsWith(slice.slice(start)));
  }
});

test("the window never begins inside a fence, at any length", () => {
  const text = body(4000);
  let start = 0;
  for (let end = 1000; end <= text.length; end += 1000) {
    const slice = text.slice(0, end);
    start = nextReasoningWindowStart(slice, start);
    // A start inside a fence would make the renderer read the closing marker as an opening one
    // and treat the rest of the thinking block as code.
    assert.equal(isOutsideFence(slice, start), true);
  }
});
