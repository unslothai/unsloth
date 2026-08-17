// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  REASONING_WINDOW_CHARS,
  alignWindowStart,
  isOutsideFence,
  linkDefinitionsBefore,
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

// A blank line inside a fence is legal CommonMark and ordinary in real code, so it is a block
// boundary that lands in the middle of a code block. Counting only bare ``` at column zero says
// such a boundary is safe for the two fence shapes below, and the slice then starts mid-code with
// the CLOSING marker ahead of it, which the renderer reads as an opening one and highlights the
// rest of the thinking block as code.

test("a fence indented inside a list item is still a fence", () => {
  const text = "- plan:\n\n  ```python\n  def f():\n      return 1\n\n  def g():\n      pass\n  ```\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("def g")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

// The two below assert on the SLICE rather than on isOutsideFence, so they cannot pass merely
// because the fence test agrees with itself.

test("the window start never lands inside an indented fence", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}- plan:\n\n  \`\`\`python\n  def f():\n      return 1\n\n  def g():\n      pass\n  \`\`\`\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("def f"));
  // Starting at "  def g" leaves the fence's closing marker ahead of the reader, which the
  // renderer reads as an opening one.
  assert.ok(!text.slice(start).startsWith("  def g"), text.slice(start, start + 20));
});

test("a tilde fence is a fence", () => {
  const text = "intro\n\n~~~python\nx = 1\n\ny = 2\n~~~\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("y = 2")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("the window start never lands inside a tilde fence", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}~~~python\ndef f():\n    return 1\n\ndef g():\n    pass\n~~~\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("def f"));
  assert.ok(!text.slice(start).startsWith("def g"), text.slice(start, start + 20));
});

test("a shorter run of the fence character does not close a longer fence", () => {
  // Four backticks are used precisely so the sample can contain three.
  const text = "intro\n\n````\n```\ninner\n```\n\nstill inner\n````\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("inner")), false);
  assert.equal(isOutsideFence(text, text.indexOf("still inner")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("an info string cannot close a fence", () => {
  const text = "intro\n\n```py\nx = 1\n\n```python\n\nstill code\n```\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("still code")), false);
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

// ── the shapes the first fence guard still missed ───────────────────

test("a fence opened on the list marker line is a fence", () => {
  const text = "- ```js\n  let x = 1;\n\n  let y = 2;\n  ```\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("let y")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("the window start never lands inside a list-marker fence", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}- \`\`\`js\n  let x = 1;\n\n  let y = 2;\n  \`\`\`\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("let x"));
  assert.ok(!text.slice(start).startsWith("  let y"), text.slice(start, start + 20));
});

test("a fence inside a blockquote is a fence", () => {
  const text = "> ```py\n> x = 1\n\n> y = 2\n> ```\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("y = 2")), false);
});

test("a display math block is not a safe place to cut", () => {
  const text = "intro\n\n$$\n\\begin{aligned}\na &= 1\n\nb &= 2\n\\end{aligned}\n$$\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("b &= 2")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("the window start never lands inside a display math block", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}$$\na &= 1\n\nb &= 2\n$$\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("a &= 1"));
  assert.ok(!text.slice(start).startsWith("b &= 2"), text.slice(start, start + 20));
});

test("inline $$ pairs on one line do not open display math", () => {
  const text = "the cost is $$5$$ per unit\n\ntail";
  assert.equal(isOutsideFence(text, text.length), true);
});

test("a link definition before the window start is carried across", () => {
  const text = "[spec]: https://example.com/spec\n\n" + "filler\n\n".repeat(50) + "see [spec]\n";
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  const carried = linkDefinitionsBefore(text, start);
  assert.ok(carried.includes("[spec]: https://example.com/spec"), carried);
  assert.ok((carried + text.slice(start)).includes("[spec]:"));
});

test("no link definitions means nothing is prepended", () => {
  const text = "filler\n\n".repeat(50);
  assert.equal(linkDefinitionsBefore(text, 100), "");
});

test("aligning inside a long unfinished fence does not rescan per blank line", () => {
  // The scan is one pass over the text. Before this it was one pass PER candidate boundary, and
  // inside an unfinished fence no candidate is ever safe, so every blank line paid a full prefix
  // scan and the whole sum repeated on every streamed token.
  const text = "intro\n\n```py\n" + "x = 1\n\n".repeat(20_000);
  const started = performance.now();
  for (let i = 0; i < 20; i += 1) assert.equal(alignWindowStart(text, 1000), 0);
  const elapsed = performance.now() - started;
  // Twenty aligns over a 140,000 character unfinished fence. Quadratic takes tens of seconds.
  assert.ok(elapsed < 4000, `${elapsed.toFixed(0)}ms for 20 aligns`);
});

test("a $$ inside inline code is prose about math, not math", () => {
  const text = "Use `$$` delimiters.\n\n$$\na &= 1\n\nb &= 2\n$$\n\ntail";
  assert.equal(isOutsideFence(text, text.indexOf("b &= 2")), false);
  assert.equal(isOutsideFence(text, text.length), true);
});

test("a link definition inside a fence is code, not a definition", () => {
  const text = "```md\n[spec]: https://example.com/in-a-fence\n```\n\n" + "filler\n\n".repeat(50);
  const start = alignWindowStart(text, 200);
  assert.ok(start > 0);
  assert.equal(linkDefinitionsBefore(text, start), "");
});

test("a blank line inside a loose list item is not a safe boundary", () => {
  // The item is still open across the blank line, and its continuation is indented BECAUSE that
  // indentation keeps it in the item. Slicing there drops the marker and a four-space
  // continuation becomes an indented code block.
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}1. first point here\n\n    the continuation of the first point\n\n2. second point\n`;
  const start = alignWindowStart(text, text.indexOf("first point"));
  assert.ok(
    !text.slice(start).startsWith("    the continuation"),
    JSON.stringify(text.slice(start, start + 30)),
  );
});

test("a top-level boundary is still accepted", () => {
  const text = "alpha\n\n".repeat(60);
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  assert.ok(text.slice(start).startsWith("alpha"));
});
