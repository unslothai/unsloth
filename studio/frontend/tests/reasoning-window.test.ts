// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { parseMarkdownIntoBlocks } from "streamdown";

import { preprocessLaTeX } from "../src/lib/latex.ts";
import {
  REASONING_WINDOW_CHARS,
  REASONING_WINDOW_RETRY_CHARS,
  advanceReasoningWindow,
  alignWindowStart,
  freshReasoningWindow,
  isOutsideBracketMath,
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

/** Where the renderer itself splits `text`. The window may only start at one of these. */
function boundaries(text: string): Set<number> {
  const starts = new Set<number>();
  let offset = 0;
  for (const block of parseMarkdownIntoBlocks(text)) {
    starts.add(offset);
    offset += block.length;
  }
  return starts;
}

// ── the invariant that makes windowing safe at all ──────────────────
//
// Streamdown renders each block independently, so if block N is already parsed without reference
// to blocks 0..N-1, dropping those blocks cannot change how block N renders. That is the whole
// safety argument, and it reduces to one checkable property: the window starts on a boundary the
// renderer itself chose. Everything the previous hand-written scanner had to be taught construct
// by construct follows from it.

const ADVERSARIAL: Record<string, string> = {
  "indented fence with a blank line":
    "- plan:\n\n  ```python\n  def f():\n      pass\n\n  def g():\n      pass\n  ```\n\ntail",
  "tilde fence": "intro\n\n~~~py\nx = 1\n\ny = 2\n~~~\n\ntail",
  // The closer is indented four spaces to stay in the item, which the hand scanner could not read.
  "ordered list fence with a four-space closer":
    "10. ```js\n    let x = 1;\n\n    let y = 2;\n    ```\n\ntail",
  // The quoted fence ends with its container; the top-level one is a NEW fence, not its closer.
  "quoted fence followed by a top-level fence":
    "> ```\n> quoted\n\n```\nnew fence\n\nstill new\n```\n\ntail",
  "four backticks containing three":
    "intro\n\n````\n```\ninner\n```\n\nstill inner\n````\n\ntail",
  "display math": "intro\n\n$$\na &= 1\n\nb &= 2\n$$\n\ntail",
  "script block": "intro\n\n<script>\nlet a = 1;\n\nlet b = 2;\n</script>\n\ntail",
  "html comment": "intro\n\n<!-- a note\n\nstill the note\n-->\n\ntail",
  "loose list": "1. first point\n\n    the continuation\n\n2. second point\n",
  "a fence that never closes": "intro\n\n```py\nno closing fence\n\nmore\n",
};

for (const [name, fragment] of Object.entries(ADVERSARIAL)) {
  test(`the window only ever starts where the renderer splits: ${name}`, () => {
    const pad = "Prose that fills space.\n\n".repeat(30);
    const text = pad + fragment;
    const allowed = boundaries(text);
    for (let target = 0; target < text.length; target += 23) {
      const start = alignWindowStart(text, target);
      if (start === 0) continue;
      assert.ok(
        allowed.has(start),
        `${name}: start ${start} is not a block boundary; slice begins ` +
          JSON.stringify(text.slice(start, start + 40)),
      );
      assert.ok(text.endsWith(text.slice(start)), `${name}: not a suffix`);
    }
  });
}

test("a slice never begins inside an ordered-list fence", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}10. \`\`\`js\n    let x = 1;\n\n    let y = 2;\n    \`\`\`\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("let x"));
  assert.ok(!text.slice(start).trimStart().startsWith("let y"), text.slice(start, start + 30));
});

test("a top-level fence after a quoted one is not treated as its closer", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}> \`\`\`\n> quoted\n\n\`\`\`\nnew fence\n\nstill new\n\`\`\`\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("new fence"));
  assert.ok(!text.slice(start).trimStart().startsWith("still new"), text.slice(start, start + 30));
});

// ── bracket display math, the one construct the boundaries cannot speak for ──
//
// preprocessLaTeX rewrites \[ ... \] into $$ ... $$ on the whole string BEFORE the document is
// split, so the split never sees this form and a raw-source slice can still land inside one.

test("an offset inside a bracket equation is not outside it", () => {
  const text = "intro\n\n\\[\na &= 1\n\nb &= 2\n\\]\n\ntail";
  assert.equal(isOutsideBracketMath(text, text.indexOf("b &= 2")), false);
  assert.equal(isOutsideBracketMath(text, text.length), true);
});

test("the window start never lands inside bracket display math", () => {
  const pad = "Prose that fills space.\n\n".repeat(30);
  const text = `${pad}\\[\na &= 1\n\nb &= 2\n\\]\n\ntail`;
  const start = alignWindowStart(text, text.indexOf("a &= 1"));
  assert.ok(!text.slice(start).startsWith("b &= 2"), text.slice(start, start + 20));
});

test("an escaped backslash-bracket is a literal, not an equation", () => {
  const text = "an array literal \\\\[1, 2\\\\] in prose\n\ntail\n";
  assert.equal(isOutsideBracketMath(text, text.length), true);
});

// ── the window rule itself ──────────────────────────────────────────

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

test("what is mounted is always a suffix of the real text", () => {
  const text = body(4000);
  let start = 0;
  for (let end = 2000; end <= text.length; end += 2000) {
    const slice = text.slice(0, end);
    start = nextReasoningWindowStart(slice, start);
    assert.ok(slice.endsWith(slice.slice(start)));
  }
});

test("with no safe boundary the whole body renders rather than a cut construct", () => {
  // Showing everything is correct and merely slow; cutting into a fence is wrong.
  const text = "intro\n\n```py\nno closing fence here\nmore\nmore\n";
  assert.equal(alignWindowStart(text, 10), 0);
});

// ── the cost of asking ──────────────────────────────────────────────

test("an alignment that found nothing is not retried on the very next chunk", () => {
  const stream = `\`\`\`python\n${"x = 1\n\n".repeat(4000)}`;
  let state = freshReasoningWindow();
  let scans = 0;
  for (let end = 24; end <= stream.length; end += 24) {
    const before = state;
    state = advanceReasoningWindow(stream.slice(0, end), state);
    if (state.retryAt !== before.retryAt) scans += 1;
  }
  assert.equal(state.start, 0);
  assert.ok(
    scans < stream.length / REASONING_WINDOW_RETRY_CHARS + 5,
    `${scans} scans over ${stream.length} chars`,
  );
});

test("backing off never delays the window by more than the retry step", () => {
  const stream = "alpha\n\n".repeat(6000);
  let state = freshReasoningWindow();
  let engagedAt = -1;
  for (let end = 24; end <= stream.length; end += 24) {
    state = advanceReasoningWindow(stream.slice(0, end), state);
    if (state.start > 0 && engagedAt < 0) engagedAt = end;
  }
  assert.ok(engagedAt > 0, "the window never engaged at all");
  assert.ok(
    engagedAt <= REASONING_WINDOW_CHARS * 1.5 + REASONING_WINDOW_RETRY_CHARS + 24,
    `engaged at ${engagedAt}`,
  );
});

test("the carried definitions are recomputed only when the start moves", () => {
  // They change only with the start, which moves every 6,000 characters, while renders arrive
  // every frame. Recomputing per render rescans the whole immutable prefix each time.
  const stream = `[spec]: https://example.com/spec\n\n${"filler text here\n\n".repeat(3000)}`;
  let state = freshReasoningWindow();
  let moves = 0;
  for (let end = 24; end <= stream.length; end += 24) {
    const before = state.start;
    state = advanceReasoningWindow(stream.slice(0, end), state);
    if (state.start !== before) moves += 1;
  }
  assert.ok(state.start > 0, "the window never engaged");
  assert.ok(state.definitions.includes("[spec]:"), state.definitions);
  assert.ok(moves < 40, `${moves} window moves`);
});

// ── link-reference definitions ──────────────────────────────────────

test("a definition before the window start is carried across", () => {
  const text = `[spec]: https://example.com/spec\n\n${"filler\n\n".repeat(50)}see [spec]\n`;
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  assert.ok(linkDefinitionsBefore(text, start).includes("[spec]: https://example.com/spec"));
});

test("no definitions means nothing is prepended", () => {
  assert.equal(linkDefinitionsBefore("filler\n\n".repeat(50), 100), "");
});

test("a definition inside a block quote is carried without its quote marker", () => {
  const text = `> [spec]: https://example.com/spec\n\n${"filler\n\n".repeat(50)}see [spec]\n`;
  const start = alignWindowStart(text, 100);
  const carried = linkDefinitionsBefore(text, start);
  assert.ok(carried.includes("[spec]: https://example.com/spec"), carried);
  assert.ok(!carried.includes(">"), carried);
});

test("a definition inside a fence is code, not a definition", () => {
  const text = `\`\`\`md\n[spec]: https://example.com/in-a-fence\n\`\`\`\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 200);
  assert.ok(start > 0);
  assert.equal(linkDefinitionsBefore(text, start), "");
});

test("a definition inside an HTML block is markup, not a definition", () => {
  const text = `<div>\n[spec]: https://example.com/in-html\n</div>\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 200);
  assert.ok(start > 0);
  assert.equal(linkDefinitionsBefore(text, start), "");
});

test("half a definition is never hoisted on its own", () => {
  // `[spec]:` alone is not a definition, so carrying it would render as literal text at the top
  // of the window: a visible artefact made by the machinery that exists to avoid one.
  const text = `[spec]:\n    https://example.com/spec\n    "The Spec"\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 200);
  assert.ok(start > 0);
  const carried = linkDefinitionsBefore(text, start);
  if (carried !== "") {
    assert.ok(carried.includes("https://example.com/spec"), carried);
  }
});

test("a label containing an escaped bracket is still a definition", () => {
  const text = `[spec\\]]: https://example.com/spec\n\n${"filler\n\n".repeat(50)}see [spec\\]]\n`;
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  assert.ok(
    linkDefinitionsBefore(text, start).includes("https://example.com/spec"),
    linkDefinitionsBefore(text, start),
  );
});

test("a bare label with no destination is still not a definition", () => {
  // The control: widening the label pattern must not start hoisting half-definitions again.
  const text = `[spec]:\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  assert.equal(linkDefinitionsBefore(text, start), "");
});

test("a literal backslash-bracket inside a fence does not disable the window forever", () => {
  // preprocessLaTeX skips code, so a `\[` in a code sample is a literal. Treating it as an opener
  // left the scanner believing it was inside an equation for the rest of the stream, and
  // alignWindowStart then returns 0 forever: the pane silently never becomes windowed.
  const text = `intro\n\n\`\`\`tex\n\\[ not really an equation\n\`\`\`\n\n${"filler\n\n".repeat(60)}`;
  assert.equal(isOutsideBracketMath(text, text.length), true);
  assert.ok(alignWindowStart(text, 400) > 0, "the window never engaged after a fenced backslash");
});

test("a literal backslash-bracket in inline code does not disable the window", () => {
  const text = `Use \`\\[\` for display math.\n\n${"filler\n\n".repeat(60)}`;
  assert.equal(isOutsideBracketMath(text, text.length), true);
  assert.ok(alignWindowStart(text, 400) > 0);
});

test("a paragraph that only looks like a definition is not carried", () => {
  // Checked against remark, the parser in this very pipeline: `[spec]: /url not a title` is a
  // PARAGRAPH, because what follows the destination is not a valid title. Carrying it would put
  // text the reader has already read back at the top of the pane.
  const text = `[spec]: /url not a title\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 100);
  assert.ok(start > 0);
  assert.equal(linkDefinitionsBefore(text, start), "");
});

test("a destination in angle brackets may contain spaces and is still a definition", () => {
  // The control. remark parses `[spec]: <invalid destination>` as a DEFINITION, spaces and all,
  // so refusing it would drop a real one.
  const text = `[spec]: <a destination with spaces>\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 100);
  assert.ok(linkDefinitionsBefore(text, start).includes("a destination with spaces"));
});

test("a real definition with a title is still carried", () => {
  const text = `[spec]: https://example.com/spec "The Spec"\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 100);
  assert.ok(linkDefinitionsBefore(text, start).includes("https://example.com/spec"));
});

test("a real definition in angle brackets is still carried", () => {
  const text = `[spec]: <https://example.com/spec>\n\n${"filler\n\n".repeat(50)}`;
  const start = alignWindowStart(text, 100);
  assert.ok(linkDefinitionsBefore(text, start).includes("https://example.com/spec"));
});

test("junk after an angle-bracket destination makes it a paragraph, not a definition", () => {
  // Ground truth from remark, the parser in this pipeline: only the first of these three is a
  // definition. The other two render visibly, so carrying one puts read text back on screen.
  const definition = `[spec]: <https://example.com/spec> "Title"`;
  const paragraphs = [
    `[spec]: <https://example.com/spec> junk here`,
    `[spec]: <https://example.com/spec>extra`,
  ];
  const filler = `\n\n${"filler\n\n".repeat(50)}`;
  const first = definition + filler;
  assert.ok(
    linkDefinitionsBefore(first, alignWindowStart(first, 100)).includes("example.com/spec"),
  );
  for (const line of paragraphs) {
    const text = line + filler;
    assert.equal(linkDefinitionsBefore(text, alignWindowStart(text, 100)), "", line);
  }
});

test("a bracket span longer than the preprocessor's cap does not disable the window", () => {
  // `preprocessLaTeX` caps a `\[ ... \]` body at 4,096 characters and leaves anything longer as
  // ordinary text. Treating such an opener as live would leave this scanner believing it was
  // inside an equation for the rest of the stream, and `alignWindowStart` would return 0 forever:
  // the pane would silently never window, which is the same quiet total failure a `\[` in a code
  // sample used to cause.
  const overCap = `\\[\n${"x + y = z\n".repeat(600)}\\]`;
  const text = `intro\n\n${overCap}\n\n${"filler\n\n".repeat(80)}`;
  // The premise, checked rather than assumed.
  assert.ok(overCap.length - 4 > 4096);
  assert.ok(!preprocessLaTeX(text).includes("$$"), "premise: the preprocessor ignores it");
  assert.ok(alignWindowStart(text, text.length - 500) > 0);

  // And the control: a span INSIDE the cap is still respected.
  const underCap = `\\[\n${"x + y = z\n".repeat(100)}\\]`;
  const inside = `intro\n\n${underCap}\n\n${"filler\n\n".repeat(80)}`;
  assert.ok(preprocessLaTeX(inside).includes("$$"), "premise: the preprocessor rewrites it");
  const openerAt = inside.indexOf("\\[");
  assert.equal(isOutsideBracketMath(inside, openerAt + 40), false);
});

test("alignment stays linear when inline code alternates with bracket openers", () => {
  // Every delimiter used to be checked against every code region. This is the shape that made
  // that quadratic: as many regions as delimiters, all of them the same kind.
  //
  // Measured as a RATIO rather than a wall clock, because the absolute numbers are small enough
  // that a machine-specific threshold would either pass on the quadratic version or fail on a
  // loaded runner. Four times the input against a linear scan costs 11.1x here and 4.1x with the
  // search, so 7 separates them with room on both sides.
  const build = (n: number) => `${"pad\n\n".repeat(20)}${"`\\[` and text\n\n".repeat(n)}`;
  const time = (n: number) => {
    const text = build(n);
    const at = performance.now();
    alignWindowStart(text, Math.floor(text.length / 2));
    return performance.now() - at;
  };
  time(1_000);
  const small = time(4_000);
  const large = time(16_000);
  assert.ok(large < small * 7, `4,000 took ${small}ms, 16,000 took ${large}ms`);
});

test("a footnote arriving mid-stream turns the window off rather than freezing it", () => {
  // The premise, checked rather than assumed: a GFM footnote definition makes the renderer treat
  // the WHOLE document as one block, so a start that was a boundary a chunk ago is not one now.
  const body = "A paragraph of reasoning text here.\n\n".repeat(900);
  const plain = `${body}Tail.\n`;
  const withNote = `${body}See it[^1].\n\n${"More reasoning text.\n\n".repeat(2000)}[^1]: The note.\n`;
  assert.ok(parseMarkdownIntoBlocks(plain).length > 100);
  assert.equal(parseMarkdownIntoBlocks(withNote).length, 1);

  // Engaged on the plain text, then the footnote arrives.
  const engaged = nextReasoningWindowStart(plain, 0);
  assert.ok(engaged > 0);
  // The freeze only shows once the mounted suffix has outgrown the window, which is the point at
  // which the alignment is asked again. Until then the start is simply not reconsidered.
  assert.ok(withNote.length - engaged > REASONING_WINDOW_CHARS * 1.5);
  assert.equal(alignWindowStart(withNote, withNote.length - REASONING_WINDOW_CHARS), 0);

  // Retaining `engaged` would slice a document the renderer treats as indivisible, AND would
  // freeze the start while the text kept growing: 31,729 mounted characters from a window that
  // caps at 18,000. Turning off is the only safe answer.
  assert.equal(nextReasoningWindowStart(withNote, engaged), 0);
  const state = advanceReasoningWindow(withNote, {
    start: engaged,
    retryAt: 0,
    definitions: "[spec]: /url\n\n",
  });
  assert.equal(state.start, 0);
  assert.equal(state.definitions, "", "carried definitions go with the window");
});
