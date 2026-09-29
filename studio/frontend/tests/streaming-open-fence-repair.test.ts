// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";

import { IncrementalMarkdownCache } from "../src/components/assistant-ui/streaming-render-schedule.ts";

/**
 * A fence that has opened and not yet closed pins the cache: the whole fence
 * lexes into one block, so no prefix is ever retained and the live tail grows
 * with the fence. Repairing that tail is worse than linear -- remend walks the
 * whole string once per `[`, and code is dense in brackets -- so appending one
 * chunk used to cost the length of the fence.
 *
 * Two things have to hold for the shortcut that fixes it: the repaired text is
 * the text the whole-tail repair would have produced, and the work per chunk no
 * longer grows with the fence.
 */

const codeLine = (index: number) =>
  `def step_${index}(x, y=${index}):\n    z = [item for item in range(${index})]  # note\n    return {'z': z, 'x': x * y}\n`;

function fenceBody(lines: number): string {
  let body = "";
  for (let index = 0; index < lines; index += 1) {
    body += codeLine(index);
  }
  return body;
}

// Comfortably past OPEN_FENCE_REPAIR_WINDOW, so the cut cannot land inside it.
const OVERLONG_BLANK_LINE = 6_000;

const tailOf = (cache: IncrementalMarkdownCache): string =>
  (cache as unknown as { tail: string }).tail;

const contextOf = (cache: IncrementalMarkdownCache): Record<string, unknown> =>
  (cache as unknown as { context: Record<string, unknown> }).context;

/**
 * Stream `text` in fixed chunks and assert every frame renders the Markdown the
 * whole-tail repair would have rendered.
 *
 * After update() the cache's own tail is the text its return value repaired,
 * whether or not that update retained a block, so `remend(tail)` is the
 * reference render. It is only the reference while the retained context carries
 * no marker prefix, which the assertion below holds these replies to.
 */
function assertRepairMatches(text: string, chunk = 192): void {
  const cache = new IncrementalMarkdownCache();
  const ends: number[] = [];
  for (let end = chunk; end < text.length; end += chunk) {
    ends.push(end);
  }
  // The last frame is fed exactly, not at whatever multiple of the chunk size
  // happens to land near it. Several of the divergences this file exists to
  // catch only show on a tail that ends on a specific character.
  ends.push(text.length);
  for (const end of ends) {
    const render = cache.update(text.slice(0, end));
    const context = contextOf(cache);
    assert.equal(
      context.bold || context.singleAsterisk || context.singleUnderscore,
      false,
      "reply must not need a marker context for remend(tail) to be the reference",
    );
    assert.equal(
      render.markdown,
      remend(tailOf(cache)),
      `frame at ${end} characters diverged from the whole-tail repair`,
    );
  }
}

test("an unterminated fence repairs to the whole-tail result", () => {
  assertRepairMatches(`Here is the code.\n\n\`\`\`python\n${fenceBody(180)}`);
});

test("a fence that closes and reopens repairs to the whole-tail result", () => {
  const text = `Intro paragraph one.\n\nIntro paragraph two.\n\n\`\`\`python\n${fenceBody(70)}\`\`\`\n\nProse between the fences.\n\n\`\`\`js\n${fenceBody(70)}\`\`\`\n\nClosing prose.\n`;
  assertRepairMatches(text);
});

test("CRLF inside an unterminated fence repairs to the whole-tail result", () => {
  const text = `Intro.\r\n\r\n\`\`\`python\r\n${fenceBody(110).replace(/\n/g, "\r\n")}`;
  assertRepairMatches(text);
});

test("a tilde fence keeps the whole-tail repair", () => {
  assertRepairMatches(`Intro.\n\n~~~python\n${fenceBody(110)}`);
});

test("a nested longer fence repairs to the whole-tail result", () => {
  assertRepairMatches(
    `Intro.\n\n\`\`\`\`markdown\n\`\`\`python\n${fenceBody(110)}`,
  );
});

test("display math inside an open fence keeps the whole-tail repair", () => {
  assertRepairMatches(`Intro.\n\n\`\`\`text\n$$\n${fenceBody(90)}`);
});

test("a marker left open before the fence keeps the whole-tail repair", () => {
  assertRepairMatches(`Intro **bold\n\n\`\`\`python\n${fenceBody(110)}`);
});

/**
 * A whitespace-only line longer than the repair window, then a tail that looks
 * like a setext underline.
 *
 * The window is cut at the first line boundary at or after its start, so a
 * preceding line this long puts the cut on the newline that ENDS it and leaves
 * the window holding the final line alone. remend's setext repair reads exactly
 * one line back, and the opener that stands in for the head is never blank, so
 * the elided blank line used to become a non-blank one and the repair appended a
 * zero-width space the whole-tail repair does not add -- a hidden character in
 * rendered code, and on the clipboard behind the copy button.
 *
 * Reported on unslothai/unsloth#9517 as review item 3836360429.
 */
for (const filler of [" ", "\t"]) {
  for (const underline of ["-", "--", "=", "=="]) {
    const label = JSON.stringify(filler);
    test(`a ${label} line longer than the window before ${underline} keeps the whole-tail repair`, () => {
      const blank = filler.repeat(OVERLONG_BLANK_LINE);
      assertRepairMatches(
        `Here is the code.\n\n\`\`\`python\nx = 1\n${blank}\n${underline}`,
      );
    });
  }
}

/**
 * A ` $ or ~ inside the part of the fence body that the window elides.
 *
 * remend has at least three notions of "inside a fence" and they disagree: the
 * escape-aware walk behind `isWithinCodeBlock`, the emphasis counters that
 * toggle on ``` without honouring a backslash, and the inline-code repair that
 * just counts /```/g. An escaped \``` or a ```` run flips some and not others,
 * so the opener that stands in for the elided text cannot reproduce them all
 * once the elided part carries one of those characters, and the repair appends
 * a stray closer the whole-tail repair does not add.
 *
 * These two are the shrunk reproducers, kept verbatim: the shape needs a stray
 * backtick before the fence, a longer opener, and an escaped fence inside it,
 * so a tidier-looking case silently stops covering the bug.
 */
test("an escaped fence in the elided body keeps the whole-tail repair", () => {
  assertRepairMatches(
    `a\`\n\n\`\`\`\`md\n\\\`\`\`\n${"ab".repeat(3000)}\n===\n`,
  );
});

test("an escaped fence before two overlong lines keeps the whole-tail repair", () => {
  assertRepairMatches(
    `- a\n- b\n\n\`\`\`\`md\n\\\`\`\`\n* * *\n${" ".repeat(5000)}\n${"x".repeat(5000)}\n\`tick\n`,
  );
});

// Built by concatenation, not escapes: a backslash followed by a triple-backtick
// run is exactly the sequence this bug needs, and writing it inside a template
// literal produced three ESCAPED backticks instead, which is a different string
// and quietly stopped covering the case.
const BACKTICK = String.fromCharCode(96);
const ESCAPED_FENCE = `\\${BACKTICK}${BACKTICK}${BACKTICK}`;
const DANGLING_INLINE_BACKTICK = `${BACKTICK}\n\n`;
const FENCE_OPENER = `${BACKTICK}${BACKTICK}${BACKTICK}\n`;

/**
 * An unmatched inline backtick BEFORE the opener, and a marker in the part of
 * the window the splice RETAINS rather than elides.
 *
 * The probe that licenses the opener standing in for the head reads the head
 * alone, and it runs before the cut is chosen. remend decides from the whole
 * string: an escaped \``` later in the body flips the global triple-run parity,
 * so repairing the whole tail closes the head's dangling backtick while the
 * spliced output, whose synthetic head never had one, does not. A `$$` there
 * moves the math parity the other way and the splice appends where the whole
 * tail does not. Neither is reachable from a refusal on the cut, which is why
 * the body-marker refusal covers the retained window and not just the middle.
 *
 * Reported on unslothai/unsloth#9517 as review item 3836846709.
 */
for (const [name, windowText] of [
  ["an escaped fence", `${ESCAPED_FENCE} more`],
  ["display math", "$$odd"],
  // A `~~` here does NOT reproduce and is deliberately absent: a case that
  // passes on the broken code documents nothing and reads as coverage.
] as const) {
  test(`${name} in the retained window keeps the whole-tail repair`, () => {
    assertRepairMatches(
      `${DANGLING_INLINE_BACKTICK}${FENCE_OPENER}${"x".repeat(OVERLONG_BLANK_LINE)}\n${windowText}\n`,
    );
  });
}

test("a fence that never closes stays live and is never committed away", () => {
  const text = `Intro.\n\n\`\`\`python\n${fenceBody(300)}`;
  const cache = new IncrementalMarkdownCache();
  let render = cache.update(text.slice(0, 64));
  for (let end = 128; end < text.length; end += 128) {
    render = cache.update(text.slice(0, end));
  }
  render = cache.update(text);
  const blocks = render.parseMarkdownIntoBlocks(render.markdown);
  assert.equal(blocks.join(""), remend(text));
});

/**
 * A complexity test, not a timing test. remend decides where a construct ends
 * with `substring` probes whose receiver is the whole string, so charging every
 * probe its receiver's length prices a repair at the length of what it was
 * given. Doubling the fence must roughly double that bill, which is what a
 * per-chunk cost independent of the fence looks like. Before this shortcut the
 * same doubling multiplied it by about 16.
 */
function repairCharacters(text: string, chunk = 256): number {
  const realSubstring = String.prototype.substring;
  let counted = 0;
  String.prototype.substring = function counting(
    this: string,
    start: number,
    end?: number,
  ): string {
    counted += this.length;
    return realSubstring.call(this, start, end);
  } as unknown as typeof realSubstring;
  try {
    const cache = new IncrementalMarkdownCache();
    for (let end = chunk; end <= text.length; end += chunk) {
      cache.update(text.slice(0, end));
    }
  } finally {
    String.prototype.substring = realSubstring;
  }
  return counted;
}

test("appending to an open fence does not cost the length of the fence", () => {
  const short = repairCharacters(`Intro.\n\n\`\`\`python\n${fenceBody(180)}`);
  const long = repairCharacters(`Intro.\n\n\`\`\`python\n${fenceBody(360)}`);
  assert.ok(
    long < short * 3,
    `doubling the fence multiplied the repair by ${(long / short).toFixed(1)}`,
  );
});
