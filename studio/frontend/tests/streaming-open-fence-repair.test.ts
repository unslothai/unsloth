// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";

import {
  IncrementalMarkdownCache,
  repairStreamingMarkdown,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";

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

function assertRepairMatches(text: string, chunk = 192): void {
  const cache = new IncrementalMarkdownCache();
  const ends: number[] = [];
  for (let end = chunk; end < text.length; end += chunk) {
    ends.push(end);
  }
  // Fed exactly: some divergences only show on a tail ending on a specific character.
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
      repairStreamingMarkdown(tailOf(cache)),
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

// Shrunk reproducers kept verbatim: a tidier-looking case silently stops covering the bug.
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

// Built by concatenation: in a template literal the backslash escapes the backticks.
const BACKTICK = String.fromCharCode(96);
const ESCAPED_FENCE = `\\${BACKTICK}${BACKTICK}${BACKTICK}`;
const DANGLING_INLINE_BACKTICK = `${BACKTICK}\n\n`;
const FENCE_OPENER = `${BACKTICK}${BACKTICK}${BACKTICK}\n`;

for (const [name, windowText] of [
  ["an escaped fence", `${ESCAPED_FENCE} more`],
  ["display math", "$$odd"],
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
  assert.equal(blocks.join(""), repairStreamingMarkdown(text));
});

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
