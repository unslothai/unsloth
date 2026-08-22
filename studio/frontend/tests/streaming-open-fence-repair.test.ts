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
  for (let end = chunk; end <= text.length; end += chunk) {
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
