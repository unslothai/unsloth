// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { parseMarkdownIntoBlocks } from "streamdown";

import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import {
  IncrementalMarkdownCache,
  repairStreamingMarkdown,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const processStreamingText = (text: string): string =>
  stabilizeStreamingMarkdown(preprocessLaTeX(text), true);

const rebuilds = (cache: IncrementalMarkdownCache): number =>
  (cache as unknown as { retainedPrefixRebuilds: number })
    .retainedPrefixRebuilds;

const rewound = (cache: IncrementalMarkdownCache): number =>
  (cache as unknown as { rewoundCharacters: number }).rewoundCharacters;

const retainedBlocks = (render: {
  parseMarkdownIntoBlocks: (markdown: string) => string[];
}): string[] => render.parseMarkdownIntoBlocks("");

const REPLY_UNITS = [
  "The residual term \\(r_i = y_i - \\hat{y}_i\\) shrinks as the fit improves.\n\n",
  "Rewriting gives\n\n\\[ L(\\theta) = \\sum_i (y_i - \\theta x_i)^2 \\]\n\nwhich is convex.\n\n",
  "At that batch size the run costs about $1,200 per epoch on rented hardware.\n\n",
  "- learning rate three ten-thousandths\n- weight decay lambda\n- budget $250\n\n",
  "```python\ndef step(theta, grad, lr):\n    return theta - lr * grad\n```\n\n",
  "That leaves a headroom of $3.5M in the yearly plan, which is the binding limit.\n\n",
];

const buildReply = (units: number): string =>
  Array.from(
    { length: units },
    (_, index) => REPLY_UNITS[index % REPLY_UNITS.length],
  ).join("");

const plainVariant = (reply: string): string =>
  reply
    .replaceAll("\\(", "(")
    .replaceAll("\\)", ")")
    .replaceAll("\\[", "(")
    .replaceAll("\\]", ")")
    .replaceAll("$1,200", "1200 dollars")
    .replaceAll("$250", "250 dollars")
    .replaceAll("$3.5M", "3.5M dollars");

function streamReply(
  reply: string,
  step: number,
): { cache: IncrementalMarkdownCache; frames: number; retained: number } {
  const cache = new IncrementalMarkdownCache();
  let frames = 0;
  let retained = 0;
  for (let length = step; length <= reply.length; length += step) {
    const render = cache.update(processStreamingText(reply.slice(0, length)));
    retained = retainedBlocks(render).join("").length;
    frames += 1;
  }
  return { cache, frames, retained };
}

test("a LaTeX or currency rewrite never rebuilds the retained prefix", () => {
  const reply = buildReply(120);
  assert.ok(reply.length > 8_000, `fixture too small: ${reply.length}`);

  const { cache, frames, retained } = streamReply(reply, 24);
  assert.ok(frames > 300, `expected a long stream, got ${frames}`);
  assert.equal(
    rebuilds(cache),
    0,
    `retained prefix rebuilt ${rebuilds(cache)} times over ${frames} frames`,
  );
  assert.ok(
    retained > reply.length * 0.8,
    `retained only ${retained} of ${reply.length} characters`,
  );

  const plain = streamReply(plainVariant(reply), 24);
  assert.equal(rebuilds(plain.cache), 0);
  assert.ok(plain.retained > reply.length * 0.8);
});

test("a rewrite behind the live tail keeps the blocks it cannot reach", () => {
  // LATEX_DELIM_RE lets an inline span run 4,096 chars, so `\(` can rewrite committed text.
  const lead = Array.from(
    { length: 30 },
    (_, index) => `Lead paragraph ${index}.\n\n`,
  ).join("");
  const span = `\\(${Array.from(
    { length: 30 },
    (_, index) => `span line ${index}\n\n`,
  ).join("")}\\)`;
  const reply = `${lead}${span} done\n\n${Array.from(
    { length: 20 },
    (_, index) => `tail ${index}\n\n`,
  ).join("")}`;

  const cache = new IncrementalMarkdownCache();
  let beforeClose = "";
  let atClose = "";
  for (let length = 1; length <= reply.length; length += 1) {
    const render = cache.update(processStreamingText(reply.slice(0, length)));
    const retained = retainedBlocks(render).join("");
    if (length === lead.length + span.length - 1) {
      beforeClose = retained;
    }
    if (length === lead.length + span.length) {
      atClose = retained;
    }
  }

  assert.equal(
    rebuilds(cache),
    0,
    "closing the span rebuilt the whole retained prefix",
  );
  assert.ok(
    beforeClose.length > lead.length,
    "the span body was never committed, so nothing was rewound",
  );
  assert.ok(
    lead.startsWith(atClose),
    `kept ${atClose.length} characters, past the ${lead.length} the rewrite ` +
      "cannot reach",
  );
  assert.ok(
    atClose.length > lead.length * 0.7,
    `expected most of the ${lead.length} characters before the opener to ` +
      `survive, kept ${atClose.length}`,
  );
  assert.ok(
    rewound(cache) < beforeClose.length - lead.length * 0.7,
    "the rewind gave back more than the span plus its rollback window",
  );
});

test("a closing fence rewrites its own body without a rebuild", () => {
  // An open fence lexes to one live block, so the opener is never behind the commit boundary.
  const lead = Array.from(
    { length: 40 },
    (_, index) =>
      `Intro paragraph number ${index}.\n\nAnother line ${index}\n\n`,
  ).join("");
  const fence = "```sh\nrun --seed $1 --limit $2\n```\n\n";
  const reply = `${lead}${fence}${Array.from(
    { length: 20 },
    (_, index) => `closing remark ${index}\n\n`,
  ).join("")}`;

  const cache = new IncrementalMarkdownCache();
  let beforeFence = "";
  let afterFence = "";
  for (let length = 1; length <= reply.length; length += 1) {
    const render = cache.update(processStreamingText(reply.slice(0, length)));
    if (length === lead.length) {
      beforeFence = retainedBlocks(render).join("");
    }
    if (length === lead.length + fence.length) {
      afterFence = retainedBlocks(render).join("");
    }
  }

  assert.ok(beforeFence.length > 0, "nothing was retained before the fence");
  assert.equal(
    rebuilds(cache),
    0,
    "the fence rewrite rebuilt the whole prefix",
  );
  assert.ok(
    afterFence.startsWith(beforeFence),
    "the prefix retained before the fence did not survive it",
  );
});

test("retaining across a rewrite still matches a full Streamdown split", () => {
  const spanning = `${Array.from(
    { length: 12 },
    (_, index) => `Lead paragraph ${index}.\n\n`,
  ).join("")}\\(${Array.from(
    { length: 12 },
    (_, index) => `span line ${index}\n\n`,
  ).join("")}\\) done\n\n`;

  const twice = `${spanning}${spanning}`;

  // Marked reads a run of blank lines as ONE separator block, so the closing `\n$$\n`
  // re-segments the block before the opener though its characters never moved.
  const displayBody = `${Array.from(
    { length: 5 },
    (_, index) => `s${index}\n\n`,
  ).join("")}`;
  const blankRunMerge = `Lead paragraph.\n\n\\[${displayBody}\\]\n\n`;

  for (const reply of [
    buildReply(6),
    plainVariant(buildReply(6)),
    spanning,
    twice,
    blankRunMerge,
    `${spanning}${blankRunMerge}`,
  ]) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 0; length <= reply.length; length += 1) {
      const input = processStreamingText(reply.slice(0, length));
      const render = cache.update(input);
      assert.deepEqual(
        render.parseMarkdownIntoBlocks(render.markdown),
        parseMarkdownIntoBlocks(repairStreamingMarkdown(input)),
        `block mismatch at prefix ${length}`,
      );
    }
  }
});

test("a rewind restores the repair context of the commit it lands on", () => {
  const tail =
    "Then _under first_ and **bold second** mixed\n\n`code *star* span`\n\nend\n\n";
  const reply = `${Array.from(
    { length: 12 },
    (_, index) => `Lead paragraph ${index}.\n\n`,
  ).join("")}\\(${Array.from(
    { length: 12 },
    (_, index) => `span _line ${index}_ here\n\n`,
  ).join("")}\\) done\n\n${tail}`;

  const cache = new IncrementalMarkdownCache();
  for (let length = 0; length <= reply.length; length += 1) {
    const input = processStreamingText(reply.slice(0, length));
    const render = cache.update(input);
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(repairStreamingMarkdown(input)),
      `block mismatch at prefix ${length}`,
    );
  }
});

test("an edit that closes up a blank line cannot keep the block before it", () => {
  // Marked reads `paragraph 0\n` plus new text as a lazy continuation, so an edit at a
  // commit boundary re-segments the previous paragraph.
  const paragraphs = Array.from(
    { length: 60 },
    (_, index) => `paragraph ${index}\n\n`,
  ).join("");
  const quoted = `> quote line\n\n${Array.from(
    { length: 40 },
    (_, index) => `body ${index}\n\n`,
  ).join("")}`;

  const cases: Array<[string, string]> = [
    [paragraphs, `${paragraphs.slice(0, 11)}!${paragraphs.slice(12)}`],
    [quoted, `> quote line$\ncost ${quoted.slice(14)}`],
  ];

  for (const [source, edited] of cases) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 7; length <= source.length; length += 7) {
      cache.update(source.slice(0, length));
    }
    cache.update(source);
    const render = cache.update(edited);
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(repairStreamingMarkdown(edited)),
      `block mismatch after ${JSON.stringify(edited.slice(0, 24))}`,
    );
  }
});

test("a reply with math streams near the cost of one without", () => {
  const reply = buildReply(300);
  assert.ok(reply.length > 20_000, `fixture too small: ${reply.length}`);
  const plain = plainVariant(reply);
  const mathTimes: number[] = [];
  const plainTimes: number[] = [];

  streamReply(reply, 24);
  streamReply(plain, 24);

  for (let repeat = 0; repeat < 5; repeat += 1) {
    let started = performance.now();
    const math = streamReply(reply, 24);
    mathTimes.push(performance.now() - started);
    started = performance.now();
    streamReply(plain, 24);
    plainTimes.push(performance.now() - started);
    assert.ok(math.retained > reply.length * 0.8);
  }

  const fastest = (values: number[]): number => Math.min(...values);
  const ratio = fastest(mathTimes) / fastest(plainTimes);
  // Measured 9.8x before the rewind and 1.1-1.2x after; 4 leaves room on a loaded host.
  assert.ok(
    ratio < 4,
    `math reply cost ${ratio.toFixed(1)}x the plain reply ` +
      `(math ${fastest(mathTimes).toFixed(0)} ms, plain ${fastest(plainTimes).toFixed(0)} ms)`,
  );
});
