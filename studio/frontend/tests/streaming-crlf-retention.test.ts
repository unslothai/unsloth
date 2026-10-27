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

const asCrlf = (text: string): string => text.replace(/\r?\n/g, "\r\n");
const asLf = (text: string): string => text.replace(/\r\n?/g, "\n");

const UNITS = [
  "The residual \\(r_i = y_i - \\hat{y}_i\\) shrinks as the fit improves.\n\n",
  "Rewriting gives\n\n\\[ L(\\theta) = \\sum_i (y_i - \\theta x_i)^2 \\]\n\nwhich is convex.\n\n",
  "At that batch size the run costs about $1,200 per epoch.\n\n",
  "- learning rate three ten-thousandths\n- budget $250\n\n",
  "```python\ndef step(theta, grad, lr):\n    return theta - lr * grad\n```\n\n",
];

function buildReply(count: number): string {
  let out = "";
  for (let index = 0; index < count; index += 1) {
    out += UNITS[index % UNITS.length];
  }
  return out;
}

function stream(reply: string, step: number) {
  const cache = new IncrementalMarkdownCache();
  let blocks: string[] = [];
  for (let length = step; length <= reply.length; length += step) {
    const render = cache.update(processStreamingText(reply.slice(0, length)));
    blocks = render.parseMarkdownIntoBlocks(render.markdown);
  }
  const render = cache.update(processStreamingText(reply));
  blocks = render.parseMarkdownIntoBlocks(render.markdown);
  const internals = cache as unknown as { committedLength: number };
  return { blocks, retained: internals.committedLength };
}

test("a CRLF reply retains as much as the same reply in LF", () => {
  // The cache compares offsets against blocks with normalised line endings; CRLF used to
  // defeat retention silently, re-lexing the whole reply every frame.
  const lf = buildReply(160);
  const crlf = asCrlf(lf);
  assert.ok(lf.length > 8_000, `fixture too small: ${lf.length}`);

  const lfRun = stream(lf, 24);
  const crlfRun = stream(crlf, 24);

  assert.ok(
    lfRun.retained > lf.length * 0.8,
    `the LF control retained only ${lfRun.retained} of ${lf.length}`,
  );
  assert.equal(
    crlfRun.retained,
    lfRun.retained,
    `CRLF retained ${crlfRun.retained} characters against LF's ${lfRun.retained}`,
  );
  assert.deepEqual(crlfRun.blocks, lfRun.blocks);
});

test("a CRLF reply matches a whole-document split at every prefix", () => {
  // Compare against the NORMALISED text: CommonMark treats LF, CR and CRLF as one line ending.
  const sources = [
    "para one\r\n\r\npara two\r\n\r\npara three\r\n\r\npara four\r\n\r\n",
    asCrlf("Cost $1,200 now.\n\nThe value \\(x^2\\) here.\n\n\\[a = b\\]\n\ndone\n\n"),
    asCrlf("```sh\nrun --seed $1\n```\n\nAfter the fence $5.\n\n"),
    asCrlf("| a | b |\n| --- | --- |\n| $5 | \\(x\\) |\n\nAfter the table.\n\n"),
    // A lone trailing CR may have its LF still to come, so it must read as a line ending.
    "a\r\n\r\nb\r",
  ];
  for (const source of sources) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 0; length <= source.length; length += 1) {
      const input = processStreamingText(source.slice(0, length));
      const render = cache.update(input);
      assert.deepEqual(
        render.parseMarkdownIntoBlocks(render.markdown),
        parseMarkdownIntoBlocks(repairStreamingMarkdown(asLf(input))),
        `block mismatch at prefix ${length} of ${JSON.stringify(source.slice(0, 60))}`,
      );
    }
  }
});

test("an LF reply is untouched by the line-ending handling", () => {
  const reply = buildReply(60);
  assert.ok(!reply.includes("\r"));
  const run = stream(reply, 24);
  const cache = new IncrementalMarkdownCache();
  for (let length = 0; length <= reply.length; length += 24) {
    const input = processStreamingText(reply.slice(0, length));
    const render = cache.update(input);
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(repairStreamingMarkdown(input)),
      `block mismatch at prefix ${length}`,
    );
  }
  assert.ok(run.retained > reply.length * 0.8);
});
