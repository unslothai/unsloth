// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import { IncrementalMarkdownCache } from "../src/components/assistant-ui/streaming-render-schedule.ts";
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
  // Providers and platforms disagree about line endings, and the cache compares
  // byte offsets in the text it is handed against blocks Streamdown returns with
  // the line endings already normalised. Before this was fixed the two disagreed
  // one block in: nothing was ever committed, the sticky full-document path took
  // over, and a CRLF reader paid a repair and a lex of the whole reply on every
  // frame for a reply that rendered exactly the same. Nothing in the output says
  // so, which is why it survived.
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
  // Retention is only worth having if the blocks are the same ones, so pin the
  // output too: a CRLF reply must render as the identical block list.
  assert.deepEqual(crlfRun.blocks, lfRun.blocks);
});

test("a CRLF reply matches a whole-document split at every prefix", () => {
  // The correctness half. The comparison is against a parse of the NORMALISED
  // text, because that is what a CommonMark parser sees: the spec counts a line
  // feed, a lone carriage return, and a carriage return followed by a line feed
  // as the same line ending, and reference parsers normalise before parsing.
  const sources = [
    "para one\r\n\r\npara two\r\n\r\npara three\r\n\r\npara four\r\n\r\n",
    asCrlf("Cost $1,200 now.\n\nThe value \\(x^2\\) here.\n\n\\[a = b\\]\n\ndone\n\n"),
    asCrlf("```sh\nrun --seed $1\n```\n\nAfter the fence $5.\n\n"),
    asCrlf("| a | b |\n| --- | --- |\n| $5 | \\(x\\) |\n\nAfter the table.\n\n"),
    // A carriage return can land at the end of a frame with its line feed still
    // to come, so a lone trailing CR has to read as a line ending too. Treating
    // only the pair would leave that CR in place and then delete it a frame
    // later, which is one more rewrite for the cache to absorb.
    "a\r\n\r\nb\r",
  ];
  for (const source of sources) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 0; length <= source.length; length += 1) {
      const input = processStreamingText(source.slice(0, length));
      const render = cache.update(input);
      assert.deepEqual(
        render.parseMarkdownIntoBlocks(render.markdown),
        parseMarkdownIntoBlocks(remend(asLf(input))),
        `block mismatch at prefix ${length} of ${JSON.stringify(source.slice(0, 60))}`,
      );
    }
  }
});

test("an LF reply is untouched by the line-ending handling", () => {
  // The guard on the other side: a reply with no carriage return in it must take
  // exactly the path it took before, so this cannot cost the common case
  // anything or move its output.
  const reply = buildReply(60);
  assert.ok(!reply.includes("\r"));
  const run = stream(reply, 24);
  const cache = new IncrementalMarkdownCache();
  for (let length = 0; length <= reply.length; length += 24) {
    const input = processStreamingText(reply.slice(0, length));
    const render = cache.update(input);
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(remend(input)),
      `block mismatch at prefix ${length}`,
    );
  }
  assert.ok(run.retained > reply.length * 0.8);
});
