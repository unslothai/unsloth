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

// A rebuild produces the same block list it discarded, so only the counter
// records that it happened.
const rebuilds = (cache: IncrementalMarkdownCache): number =>
  (cache as unknown as { retainedPrefixRebuilds: number })
    .retainedPrefixRebuilds;

// Characters a rewind gave back to the live tail. Same problem: the next update
// re-commits the same boundary, so the extra work leaves no trace in the output.
const rewound = (cache: IncrementalMarkdownCache): number =>
  (cache as unknown as { rewoundCharacters: number }).rewoundCharacters;

// `parseMarkdownIntoBlocks("")` splits to nothing, so passing an empty string
// returns exactly the blocks the cache has retained.
const retainedBlocks = (render: {
  parseMarkdownIntoBlocks: (markdown: string) => string[];
}): string[] => render.parseMarkdownIntoBlocks("");

// Prose, inline math, display math, a list, a fence and currency: what an
// answer to a modelling question actually looks like.
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

// Control arm: every rewritten construct spelled out, so the pipeline output
// only ever grows.
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
  // `\(...\)` -> `$...$`, `\[...\]` -> a `$$` block and `$1,200` -> `\$1,200`
  // each rewrite text an earlier frame emitted, but all land in the live tail,
  // so the prefix has to survive every one.
  const reply = buildReply(120);
  assert.ok(reply.length > 8_000, `fixture too small: ${reply.length}`);

  const { cache, frames, retained } = streamReply(reply, 24);
  assert.ok(frames > 300, `expected a long stream, got ${frames}`);
  assert.equal(
    rebuilds(cache),
    0,
    `retained prefix rebuilt ${rebuilds(cache)} times over ${frames} frames`,
  );
  // Never rebuilding also looks like never retaining, which would re-lex the
  // whole reply every frame. Pin the retention too.
  assert.ok(
    retained > reply.length * 0.8,
    `retained only ${retained} of ${reply.length} characters`,
  );

  // Control: a discard here would mean the first assertion measures the
  // fixture, not the rewrite.
  const plain = streamReply(plainVariant(reply), 24);
  assert.equal(rebuilds(plain.cache), 0);
  assert.ok(plain.retained > reply.length * 0.8);
});

test("a rewrite behind the live tail keeps the blocks it cannot reach", () => {
  // `LATEX_DELIM_RE` lets an inline span run 4,096 characters across newlines,
  // so a `\(` closing several paragraphs later rewrites already committed text.
  // Everything before the opener is untouched and has to survive.
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
  // Nothing the rewrite reaches may survive, so what is kept is a prefix of the
  // text before the opener, never more.
  assert.ok(
    lead.startsWith(atClose),
    `kept ${atClose.length} characters, past the ${lead.length} the rewrite ` +
      "cannot reach",
  );
  // Giving back the whole prefix is what a rebuild looks like from outside. The
  // rewind stops a rollback window short of the change, so the bound is the text
  // before the opener minus that window. Only the counter shows over-rewinding:
  // the same update re-commits what the retained prefix gave back.
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
  // A closing fence turns its body into code, so a `$1` inside stops being
  // escaped. An open fence lexes to a single live block, so the opener is never
  // behind the commit boundary and this rewind keeps everything.
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
  // The output guard: whatever is retained, the block list handed to Streamdown
  // must match a whole-document parse. Catches a rewind keeping a block the edit
  // reached. The third shape makes the rewind drop blocks rather than just
  // reseat the tail, which is the branch with something to get wrong.
  const spanning = `${Array.from(
    { length: 12 },
    (_, index) => `Lead paragraph ${index}.\n\n`,
  ).join("")}\\(${Array.from(
    { length: 12 },
    (_, index) => `span line ${index}\n\n`,
  ).join("")}\\) done\n\n`;

  // Two spans in a row, so a second rewind lands on commit points the first
  // already trimmed.
  const twice = `${spanning}${spanning}`;

  // A `\[...\]` whose body spans blank lines. Closing it emits `\n$$\n`, whose
  // leading newline lands against the blank line before the opener, and Marked
  // reads a run of blank lines as ONE separator block, so the block before the
  // rewrite is re-segmented though its characters never moved. The body needs
  // enough chunks for that separator to have been committed at all.
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
        parseMarkdownIntoBlocks(remend(input)),
        `block mismatch at prefix ${length}`,
      );
    }
  }
});

test("a rewind restores the repair context of the commit it lands on", () => {
  // Rewinding the block list is only half of it: `advanceContext` accumulates
  // the emphasis markers the prefix carries into the tail repair and cannot be
  // undone, so the commit's own context has to come back too. The span body
  // opens single underscores, which makes the context after the rewind differ
  // from the one before the span.
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
      parseMarkdownIntoBlocks(remend(input)),
      `block mismatch at prefix ${length}`,
    );
  }
});

test("an edit that closes up a blank line cannot keep the block before it", () => {
  // Unchanged characters are not enough to keep a block: Marked reads
  // `paragraph 0\n` plus new text as a lazy continuation, so an edit at exactly
  // a commit boundary re-segments the paragraph before it though that
  // paragraph's characters never moved. `preprocessLaTeX` cannot produce this
  // today (its rewrites diverge on a `\` or `$`, neither of which joins two
  // lines), but nothing downstream owes us that, so the rule stands alone.
  const paragraphs = Array.from(
    { length: 60 },
    (_, index) => `paragraph ${index}\n\n`,
  ).join("");
  const quoted = `> quote line\n\n${Array.from(
    { length: 40 },
    (_, index) => `body ${index}\n\n`,
  ).join("")}`;

  const cases: Array<[string, string]> = [
    // Replace the first newline of the blank line after `paragraph 0`, whose
    // commit boundary sits at exactly that offset.
    [paragraphs, `${paragraphs.slice(0, 11)}!${paragraphs.slice(12)}`],
    // The same shape where the lazy continuation runs into a blockquote.
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
      parseMarkdownIntoBlocks(remend(edited)),
      `block mismatch after ${JSON.stringify(edited.slice(0, 24))}`,
    );
  }
});

test("a reply with math streams near the cost of one without", () => {
  // The rebuild shows only as time, so pin the time too. The arms are
  // interleaved and the figure is a ratio, because absolute milliseconds move
  // with whatever else the host runs. The rebuild costs the whole reply, so the
  // gap only opens with length: at 4,600 characters it is 1.7x and proves
  // nothing.
  const reply = buildReply(300);
  assert.ok(reply.length > 20_000, `fixture too small: ${reply.length}`);
  const plain = plainVariant(reply);
  const mathTimes: number[] = [];
  const plainTimes: number[] = [];

  // Untimed warmup. The math arm runs first every repeat, so without this it
  // pays the JIT cost for both and reads about 1.7x instead of about 1.2x.
  streamReply(reply, 24);
  streamReply(plain, 24);

  for (let repeat = 0; repeat < 5; repeat += 1) {
    let started = performance.now();
    const math = streamReply(reply, 24);
    mathTimes.push(performance.now() - started);
    started = performance.now();
    streamReply(plain, 24);
    plainTimes.push(performance.now() - started);
    // Both arms must be doing the retained-prefix work, or the ratio compares
    // two equally slow paths.
    assert.ok(math.retained > reply.length * 0.8);
  }

  // The minimum is the un-preempted cost, the one comparable figure on a host
  // running other work.
  const fastest = (values: number[]): number => Math.min(...values);
  const ratio = fastest(mathTimes) / fastest(plainTimes);
  // Measured 9.8x before the rewind and 1.1x-1.2x after, so a threshold of 4
  // has room in both directions on a loaded host.
  assert.ok(
    ratio < 4,
    `math reply cost ${ratio.toFixed(1)}x the plain reply ` +
      `(math ${fastest(mathTimes).toFixed(0)} ms, plain ${fastest(plainTimes).toFixed(0)} ms)`,
  );
});
