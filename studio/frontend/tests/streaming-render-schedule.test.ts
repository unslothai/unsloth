// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import {
  IncrementalMarkdownCache,
  withoutStreamdownAnimationPlugin,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

test("only Streamdown's animation transformer is removed", () => {
  const first = () => undefined;
  const animation = () => undefined;
  const configured: [typeof first, { enabled: boolean }] = [
    first,
    {
      enabled: true,
    },
  ];
  const plugins = [first, animation, configured];

  assert.deepEqual(
    withoutStreamdownAnimationPlugin(plugins, {
      name: "animate",
      type: "animate",
      rehypePlugin: animation,
      getLastRenderCharCount: () => 0,
      setPrevContentLength: () => undefined,
    }),
    [first, configured],
  );
});

const paragraphs = (count: number, label = "part") =>
  Array.from({ length: count }, (_, index) => `${label} ${index}\n\n`).join("");
const retainedCase = (fragment: string) =>
  `${paragraphs(12, "before")} ${fragment}\n\n${paragraphs(12, "after")}`;

const MARKDOWN_CASES = [
  retainedCase(
    "# Heading\n\nParagraph with **bold** and [link](https://x.test).",
  ),
  retainedCase("Before\n\n```ts\nconst x = 1;\n```\n\nAfter"),
  retainedCase("Math \\(x+1\\)\n\n\\[a=b\\]\n\nend"),
  retainedCase("> quote\n> next\n\nparagraph\n\n---\n\nend"),
  retainedCase("<div>\ninside\n\nmore\n</div>\n\nafter"),
  retainedCase("A [ref][x]\n\n[x]: https://example.com"),
  retainedCase("one\n\n$$\na+b\n$$\n\ntwo"),
  retainedCase("a\n\n* item\n  * nested\n\nb"),
  paragraphs(30),
  `${paragraphs(20)}term[^note]\n\n[^note]: detail`,
  `$$\nx+y\n$$\n\n${paragraphs(20)}$$\nz`,
  `In Python 2 ** 3 is eight.\n\n${paragraphs(12)}x** y **bold`,
  `Match *.py files here.\n\n${paragraphs(12)}x *y and *italic`,
  `Call _private first.\n\n${paragraphs(12)}x _y and _italic`,
  `A ***** marker run.\n\n${paragraphs(12)}x ***y and ***bold`,
  `Plain **bold** and *italic*.\n\n$$\nx+y\n$$\n\n${paragraphs(12)}A ***** marker.\n\n${paragraphs(4, "later")}\`\`\``,
  `Plain **bold** and *italic*.\n\n$$\nx+y\n$$\n\n${paragraphs(12)}Call _private first.\n\n${paragraphs(4, "later")}Plain **b`,
  `A stray \` marker in prose.\n\n${paragraphs(12)}\`tail`,
  `A stray __ marker in prose.\n\n${paragraphs(12)}x__ y __tail`,
  `A stray ~~ marker in prose.\n\n${paragraphs(12)}x~~ y ~~tail`,
  `A stray $ marker in prose.\n\n${paragraphs(12)}$tail`,
  `\`\`\`text\n$$\n\`\`\`\n\n${paragraphs(12)}$$\nx`,
  `takes 5~10 minutes\n\n${paragraphs(20)}`,
  `- >= 16 GB of RAM\n\n${paragraphs(20)}`,
];

const processStreamingText = (text: string): string =>
  stabilizeStreamingMarkdown(preprocessLaTeX(text), true);

test("incremental blocks match a full Streamdown split at every prefix", () => {
  for (const source of MARKDOWN_CASES) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 0; length <= source.length; length += 1) {
      const input = processStreamingText(source.slice(0, length));
      const render = cache.update(input);
      assert.deepEqual(
        render.parseMarkdownIntoBlocks(render.markdown),
        parseMarkdownIntoBlocks(remend(input)),
        `block mismatch at prefix ${length} of ${JSON.stringify(source)}`,
      );
    }
  }
});

test("incremental parsing bounds the live tail and resets after an edit", () => {
  const source = Array.from(
    { length: 100 },
    (_, index) => `paragraph ${index}\n\n`,
  ).join("");
  const cache = new IncrementalMarkdownCache();
  const streamed = cache.update(source);
  assert.ok(streamed.markdown.length < source.length / 4);

  const edited = source.replace("paragraph 0", "changed 0");
  const reset = cache.update(edited);
  assert.deepEqual(
    reset.parseMarkdownIntoBlocks(reset.markdown),
    parseMarkdownIntoBlocks(remend(edited)),
  );
});

test("mid-string remend repairs use the sticky full-document fallback", () => {
  for (const firstBlock of ["takes 5~10 minutes", "- >= 16 GB of RAM"]) {
    const cache = new IncrementalMarkdownCache();
    const source = `${firstBlock}\n\n${paragraphs(100)}`;
    const first = cache.update(source);
    assert.equal(first.markdown, remend(source));
    assert.equal(
      (cache as unknown as { fullDocumentMode: boolean }).fullDocumentMode,
      true,
    );

    const appended = `${source}one more paragraph\n\n`;
    const next = cache.update(appended);
    assert.equal(next.markdown, remend(appended));
  }
});

test("a transient marker imbalance can recover incremental parsing", () => {
  const cache = new IncrementalMarkdownCache();
  const unbalanced = `Match *.py files here.\n\n${paragraphs(20)}`;
  cache.update(unbalanced);
  assert.equal(
    (cache as unknown as { fullDocumentMode: boolean }).fullDocumentMode,
    false,
  );

  const balanced = `${unbalanced}Finish the *italic example.\n\n${paragraphs(20, "later")}`;
  const render = cache.update(balanced);
  assert.ok(render.markdown.length < balanced.length / 2);
});

test("single-dollar parity affects the retained tail repair", () => {
  const source = `A $ marker in prose.\n\n${paragraphs(12)}x$ y $tail`;
  const cache = new IncrementalMarkdownCache();
  for (let length = 0; length <= source.length; length += 1) {
    const input = source.slice(0, length);
    const render = cache.update(input);
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(remend(input)),
    );
  }
});

test("a code character class does not disable incremental parsing", () => {
  const source = `\`\`\`js\nconst token = /[^\\s]+/;\n\`\`\`\n\n${paragraphs(100)}`;
  const render = new IncrementalMarkdownCache().update(source);
  assert.ok(render.markdown.length < source.length / 4);
});
