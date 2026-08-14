// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";
import { parseMarkdownIntoBlocks, Streamdown } from "streamdown";

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
// Just past the rollback window, since the block list interleaves separators.
const SHORT_GAP = "\n\np0\n\np1\n\np2\n\np3\n\n";

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
  `\`\`\`sh\necho $HOME\n\`\`\`\n\n${paragraphs(12)}see *italic`,
  `\`\`\`text\ncost $$ each\n\`\`\`\n\n${paragraphs(12)}see *italic`,
  `use \`a *b\` here\n\n${paragraphs(12)}see *italic`,
  `use \`a _b\` here\n\n${paragraphs(12)}see _italic`,
  `Call _private first.\n\n${paragraphs(12)}Plain **bold`,
  `$$E=mc^2$$ is famous\n\n${paragraphs(12)}$$`,
  `A stray \` marker in prose.\n\n${paragraphs(12)}\`tail`,
  `A stray __ marker in prose.\n\n${paragraphs(12)}x__ y __tail`,
  `A stray ~~ marker in prose.\n\n${paragraphs(12)}x~~ y ~~tail`,
  `A stray $ marker in prose.\n\n${paragraphs(12)}$tail`,
  `\`\`\`text\n$$\n\`\`\`\n\n${paragraphs(12)}$$\nx`,
  `takes 5~10 minutes\n\n${paragraphs(20)}`,
  `- >= 16 GB of RAM\n\n${paragraphs(20)}`,
  // Remend completes a dangling link, or truncates at a dangling image, using
  // the end of the whole document.
  `Pick x in the interval [0, 1) for the ratio.\n\n${paragraphs(14)}done`,
  `see ![alt text\n\n${paragraphs(12)}later y](`,
  // A backtick that ends a retained block still closes inline code for remend.
  `Escape a backtick as \\\`\n\n${paragraphs(10)}$$\nE=mc^2\n$$`,
  // Remend orders its closers from a raw "**" search, fenced code included.
  `\`\`\`c\nchar **argv;\n\`\`\`\n\n${paragraphs(12)}set _flag to **on`,
  // Reduced shapes, one per whole-document rule the retained prefix has to
  // reproduce. Reported by @mahiatlinux on the PR.
  `\`\`\`x\`\`\`\`${SHORT_GAP}~~ y`,
  `see [note${SHORT_GAP}tail`,
  `\\\`${SHORT_GAP}$$`,
  `********${SHORT_GAP}*a`,
  `\`\`\`\n**\n\`\`\`${SHORT_GAP}_u then **v then **`,
  `use \`a *b* c\` here${SHORT_GAP}****x`,
  `a \`\`\` b\n\nc \\\`\`\` d${SHORT_GAP}- >= 4 GB`,
  `[x]: https://e.test\n\n${paragraphs(12)}[x]: https://e.test\n\nq\n\n`,
  `\`\`\`md\n[x]: https://e.test\n\`\`\`\n\n${paragraphs(12)}[x]: https://e.test\n\nq\n\n`,
  // Retained-prefix contexts that nothing else reaches: a balanced single
  // underscore, one first seen inside inline code, and an underscore that
  // precedes the first bold marker.
  `Use _snake_ case.${SHORT_GAP}see _italic`,
  `use \`a _b_ c\` here${SHORT_GAP}see _italic`,
  `Set _flag_ and **mode** now.${SHORT_GAP}see _italic and **bold`,
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

test("a non-prefix replacement clears the sticky fallback", () => {
  const cache = new IncrementalMarkdownCache();
  cache.update(`takes 5~10 minutes\n\n${paragraphs(100)}`);
  assert.equal(
    (cache as unknown as { fullDocumentMode: boolean }).fullDocumentMode,
    true,
  );

  const replacement = paragraphs(100, "replacement");
  const render = cache.update(replacement);
  assert.equal(
    (cache as unknown as { fullDocumentMode: boolean }).fullDocumentMode,
    false,
  );
  assert.ok(render.markdown.length < replacement.length / 4);
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

test("an edit that drops retained blocks moves the render identity", () => {
  // Deleting exactly the retained prefix leaves the live tail, and with it the
  // only thing Streamdown compares, unchanged.
  const source = paragraphs(30);
  const cache = new IncrementalMarkdownCache();
  const streamed = cache.update(source);
  assert.ok(streamed.markdown.length < source.length / 3);
  const streamedGeneration = cache.renderGeneration;

  const edited = cache.update(streamed.markdown);
  assert.equal(edited.markdown, streamed.markdown);
  assert.notEqual(cache.renderGeneration, streamedGeneration);
  assert.deepEqual(
    edited.parseMarkdownIntoBlocks(edited.markdown),
    parseMarkdownIntoBlocks(remend(streamed.markdown)),
  );

  // A reply that only grows never remounts, including while the stabilizer
  // withholds an ambiguous trailing line.
  const streaming = new IncrementalMarkdownCache();
  for (let length = 1; length <= source.length; length += 5) {
    streaming.update(source.slice(0, length));
    streaming.update(`${source.slice(0, length)}* **`);
  }
  assert.equal(streaming.renderGeneration, 0);
});

test("only a real duplicate link label gives up the retained prefix", () => {
  // Marked reads a definition inside a fenced example as code, so recording it
  // would strand the reply in the full-document path for the rest of the stream.
  const shown = `\`\`\`md\n[x]: https://e.test\n\`\`\`\n\n${paragraphs(30)}[x]: https://e.test\n\nend`;
  const shownCache = new IncrementalMarkdownCache();
  let render = shownCache.update("");
  for (let length = 1; length <= shown.length; length += 1) {
    render = shownCache.update(processStreamingText(shown.slice(0, length)));
  }
  assert.ok(render.markdown.length < shown.length / 3);

  const repeated = `[x]: https://e.test\n\n${paragraphs(30)}[x]: https://e.test\n\nend`;
  const repeatedCache = new IncrementalMarkdownCache();
  let repeatedRender = repeatedCache.update("");
  for (let length = 1; length <= repeated.length; length += 1) {
    repeatedRender = repeatedCache.update(
      processStreamingText(repeated.slice(0, length)),
    );
  }
  assert.deepEqual(
    repeatedRender.parseMarkdownIntoBlocks(repeatedRender.markdown),
    parseMarkdownIntoBlocks(remend(processStreamingText(repeated))),
  );
});

test("Streamdown re-renders only when the Markdown string changes", () => {
  // Its memo comparator is what decides whether the parser callback runs again,
  // and it does not compare that callback. Retaining a block therefore has to
  // change the Markdown too, or the retained block never reaches the DOM.
  const { compare } = Streamdown as unknown as {
    compare: (previous: object, next: object) => boolean;
  };
  const shared = { mode: "streaming", isAnimating: true, children: "reply" };

  assert.equal(
    compare(
      { ...shared, parseMarkdownIntoBlocksFn: () => [] },
      { ...shared, parseMarkdownIntoBlocksFn: () => [] },
    ),
    true,
  );
  assert.equal(compare(shared, { ...shared, children: "longer reply" }), false);
});

test("a repeating reply keeps displaying every retained block", () => {
  // A reply that repeats a line leaves the tail unchanged when an update
  // retains exactly what it appended, so the cache must not hand Streamdown a
  // Markdown string it already holds.
  const line = "I cannot provide that information.\n\n";
  const step = line.length;
  const source = `Here is the answer.\n\n${line.repeat(60)}`;
  const cache = new IncrementalMarkdownCache();
  let displayedMarkdown: string | null = null;
  let displayed: string[] = [];

  for (let length = step; length <= source.length; length += step) {
    const input = source.slice(0, length);
    const render = cache.update(input);
    if (render.markdown !== displayedMarkdown) {
      displayedMarkdown = render.markdown;
      displayed = render.parseMarkdownIntoBlocks(render.markdown);
    }
    assert.deepEqual(displayed, parseMarkdownIntoBlocks(remend(input)));
    // Retaining one update later must not let the live tail track the reply.
    assert.ok(render.markdown.length < step * 6);
  }
});
