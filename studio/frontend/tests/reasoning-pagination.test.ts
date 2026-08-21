// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  REASONING_PAGE_CHARACTERS,
  selectReasoningMarkdownPage,
} from "../src/components/assistant-ui/reasoning-pagination.ts";

const richReasoning = Array.from(
  { length: 3_000 },
  (_, index) => `- item ${index}: ${"rich reasoning ".repeat(8)}`,
).join("\n");

test("the latest reasoning page is bounded at a Markdown block boundary", () => {
  const page = selectReasoningMarkdownPage(richReasoning, {
    enabled: true,
  });

  assert.equal(page.end, richReasoning.length);
  assert.ok(page.start > 0);
  assert.ok(page.markdown.length <= REASONING_PAGE_CHARACTERS);
  assert.match(page.markdown, /^- item \d+:/);
  assert.equal(richReasoning.slice(page.start, page.end), page.markdown);
  assert.equal(page.hasEarlier, true);
  assert.equal(page.hasNewer, false);
});

test("earlier pages replace the mounted page instead of accumulating", () => {
  let page = selectReasoningMarkdownPage(richReasoning, { enabled: true });
  const mountedLengths: number[] = [page.markdown.length];
  let pages = 1;

  while (page.hasEarlier) {
    const newerStart = page.start;
    page = selectReasoningMarkdownPage(richReasoning, {
      enabled: true,
      end: newerStart,
    });
    mountedLengths.push(page.markdown.length);
    assert.equal(page.end, newerStart);
    assert.equal(page.hasNewer, true);
    pages += 1;
  }

  assert.ok(pages > 20);
  assert.equal(page.start, 0);
  assert.ok(
    mountedLengths.every((length) => length <= REASONING_PAGE_CHARACTERS),
  );
  assert.ok(mountedLengths.every((length) => length < richReasoning.length));
});
test("the live page advances in stable overlapping strides", () => {
  const markdown = Array.from(
    { length: 2_000 },
    (_, index) => `line ${index}: ${"streaming ".repeat(6)}`,
  ).join("\n");
  const first = selectReasoningMarkdownPage(markdown.slice(0, 9_000), {
    enabled: true,
    streaming: true,
  });
  const appended = selectReasoningMarkdownPage(markdown.slice(0, 11_000), {
    enabled: true,
    streaming: true,
  });
  const advanced = selectReasoningMarkdownPage(markdown.slice(0, 13_000), {
    enabled: true,
    streaming: true,
  });

  assert.equal(appended.start, first.start);
  assert.ok(advanced.start > appended.start);
  for (const page of [first, appended, advanced]) {
    assert.ok(page.markdown.length <= REASONING_PAGE_CHARACTERS);
    assert.ok(page.markdown.length >= REASONING_PAGE_CHARACTERS / 2 - 100);
  }
});

test("a nearby blank line keeps the page on a Markdown block boundary", () => {
  const markdown = `${"prefix".repeat(200)}\n\n**complete paragraph**\n\n${"tail".repeat(2_000)}`;
  const page = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 8_192,
  });

  assert.match(page.markdown, /^\*\*complete paragraph\*\*/);
  assert.ok(page.markdown.length <= 8_192);
});

test("a fixed earlier page remains stable while new reasoning streams", () => {
  const latest = selectReasoningMarkdownPage(richReasoning, { enabled: true });
  const earlier = selectReasoningMarkdownPage(richReasoning, {
    enabled: true,
    end: latest.start,
  });
  const afterAppend = selectReasoningMarkdownPage(
    `${richReasoning}\n- newly streamed tail`,
    {
      enabled: true,
      end: latest.start,
    },
  );

  assert.deepEqual(afterAppend, earlier);
});

test("pages cut a giant fence safely instead of mounting the whole fence", () => {
  const prefix = `${"old paragraph\n\n".repeat(100)}\n`;
  const fence = `~~~typescript\n${"const value = 1;\n".repeat(2_000)}~~~\n`;
  const markdown = prefix + fence;
  const latest = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 2_048,
  });

  assert.ok(latest.start > prefix.length);
  assert.ok(latest.markdown.length <= 2_304);
  assert.match(latest.markdown, /^~~~typescript\n/);
  assert.match(latest.markdown, /~~~\n?$/);

  const earlier = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    end: latest.start,
    maxCharacters: 2_048,
  });
  assert.ok(earlier.markdown.length <= 2_304);
  assert.match(earlier.markdown, /^~~~typescript\n/);
  assert.match(earlier.markdown, /~~~\n?$/);
});

test("a paginated fence retains its complete canonical source for code actions", () => {
  const source = "const preserved = true;\n".repeat(2_000).trimEnd();
  const markdown = `~~~typescript\n${source}\n~~~\n`;
  const page = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 2_048,
  });

  assert.deepEqual(page.canonicalCodeSources, [source]);
});

test("a page inside a raw pre block stays literal Markdown", () => {
  const markdown = `<pre>\n${"literal html line\n".repeat(1_000)}~~~js\n${"not a fence\n".repeat(1_000)}</pre>\n`;
  const page = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 2_048,
  });

  assert.match(page.markdown, /^<pre>\n/);
  assert.match(page.markdown, /<\/pre>\n?$/);

  assert.deepEqual(page.canonicalCodeSources, []);
});

test("a giant single line is hard-bounded", () => {
  const markdown = "continuous reasoning ".repeat(20_000);
  const page = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 8_192,
  });

  assert.equal(page.markdown.length, 8_192);
  assert.equal(page.start, markdown.length - 8_192);
});

test("short and non-reasoning Markdown remain exact", () => {
  const short = "# Thought\n\nStill working";
  assert.deepEqual(selectReasoningMarkdownPage(short, { enabled: true }), {
    canonicalCodeSources: [],
    end: short.length,
    hasEarlier: false,
    hasNewer: false,
    markdown: short,
    start: 0,
  });
  assert.deepEqual(
    selectReasoningMarkdownPage(richReasoning, { enabled: false }),
    {
      canonicalCodeSources: [],
      end: richReasoning.length,
      hasEarlier: false,
      hasNewer: false,
      markdown: richReasoning,
      start: 0,
    },
  );
});
