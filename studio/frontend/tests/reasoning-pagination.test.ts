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
test("opaque and block HTML keep inner Markdown inert across cuts", () => {
  const body = `$$\n${"literal html line\n".repeat(1_000)}~~~js\n`;
  const pages = [
    {
      close: "-->",
      markdown: `<!--\n${body}-->`,
      open: "<!--\n",
    },
    {
      close: "</details>\n",
      markdown: `<details open>\n${body}</details>`,
      open: "<details open>\n",
    },
  ].map(({ close, markdown, open }) => ({
    close,
    open,
    page: selectReasoningMarkdownPage(markdown, {
      enabled: true,
      maxCharacters: 2_048,
    }),
  }));

  for (const { close, open, page } of pages) {
    assert.equal(page.markdown.startsWith(open), true);
    assert.equal(page.markdown.endsWith(close), true);
    assert.deepEqual(page.canonicalCodeSources, []);
  }
});

test("containerized code and math retain wrappers across cuts", () => {
  const code = Array.from({ length: 1_500 }, () => "print(1)").join("\n");
  const quotedFence = `> \`\`\`python\n> ${code.replaceAll("\n", "\n> ")}\n> \`\`\`\n`;
  const codePage = selectReasoningMarkdownPage(quotedFence, {
    enabled: true,
    maxCharacters: 2_048,
  });
  assert.equal(codePage.markdown.startsWith("> ```python\n"), true);
  assert.equal(codePage.markdown.endsWith("\n> ```\n"), true);
  assert.deepEqual(codePage.canonicalCodeSources, [code]);

  const math = Array.from(
    { length: 1_500 },
    (_, index) => `x_{${index}} = ${index}`,
  ).join("\n  ");
  const listMath = `- $$$\n  ${math}\n  $$$\n`;
  const mathPage = selectReasoningMarkdownPage(listMath, {
    enabled: true,
    maxCharacters: 2_048,
  });
  assert.equal(mathPage.markdown.startsWith("- $$$\n"), true);
  assert.equal(mathPage.markdown.endsWith("\n  $$$\n"), true);
  const topLevel = "top-level prose\n".repeat(1_000);
  for (const opening of [
    "> ```python\n> print(1)",
    "> $$\n> x = 1",
    "> <details open>\n> hidden",
  ]) {
    const page = selectReasoningMarkdownPage(`${opening}\n${topLevel}`, {
      enabled: true,
      maxCharacters: 2_048,
    });
    assert.equal(page.markdown.startsWith("top-level prose\n"), true);
  }
  const transitioned = selectReasoningMarkdownPage(
    `> <details open>\n> hidden\n\`\`\`js\n${"const x = 1;\n".repeat(1_000)}`,
    { enabled: true, maxCharacters: 2_048 },
  );
  assert.equal(transitioned.markdown.startsWith("```js\n"), true);
});

test("pages preserve display math across cuts", () => {
  const markdown = `$$$\n${Array.from(
    { length: 1_500 },
    (_, index) => `x_{${index}} = ${index}`,
  ).join("\n")}\n$$$\n`;
  const latest = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 2_048,
  });
  const earlier = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    end: latest.start,
    maxCharacters: 2_048,
  });

  for (const page of [latest, earlier]) {
    assert.equal(page.markdown.startsWith("$$$\n"), true);
    assert.equal(page.markdown.endsWith("\n$$$\n"), true);
  }
});
test("long GFM tables retain their header across page cuts", () => {
  const header = "| step | result |";
  const delimiter = "| ---: | :--- |";
  const markdown = `${header}\n${delimiter}\n${Array.from(
    { length: 3_000 },
    (_, index) => `| ${index} | ${"value ".repeat(4)} |`,
  ).join("\n")}\n`;
  const latest = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 2_048,
  });
  const earlier = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    end: latest.start,
    maxCharacters: 2_048,
  });

  for (const page of [latest, earlier]) {
    assert.equal(page.markdown.startsWith(`${header}\n${delimiter}\n`), true);
    assert.ok(page.markdown.length <= 2_100);
  }
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
