// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createReasoningPageBoundary,
  isReasoningPageBoundaryValid,
  REASONING_PAGE_CHARACTERS,
  selectReasoningMarkdownPage,
} from "../src/components/assistant-ui/reasoning-pagination.ts";
import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import { getCompletedCodeFences } from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

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

test("page history stays stable on append and rejects retroactive rewrites", () => {
  const latest = selectReasoningMarkdownPage(richReasoning, { enabled: true });
  const boundary = createReasoningPageBoundary(richReasoning, latest.start);
  const earlier = selectReasoningMarkdownPage(richReasoning, {
    enabled: true,
    end: boundary.end,
  });
  const appended = `${richReasoning}\n- newly streamed tail`;
  const afterAppend = selectReasoningMarkdownPage(appended, {
    enabled: true,
    end: boundary.end,
  });

  assert.deepEqual(afterAppend, earlier);
  assert.equal(isReasoningPageBoundaryValid(appended, boundary), true);

  const unclosedSource =
    "p".repeat(5_000) + "\\[" + "x".repeat(88) + "\n\n" + "y".repeat(3_908);
  const unclosed = stabilizeStreamingMarkdown(
    preprocessLaTeX(unclosedSource),
    true,
  );
  const livePage = selectReasoningMarkdownPage(unclosed, {
    enabled: true,
    streaming: true,
  });
  const latexBoundary = createReasoningPageBoundary(unclosed, livePage.start);
  const closed = stabilizeStreamingMarkdown(
    preprocessLaTeX(`${unclosedSource}\\] tail`),
    true,
  );

  assert.ok(livePage.start > 0);
  assert.equal(isReasoningPageBoundaryValid(closed, latexBoundary), false);
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
      close: "]]>",
      markdown: `<![CDATA[\n${body}]]>`,
      open: "<![CDATA[\n",
    },
    {
      close: "?>",
      markdown: `<?pi\n${body}?>`,
      open: "<?\n",
    },
    {
      close: ">",
      markdown: `<!DECLARATION\n${body}>`,
      open: "<!A\n",
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

  const hugeDetails = selectReasoningMarkdownPage(
    `<details open data-x="${"x".repeat(100_000)}">\n${body}</details>`,
    { enabled: true, maxCharacters: 2_048 },
  );
  assert.ok(hugeDetails.markdown.length <= 2_304);
  assert.equal(hugeDetails.markdown.startsWith("<details open>\n"), true);
  assert.equal(hugeDetails.markdown.endsWith("</details>\n"), true);
});

test("containerized code and math retain wrappers across cuts", () => {
  const codeLines = Array.from({ length: 1_500 }, () => "print(1)");
  const code = codeLines.join("\n");
  const quotedFence = `> \`\`\`python\n${codeLines
    .map((line, index) => `${index % 2 === 0 ? "> " : ">"}${line}`)
    .join("\n")}\n> \`\`\`\n`;
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

  const listPage = selectReasoningMarkdownPage(
    `- lead\n${Array.from(
      { length: 3_000 },
      (_, index) => `  continuation ${index}: ${"text ".repeat(20)}`,
    ).join("\n")}`,
    { enabled: true, maxCharacters: 2_048 },
  );
  assert.equal(listPage.markdown.startsWith("-\n  continuation"), true);
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
    (_, index) =>
      index % 2 === 0 ? `| ${index} | ${"value ".repeat(4)} |` : `row ${index}`,
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

  const surrogateBoundary = `a😀${"b".repeat(8_191)}`;
  const unicodePage = selectReasoningMarkdownPage(surrogateBoundary, {
    enabled: true,
    maxCharacters: 8_192,
  });
  assert.equal(unicodePage.start, 3);
  assert.equal(unicodePage.markdown.startsWith("b"), true);
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

// Every page of `markdown`, oldest first.
function paginate(markdown: string, maxCharacters: number) {
  const pages = [
    selectReasoningMarkdownPage(markdown, {
      enabled: true,
      maxCharacters,
    }),
  ];
  while (pages[0].hasEarlier) {
    pages.unshift(
      selectReasoningMarkdownPage(markdown, {
        enabled: true,
        end: pages[0].start,
        maxCharacters,
      }),
    );
  }
  return pages;
}

const crlfCode = Array.from(
  { length: 800 },
  (_, index) => `const value${index} = ${index};`,
).join("\r\n");
const crlfFenceDocument = `intro paragraph\r\n\r\n\`\`\`typescript\r\n${crlfCode}\r\n\`\`\`\r\n`;

test("CRLF pages reopen the fence instead of spilling source as prose", () => {
  const pages = paginate(crlfFenceDocument, 2_048);

  assert.ok(pages.length > 3);
  assert.equal(pages[0].markdown.startsWith("intro paragraph\r\n"), true);
  for (const page of pages.slice(1)) {
    assert.equal(page.markdown.startsWith("```typescript\r\n"), true);
    assert.equal(page.markdown.endsWith("\r\n```\r\n"), true);
  }
});

test("a paginated CRLF fence keeps the exact bytes of the unpaginated source", () => {
  const [unpaginated] = getCompletedCodeFences(crlfFenceDocument);
  assert.equal(unpaginated.source, crlfCode);

  for (const page of paginate(crlfFenceDocument, 2_048)) {
    assert.deepEqual(page.canonicalCodeSources, [crlfCode]);
  }
});

test("an LF fence opener over a CRLF body copies every carriage return", () => {
  const body = Array.from({ length: 400 }, (_, index) => `x${index}`).join(
    "\r\n",
  );
  const markdown = `\`\`\`txt\n${body}\n\`\`\`\n`;
  const page = selectReasoningMarkdownPage(markdown, {
    enabled: true,
    maxCharacters: 512,
  });
  const [source] = page.canonicalCodeSources;

  assert.equal(source, body);
  assert.equal((source ?? "").split("\r\n").length - 1, 399);
});

test("CRLF pages preserve display math and quoted fences across cuts", () => {
  const math = Array.from(
    { length: 1_500 },
    (_, index) => `x_{${index}} = ${index}`,
  ).join("\r\n");
  for (const page of paginate(`$$$\r\n${math}\r\n$$$\r\n`, 2_048)) {
    assert.equal(page.markdown.startsWith("$$$\r\n"), true);
    assert.equal(page.markdown.endsWith("\r\n$$$\r\n"), true);
  }

  const codeLines = Array.from({ length: 1_500 }, () => "print(1)");
  const quoted = `> \`\`\`python\r\n${codeLines
    .map((line, index) => `${index % 2 === 0 ? "> " : ">"}${line}`)
    .join("\r\n")}\r\n> \`\`\`\r\n`;
  for (const page of paginate(quoted, 2_048)) {
    assert.equal(page.markdown.startsWith("> ```python\r\n"), true);
    assert.equal(page.markdown.endsWith("\r\n> ```\r\n"), true);
    assert.deepEqual(page.canonicalCodeSources, [codeLines.join("\r\n")]);
  }
});

test("CRLF pages partition the source byte for byte", () => {
  const table = `| step | result |\r\n| ---: | :--- |\r\n${Array.from(
    { length: 400 },
    (_, index) => `| ${index} | ${"value ".repeat(4)} |`,
  ).join("\r\n")}\r\n`;
  const html = `<details open>\r\n${"hidden line\r\n".repeat(400)}</details>\r\n`;
  const documents = [
    crlfFenceDocument,
    table,
    html,
    `${table}\r\n${crlfFenceDocument}\r\n${html}`,
    "unterminated ```js\r\n".repeat(300),
  ];

  for (const markdown of documents) {
    for (const maxCharacters of [512, 2_048]) {
      const pages = paginate(markdown, maxCharacters);
      assert.equal(pages[0].start, 0);
      assert.equal(pages[pages.length - 1].end, markdown.length);
      for (const [index, page] of pages.slice(1).entries()) {
        assert.equal(page.start, pages[index].end);
      }
      assert.equal(
        pages.map((page) => markdown.slice(page.start, page.end)).join(""),
        markdown,
      );
    }
  }
});
