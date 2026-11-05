// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The retrieved passages a document search shows on request (#13788): in the order the model received
// them, numbered as it saw them, opening at their page, long ones clamped, and capped like the document chips.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import * as citationUtils from "../src/components/assistant-ui/citation-utils.ts";
import type * as PassagesModule from "../src/components/assistant-ui/knowledge-base-passages.tsx";
import { attachmentFileKind } from "../src/features/chat/lib/attachment-file-kind.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const {
  parseCitations,
  passageDisplayText,
  orderedPassages,
  passageNeedsClamp,
} = citationUtils;

const { KnowledgeBasePassages } = loadWithStubs<typeof PassagesModule>(
  new URL(
    "../src/components/assistant-ui/knowledge-base-passages.tsx",
    import.meta.url,
  ),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "./citation-utils": citationUtils,
    "@/features/chat": { attachmentFileKind },
    "@/lib/file-icon": { FileGlyph: () => null },
    // Marks where each passage opens, so the markup shows the document and page it was given.
    "@/features/rag": {
      RagDocumentTrigger: ({
        documentId,
        page,
        children,
      }: {
        documentId: string;
        page?: number;
        children: React.ReactNode;
      }) =>
        React.createElement(
          "span",
          { "data-opens": `${documentId}@${page ?? "start"}` },
          children,
        ),
    },
  },
);

// A search result as the backend stores it on the tool call: rendered chunks, then the source map.
const RADIUM =
  "Chapter 3: Radiation limits. The maximum allowed level of combined radium-226 and\nradium-228 in drinking water is 5 picocuries per liter (pCi/L).";
const RERUN =
  "Any result above 4 pCi/L is re-run from the retained duplicate before it is reported.";
const SCOPE =
  "This handbook sets out how the Authority samples and tests drinking water.";
const searchResult = (rows: unknown[]) =>
  `<chunk id="1">…</chunk>\n__RAG_SOURCES__:${JSON.stringify(rows)}`;
const SEARCH = searchResult([
  {
    citationId: 1,
    chunkId: "h:2",
    documentId: "h",
    filename: "handbook.pdf",
    page: 3,
    text: RADIUM,
    score: 0.0328,
  },
  {
    citationId: 2,
    chunkId: "l:1",
    documentId: "l",
    filename: "lab.md",
    page: null,
    text: RERUN,
    score: 0.0323,
  },
  {
    citationId: 3,
    chunkId: "h:0",
    documentId: "h",
    filename: "handbook.pdf",
    page: 1,
    text: SCOPE,
    score: 0.032,
  },
  {
    citationId: 4,
    chunkId: "h:9",
    documentId: "h",
    filename: "handbook.pdf",
    page: 5,
    text: "  \n ",
    score: 0.03,
  },
]);

const render = (result: unknown) =>
  renderToStaticMarkup(
    React.createElement(KnowledgeBasePassages, {
      citations: parseCitations(result),
    }),
  );

test("passages keep the search's order, numbered as the model saw them, across documents", () => {
  const passages = orderedPassages(parseCitations(SEARCH));
  assert.deepEqual(
    passages.map((p) => [p.number, p.citation.filename, p.citation.page]),
    [
      [1, "handbook.pdf", 3],
      [2, "lab.md", null],
      [3, "handbook.pdf", 1],
    ],
  );
});

test("an empty chunk is not a passage, and the passages after it keep their own numbers", () => {
  const passages = orderedPassages(
    parseCitations(
      searchResult([
        {
          chunkId: "a:0",
          documentId: "a",
          filename: "a.pdf",
          page: 1,
          text: "first",
          score: 0.9,
        },
        {
          chunkId: "a:1",
          documentId: "a",
          filename: "a.pdf",
          page: 2,
          text: " \n ",
          score: 0.8,
        },
        {
          chunkId: "a:2",
          documentId: "a",
          filename: "a.pdf",
          page: 3,
          text: "third",
          score: 0.7,
        },
      ]),
    ),
  );
  assert.deepEqual(
    passages.map((p) => [p.number, p.citation.text]),
    [
      [1, "first"],
      [3, "third"],
    ],
  );
});

// The backend appends ranked project passages after a whole-document read under one numbering
// (render_sources), and the model cites them by that numbering.
test("passages merged after a whole-document read continue its numbering", () => {
  const passages = orderedPassages(
    parseCitations(
      searchResult([
        {
          chunkId: "a:0",
          documentId: "a",
          filename: "a.txt",
          page: null,
          text: "whole 1",
          score: null,
        },
        {
          chunkId: "a:1",
          documentId: "a",
          filename: "a.txt",
          page: null,
          text: "whole 2",
          score: null,
        },
        {
          chunkId: "p:4",
          documentId: "p",
          filename: "p.pdf",
          page: 4,
          text: "best match",
          score: 0.033,
        },
        {
          chunkId: "p:9",
          documentId: "p",
          filename: "p.pdf",
          page: 9,
          text: "next match",
          score: 0.031,
        },
      ]),
    ),
  );
  assert.deepEqual(
    passages.map((p) => [p.number, p.citation.text]),
    [
      [1, "whole 1"],
      [2, "whole 2"],
      [3, "best match"],
      [4, "next match"],
    ],
  );
});

test("each passage shows its exact text, its number, and opens its document at its own page", () => {
  const html = render(SEARCH);
  assert.equal(html.match(/data-slot="knowledge-base-passage"/g)?.length, 3);
  // The chunk text itself, line breaks and spacing kept for whitespace-pre-wrap to show.
  assert.ok(html.includes("radium-226 and\nradium-228"));
  assert.ok(html.includes(RERUN) && html.includes(SCOPE));
  assert.ok(html.includes("#1") && html.includes("#2") && html.includes("#3"));
  // The answer's passage opens at its page, not at the document's first cited page.
  assert.match(html, /data-opens="h@3"><button[^>]*>Open at p\.3</);
  assert.match(html, /data-opens="h@1"><button[^>]*>Open at p\.1</);
  // A file without pages still opens, from the start.
  assert.match(html, /data-opens="l@start"><button[^>]*>Open document</);
  // Rendered in the search's order, so lab.md's #2 sits between handbook.pdf's #1 and #3.
  const numbers = [
    ...html.matchAll(/model was given">#(?:<!-- -->)?(\d+)</g),
  ].map((m) => Number(m[1]));
  assert.deepEqual(numbers, [1, 2, 3]);
  // With more than one document, every passage names its own.
  assert.equal(html.match(/title="handbook\.pdf"/g)?.length, 2);
  assert.equal(html.match(/title="lab\.md"/g)?.length, 1);
});

test("a single-document search leaves the filename to its chip", () => {
  const html = render(
    searchResult([
      {
        chunkId: "h:2",
        documentId: "h",
        filename: "handbook.pdf",
        page: 3,
        text: RADIUM,
        score: 0.9,
      },
      {
        chunkId: "h:0",
        documentId: "h",
        filename: "handbook.pdf",
        page: 1,
        text: SCOPE,
        score: 0.8,
      },
    ]),
  );
  assert.equal(html.match(/data-slot="knowledge-base-passage"/g)?.length, 2);
  assert.ok(!html.includes('title="handbook.pdf"'));
});

test("a source with no document id shows its page but offers nothing to open", () => {
  const html = render([
    { text: "legacy passage", filename: "old.pdf", page: 2, score: 0.5 },
  ]);
  assert.ok(html.includes("legacy passage") && html.includes("p.2"));
  assert.ok(!html.includes("data-opens") && !html.includes("Open "));
});

test("a long passage starts clamped behind Show more; a short one shows whole", () => {
  assert.equal(passageNeedsClamp("short"), false);
  // Under about four lines at text-xs in the chat column, a passage shows whole.
  assert.equal(passageNeedsClamp("x".repeat(450)), false);
  assert.equal(passageNeedsClamp("x".repeat(451)), true);
  assert.equal(passageNeedsClamp("a\nb\nc\nd\ne"), true);
  const long = render(
    searchResult([
      {
        chunkId: "d:0",
        documentId: "d",
        filename: "d.pdf",
        page: 1,
        text: "word ".repeat(120),
        score: 1,
      },
    ]),
  );
  assert.match(long, /line-clamp-4/);
  assert.match(long, /aria-expanded="false"[^>]*>Show more</);
  const short = render(SEARCH);
  assert.ok(!short.includes("line-clamp-4") && !short.includes("Show more"));
});

test("a large search shows the first 20 passages and holds the rest behind Show N more", () => {
  const rows = Array.from({ length: 23 }, (_, i) => ({
    chunkId: `d${i % 2}:${i}`,
    documentId: `d${i % 2}`,
    filename: `doc${i % 2}.pdf`,
    page: i + 1,
    text: `passage ${i + 1}`,
    score: 1 - i / 100,
  }));
  const html = render(searchResult(rows));
  assert.equal(html.match(/data-slot="knowledge-base-passage"/g)?.length, 20);
  // The first 20 in the search's order, across both documents.
  const numbers = [
    ...html.matchAll(/model was given">#(?:<!-- -->)?(\d+)</g),
  ].map((m) => Number(m[1]));
  assert.deepEqual(
    numbers,
    Array.from({ length: 20 }, (_, i) => i + 1),
  );
  assert.match(html, />Show 3 more passages</);
});

test("the search card keeps one chip per document and opens the passages only on request", () => {
  const kb = readFileSync(
    new URL(
      "../src/components/assistant-ui/tool-ui-knowledge-base.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    kb,
    /const \[showPassages, setShowPassages\] = useState\(false\)/,
  );
  assert.match(kb, /aria-expanded=\{showPassages\}/);
  assert.match(
    kb,
    /\{showPassages \? \(\s*<KnowledgeBasePassages citations=\{citations\} \/>/,
  );
  assert.match(kb, /`Show passages \(\$\{passageCount\}\)`/);
});

test("a PDF passage drops the heading markers its conversion adds; every passage folds only empty lines", () => {
  assert.equal(
    passageDisplayText(
      "# Chapter 3: Radiation limits\n\nThe limit is 5 pCi/L.",
      "standard.pdf",
    ),
    "Chapter 3: Radiation limits\nThe limit is 5 pCi/L.",
  );
  assert.equal(
    passageDisplayText(
      "  ## Sample custody\nBottles are labelled. ",
      "Lab.PDF",
    ),
    "Sample custody\nBottles are labelled.",
  );
  // Only a heading at a line's start: a hash inside a sentence, or a tag, stays.
  assert.equal(
    passageDisplayText("Issue #13788 and #hashtag", "a.pdf"),
    "Issue #13788 and #hashtag",
  );
  assert.equal(passageDisplayText("#nospace", "a.pdf"), "#nospace");
  // Line breaks inside the text stay; only empty lines fold.
  assert.equal(passageDisplayText("a\nb\n\n \n\nc"), "a\nb\nc");
});

// pymupdf4llm fences a PDF's code listing; its `#` comments are content, not headings. A chunk can start inside a
// listing, since the chunker splits at "\n# ".
test("PDF code listings keep their # comments", () => {
  const page =
    "# Retry Policy\n\nRetries are configured below.\n\n```\n# retry limit\nMAX_RETRIES = 3\n## backoff seconds\nBACKOFF = 2\n\n```\n\n# Logging";
  assert.equal(
    passageDisplayText(page, "client.pdf"),
    "Retry Policy\nRetries are configured below.\n```\n# retry limit\nMAX_RETRIES = 3\n## backoff seconds\nBACKOFF = 2\n```\nLogging",
  );
  assert.equal(
    passageDisplayText(
      "# backoff seconds\nBACKOFF = 2\n\n```\n\n# Logging",
      "client.pdf",
    ),
    "# backoff seconds\nBACKOFF = 2\n```\nLogging",
  );
});

// Text sources (code, config, Markdown) are indexed as written, so their # lines are content.
test("code and config passages keep their # comments and indentation", () => {
  const python =
    "# retry limit\nretries = 3\n\ndef f():\n    x = 1\n\n    return x";
  assert.equal(
    passageDisplayText(python, "worker.py"),
    "# retry limit\nretries = 3\ndef f():\n    x = 1\n    return x",
  );
  assert.equal(
    passageDisplayText("# Sample custody\nBottles are labelled.", "lab.md"),
    "# Sample custody\nBottles are labelled.",
  );
  const html = render(
    searchResult([
      {
        chunkId: "w:0",
        documentId: "w",
        filename: "worker.py",
        page: null,
        text: python,
        score: 1,
      },
    ]),
  );
  assert.ok(html.includes("# retry limit"));
  assert.ok(html.includes("\n    x = 1\n    return x"));
  // pre-wrap keeps the indentation on screen; pre-line would collapse it.
  assert.match(html, /whitespace-pre-wrap/);
});

test("code and config passages render in a monospace font; documents and notes do not", () => {
  const passage = (filename: string) =>
    render(
      searchResult([
        {
          chunkId: `${filename}:0`,
          documentId: filename,
          filename,
          page: null,
          text: "retries = 3",
          score: 1,
        },
      ]),
    );
  for (const filename of [
    "worker.py",
    "config.yaml",
    "settings.toml",
    "query.sql",
  ]) {
    assert.match(passage(filename), /font-mono/, filename);
  }
  for (const filename of [
    "handbook.pdf",
    "lab.md",
    "notes.txt",
    "report.docx",
  ]) {
    assert.ok(!passage(filename).includes("font-mono"), filename);
  }
});
