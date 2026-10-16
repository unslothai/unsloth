// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The retrieved passages a document search shows on request (#13788): grouped under their document,
// ranked, opening at their page, long ones clamped, and capped like the document chips.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import * as citationUtils from "../src/components/assistant-ui/citation-utils.ts";
import type * as PassagesModule from "../src/components/assistant-ui/knowledge-base-passages.tsx";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const {
  parseCitations,
  passageDisplayText,
  passageNeedsClamp,
  passagesPerDocument,
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

test("passages group under their document in chip order, each keeping its rank in the whole search", () => {
  const groups = passagesPerDocument(parseCitations(SEARCH));
  assert.deepEqual(
    groups.map((g) => [
      g.filename,
      g.passages.map((p) => [p.rank, p.citation.page]),
    ]),
    [
      [
        "handbook.pdf",
        [
          [1, 3],
          [3, 1],
        ],
      ],
      ["lab.md", [[2, null]]],
    ],
  );
  // An empty chunk is not a passage: it would render as a blank card.
  assert.equal(groups[0].passages.length, 2);
});

test("a whole-document read has no ranking, so its passages carry none", () => {
  const groups = passagesPerDocument(
    parseCitations(
      searchResult([
        {
          chunkId: "a:0",
          documentId: "a",
          filename: "a.txt",
          page: null,
          text: "first",
          score: null,
        },
        {
          chunkId: "a:1",
          documentId: "a",
          filename: "a.txt",
          page: null,
          text: "second",
          score: null,
        },
      ]),
    ),
  );
  assert.deepEqual(
    groups[0].passages.map((p) => p.rank),
    [null, null],
  );
});

// The backend appends ranked project passages after a whole-document read under one numbering.
test("ranked passages merged after a whole-document read still start at #1", () => {
  const groups = passagesPerDocument(
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
    groups.map((g) => g.passages.map((p) => p.rank)),
    [
      [null, null],
      [1, 2],
    ],
  );
});

test("each passage shows its exact text, its rank, and opens its document at its own page", () => {
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
  // Each document is named once above its passages.
  assert.equal(html.match(/title="handbook\.pdf"/g)?.length, 1);
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
