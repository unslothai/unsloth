// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The window against the real renderer.
//
// PR #9073 windowed the reasoning pane by CHARACTER BUDGET: it sliced the Markdown SOURCE STRING
// before handing it to Streamdown. That failed repeatedly because the window had to predict where
// the renderer considers the document divisible, and that judgement changes retroactively. This
// window never touches the string; it withholds BLOCKS BY INDEX, and Streamdown decides what a
// block is. These tests are the proof of that difference, taken against the three shapes that
// broke #9073.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Block, type BlockProps, Streamdown } from "streamdown";

import { isBlockSuffix } from "../src/components/assistant-ui/block-window.ts";

/** Every `content` Streamdown handed a block component, in index order. */
function renderWithWindow(markdown: string, start: number) {
  const seen: string[] = [];
  const mounted: string[] = [];
  function Windowed(props: BlockProps) {
    seen[props.index] = props.content;
    if (props.index < start) {
      return null;
    }
    mounted.push(props.content);
    return createElement(Block, props);
  }
  const html = renderToStaticMarkup(
    createElement(
      Streamdown,
      { mode: "streaming", BlockComponent: Windowed },
      markdown,
    ),
  );
  return { seen, mounted, html };
}

/** The reference #9073 could not have: the block boundaries the renderer actually chose. */
const KILLERS: Record<string, string[]> = {
  // A GFM footnote definition arriving late. In this renderer a footnote REFERENCE alone already
  // makes the whole document one block, which is the 163-blocks-into-1 collapse in its purest
  // form, and the definition arriving does not put the boundaries back.
  "gfm footnote definition appended late": [
    "Paragraph 0 with a ref[^a] in it.\n\nParagraph 1 with a ref[^a] in it.",
    "Paragraph 0 with a ref[^a] in it.\n\nParagraph 1 with a ref[^a] in it.\n\nParagraph 2 with a ref[^a] in it.",
    "Paragraph 0 with a ref[^a] in it.\n\nParagraph 1 with a ref[^a] in it.\n\nParagraph 2 with a ref[^a] in it.\n\n[^a]: the definition arrives late.",
  ],
  "unterminated bracket span": [
    "Body.\n\nAn unterminated [bracket",
    "Body.\n\nAn unterminated [bracket span",
    "Body.\n\nAn unterminated [bracket span that never closes\n\nMore text.",
  ],
  "angle bracket link destination with trailing junk": [
    "Alpha.\n\nBeta.\n\n[spec]: <https://example.com/x>",
    "Alpha.\n\nBeta.\n\n[spec]: <https://example.com/x> junk",
    "Alpha.\n\nBeta.\n\n[spec]: <https://example.com/x> junk\n\nGamma.",
  ],
};

test("windowing cannot change the string the renderer is given, or how it is cut up", () => {
  for (const [name, stages] of Object.entries(KILLERS)) {
    for (const markdown of stages) {
      const plain = renderWithWindow(markdown, 0);
      for (const start of [1, 2, 3, 99]) {
        const windowed = renderWithWindow(markdown, start);
        // The parse is identical, block for block. Windowing is invisible to it, because the
        // string it parses is the same object.
        assert.deepEqual(
          windowed.seen,
          plain.seen,
          `${name}: withholding blocks changed the parse`,
        );
      }
      // And the blocks put back together are the same string at every window position. Streamdown
      // repairs an incomplete tail before it splits ("[bracket" becomes a link), so this is not
      // always the raw input; what matters is that the window cannot change it, because the
      // window never touches the string.
      for (const start of [0, 1, 2, 3, 99]) {
        assert.equal(
          renderWithWindow(markdown, start).seen.join(""),
          plain.seen.join(""),
          `${name}: withholding blocks changed the string that was parsed`,
        );
      }
    }
  }
});

test("a complete document's blocks are its source string, byte for byte", () => {
  const markdown =
    "# Heading\n\nParagraph one.\n\nParagraph two.\n\n- a\n- b\n\nDone.\n";
  const plain = renderWithWindow(markdown, 0);
  assert.equal(plain.seen.join(""), markdown);
  assert.ok(plain.seen.length > 5, "the document must actually be split up");
  for (const start of [1, 4, 7]) {
    assert.equal(renderWithWindow(markdown, start).seen.join(""), markdown);
  }
});

test("what is mounted is always a suffix of the renderer's block list", () => {
  for (const [name, stages] of Object.entries(KILLERS)) {
    for (const markdown of stages) {
      const plain = renderWithWindow(markdown, 0);
      for (let start = 0; start <= plain.seen.length; start += 1) {
        const windowed = renderWithWindow(markdown, start);
        assert.ok(
          isBlockSuffix(plain.seen, windowed.mounted),
          `${name}: start ${start} did not mount a suffix`,
        );
        assert.equal(windowed.mounted.length, plain.seen.length - start);
      }
    }
  }
});

test("a withheld block removes its own markup and nothing else", () => {
  const markdown =
    "# Heading\n\nParagraph one.\n\nParagraph two.\n\n- a\n- b\n\nParagraph three.";
  const plain = renderWithWindow(markdown, 0);
  for (let start = 1; start < plain.seen.length; start += 1) {
    const windowed = renderWithWindow(markdown, start);
    const dropped = renderWithWindow(markdown, 0).html;
    // Each block renders independently, so the windowed markup is the tail of the full markup.
    const inner = (html: string) =>
      html.slice(html.indexOf(">") + 1, html.lastIndexOf("</div>"));
    assert.ok(
      inner(dropped).endsWith(inner(windowed.html)),
      `start ${start}: the windowed markup is not the tail of the full markup`,
    );
  }
});

test("the footnote shape leaves the window with nothing to do, rather than breaking it", () => {
  // The honest result, and the reason this design survives the case that killed #9073: with the
  // whole document in one block there is no index to move the window to, so it stays at 0 and the
  // pane renders exactly what it renders today. The optimisation stops applying; nothing breaks.
  const stages = KILLERS["gfm footnote definition appended late"];
  for (const markdown of stages) {
    const plain = renderWithWindow(markdown, 0);
    assert.equal(
      plain.seen.length,
      1,
      "a footnote reference is expected to collapse the document into one block",
    );
    // Block 0 is never given a marker, so it can never be measured, so it can never be dropped.
    const windowed = renderWithWindow(markdown, 0);
    assert.deepEqual(windowed.mounted, plain.seen);
  }
});

// ── the wiring, which the SSR above cannot see ──────────────────────

const read = (relative: string) =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const MARKDOWN_TEXT = read("../src/components/assistant-ui/markdown-text.tsx");
const REASONING = read("../src/components/assistant-ui/reasoning.tsx");
const INDEX_CSS = read("../src/index.css");

test("only a streaming reasoning pane gets the windowed block component", () => {
  assert.match(
    MARKDOWN_TEXT,
    /BlockComponent=\{windowed \? WindowedStreamdownBlock : StreamdownBlock\}/,
    "the answer text and settled panes must keep the block component they have today",
  );
  assert.match(
    MARKDOWN_TEXT,
    /const windowed = useBlockWindowPaneActive\(\)/,
  );
  // The pane context is supplied by ReasoningText and by nothing else, and only while streaming.
  assert.match(
    REASONING,
    /<BlockWindowPaneProvider paneRef=\{scrollRef\} enabled=\{Boolean\(streaming\)\}>/,
  );
  const providers = [
    ...MARKDOWN_TEXT.matchAll(/BlockWindowPaneProvider/g),
  ].length;
  assert.equal(providers, 0, "MarkdownText must not declare itself a pane");
});

test("scroll anchoring is switched off rather than depended on", () => {
  // Playwright's WebKit implements overflow-anchor and real Safari does not, so leaving it on
  // would let the sanctioned test proxy show anchoring working while the WebKitGTK embed drifted.
  assert.match(REASONING, /style=\{\{ overflowAnchor: "none" \}\}/);
});

test("the slot restores the three Streamdown spacing rules it displaces", () => {
  // The slot is display: contents so it generates no box and margin collapsing between adjacent
  // markdown blocks is unaffected. That also means it cannot carry a margin, and all three of
  // Streamdown's spacing rules select the container's DIRECT children, which the slot has become.
  assert.match(INDEX_CSS, /:where\(\[data-aui-block-slot\]\) \{\s*display: contents;/);
  assert.match(
    INDEX_CSS,
    /:where\(\[data-aui-block-slot\]\) > \* \{\s*margin-block-end: 1rem;/,
  );
  assert.match(
    INDEX_CSS,
    /\[data-aui-block-slot\]:first-child > \*:first-child \{\s*margin-block-start: 0;/,
  );
  assert.match(
    INDEX_CSS,
    /\[data-aui-block-slot\]:not\(:has\(~ \[data-aui-block-slot\] > \*\)\) > \*:last-child \{\s*margin-block-end: 0;/,
  );
});

test("the restored gap is still the gap Streamdown asks for", () => {
  // The 1rem above is a copy of Streamdown's own `space-y-4`. If a Streamdown upgrade changes
  // that class, the copy is silently wrong and every block in a streaming reasoning pane moves,
  // so read the class back off a real render rather than trusting it.
  const html = renderToStaticMarkup(
    createElement(Streamdown, { mode: "streaming" }, "a\n\nb"),
  );
  const container = html.slice(0, html.indexOf(">"));
  assert.match(container, /class="[^"]*\bspace-y-4\b/);
  assert.match(container, /\[&amp;&gt;\*:first-child\]:mt-0/);
  assert.match(container, /\[&amp;&gt;\*:last-child\]:mb-0/);
});

test("block 0 is never wrapped, so the plain first-child rule still reaches it", () => {
  assert.match(
    MARKDOWN_TEXT,
    /if \(props\.index === 0\) \{\s*return <StreamdownBlock \{\.\.\.props\} \/>;/,
  );
});
