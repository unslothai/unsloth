// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The preview formatter is pure and only recognizes real table structure. It
// never infers a table from newline runs in ordinary passage text.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type QuotePreviewNode,
  tableQuotePreview,
} from "../src/components/assistant-ui/message-quote-preview.ts";

function text(data: string): QuotePreviewNode {
  return { nodeType: 3, data, childNodes: [] };
}

function el(
  tagName: string,
  ...children: QuotePreviewNode[]
): QuotePreviewNode {
  return { nodeType: 1, tagName, childNodes: children };
}

function fragment(...children: QuotePreviewNode[]): QuotePreviewNode {
  return { nodeType: 11, childNodes: children };
}

function row(...cells: QuotePreviewNode[]): QuotePreviewNode {
  return el("tr", ...cells);
}

function cell(tagName: "th" | "td", value: string): QuotePreviewNode {
  return el(tagName, text(value));
}

test("a 2x2 table compacts to one line per row", () => {
  const table = el(
    "table",
    el("thead", row(cell("th", "Name"), cell("th", "Value"))),
    el("tbody", row(cell("td", "Alpha"), cell("td", "1"))),
  );

  assert.equal(
    tableQuotePreview(fragment(table), "Name\nValue\nAlpha\n1"),
    "Name | Value\nAlpha | 1",
  );
});

test("multiple rows and columns stay row-major and compact", () => {
  const table = el(
    "table",
    el(
      "thead",
      row(cell("th", "Name"), cell("th", "Value"), cell("th", "Notes")),
    ),
    el(
      "tbody",
      row(cell("td", "Alpha"), cell("td", "1"), cell("td", "first")),
      row(cell("td", "Beta"), cell("td", "2"), cell("td", "second")),
    ),
  );

  assert.equal(
    tableQuotePreview(
      fragment(table),
      "Name\nValue\nNotes\nAlpha\n1\nfirst\nBeta\n2\nsecond",
    ),
    "Name | Value | Notes\nAlpha | 1 | first\nBeta | 2 | second",
  );
});

test("bare TH/TD siblings from a row-level clone form one row", () => {
  const clone = fragment(
    el("th", text("Name")),
    text(" "),
    el("th", text("Value")),
  );

  assert.equal(tableQuotePreview(clone, "Name \nValue"), "Name | Value");
});

test("multiline cells keep their text in one row line", () => {
  const table = el(
    "table",
    el(
      "tbody",
      row(
        cell("td", "Alpha"),
        el("td", text("first"), el("br"), text("second")),
      ),
    ),
  );

  assert.equal(
    tableQuotePreview(fragment(table), "Alpha\nfirst\nsecond"),
    "Alpha | first second",
  );
});

test("mixed paragraphs keep their lines around compact table rows", () => {
  const table = el(
    "table",
    el("thead", row(cell("th", "Name"), cell("th", "Value"))),
    el("tbody", row(cell("td", "Alpha"), cell("td", "1"))),
  );
  const clone = fragment(el("p", text("Intro")), table, el("p", text("Outro")));

  assert.equal(
    tableQuotePreview(clone, "Intro\nName\nValue\nAlpha\n1\nOutro"),
    "Intro\nName | Value\nAlpha | 1\nOutro",
  );
});

test("empty cells keep their separators", () => {
  const table = el(
    "table",
    el("tbody", row(cell("td", "A"), el("td"), cell("td", "C"))),
  );

  assert.equal(tableQuotePreview(fragment(table), "A\nC"), "A |  | C");
});

test("one-column tables return no preview", () => {
  const table = el(
    "table",
    el("tbody", row(cell("td", "Alpha")), row(cell("td", "Beta"))),
  );

  assert.equal(tableQuotePreview(fragment(table), "Alpha\nBeta"), null);
});

test("paragraph-only fragments return no preview", () => {
  const clone = fragment(el("p", text("One")), el("p", text("Two")));

  assert.equal(tableQuotePreview(clone, "One\nTwo"), null);
});

test("list-only fragments return no preview", () => {
  const clone = fragment(
    el("ul", el("li", text("One")), el("li", text("Two"))),
  );

  assert.equal(tableQuotePreview(clone, "One\nTwo"), null);
});

test("code-only fragments return no preview", () => {
  const clone = fragment(el("pre", el("code", text("const alpha = 1;"))));

  assert.equal(tableQuotePreview(clone, "const alpha = 1;"), null);
});

test("mismatched canonical text returns no preview", () => {
  const table = el(
    "table",
    el("tbody", row(cell("td", "Name"), cell("td", "Value"))),
  );

  assert.equal(tableQuotePreview(fragment(table), "Name\nOther"), null);
});
