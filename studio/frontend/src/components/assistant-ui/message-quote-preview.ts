// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compact display-only text for quote selections that contain tables. The
// stored anchor text stays search-authoritative; this module only formats the
// same pruned Range clone for the hover preview.

import { normalizeQuoteText } from "./message-quote-model.ts";

export type QuotePreviewNode = {
  readonly nodeType: number;
  readonly tagName?: string;
  readonly data?: string;
  readonly childNodes: ArrayLike<QuotePreviewNode>;
};

export type QuotePreviewBlock =
  | { readonly kind: "text"; readonly text: string }
  | {
      readonly kind: "table";
      readonly rows: readonly (readonly string[])[];
    };

const CELL_TAGS: ReadonlySet<string> = new Set(["TD", "TH"]);
const CELL_SEPARATOR = " | ";
const TRAILING_SPACE = /[ \t]+$/;
const WHITESPACE_RUN = /\s+/g;

/** Block and table tags contribute a line or word boundary while flattening. */
const BLOCK_TAGS: ReadonlySet<string> = new Set([
  "ADDRESS",
  "ARTICLE",
  "ASIDE",
  "BLOCKQUOTE",
  "BR",
  "CAPTION",
  "DD",
  "DETAILS",
  "DIALOG",
  "DIV",
  "DL",
  "DT",
  "FIELDSET",
  "FIGCAPTION",
  "FOOTER",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "HEADER",
  "HR",
  "LI",
  "MAIN",
  "NAV",
  "OL",
  "P",
  "PRE",
  "SECTION",
  "SUMMARY",
  "TABLE",
  "TBODY",
  "TD",
  "TFOOT",
  "TH",
  "THEAD",
  "TR",
  "UL",
]);

function tagNameOf(node: QuotePreviewNode): string {
  return node.nodeType === 1 ? (node.tagName ?? "").toUpperCase() : "";
}

function rawText(node: QuotePreviewNode): string {
  if (node.nodeType === 3) {
    return node.data ?? "";
  }
  let text = "";
  for (const child of Array.from(node.childNodes)) {
    text += rawText(child);
  }
  return text;
}

/** One row must stay on one line, so block and `<br>` breaks become spaces. */
function cellText(cell: QuotePreviewNode): string {
  let text = "";
  const visit = (node: QuotePreviewNode): void => {
    if (node.nodeType === 3) {
      text += node.data ?? "";
      return;
    }
    if (node.nodeType !== 1) {
      for (const child of Array.from(node.childNodes)) {
        visit(child);
      }
      return;
    }
    const block = BLOCK_TAGS.has(tagNameOf(node));
    if (block) {
      text += " ";
    }
    for (const child of Array.from(node.childNodes)) {
      visit(child);
    }
    if (block) {
      text += " ";
    }
  };
  visit(cell);
  return text.replace(WHITESPACE_RUN, " ").trim();
}

/**
 * Returns compact display text only for real table structure. A common
 * ancestor-`TR` clone exposes bare `TH`/`TD` siblings instead of a `TR`, so
 * consecutive cells are treated as a row too.
 */
export function tableQuotePreview(
  fragment: QuotePreviewNode,
  canonicalText: string,
): string | null {
  const lines: string[] = [];
  let current = "";
  let sawMultiCellRow = false;

  const flush = (): void => {
    const line = current.replace(TRAILING_SPACE, "");
    if (line.trim().length > 0) {
      lines.push(line);
    }
    current = "";
  };

  const emitRow = (cells: readonly QuotePreviewNode[]): void => {
    if (cells.length === 0) {
      return;
    }
    const values = cells.map(cellText);
    if (values.every((value) => value.length === 0)) {
      return;
    }
    if (cells.length > 1) {
      sawMultiCellRow = true;
    }
    flush();
    lines.push(values.join(CELL_SEPARATOR).replace(TRAILING_SPACE, ""));
  };

  const isCell = (node: QuotePreviewNode): boolean =>
    node.nodeType === 1 && CELL_TAGS.has(tagNameOf(node));
  const isWhitespace = (node: QuotePreviewNode): boolean =>
    node.nodeType === 3 && (node.data ?? "").trim().length === 0;

  const cellsIn = (node: QuotePreviewNode): QuotePreviewNode[] => {
    const cells: QuotePreviewNode[] = [];
    for (const child of Array.from(node.childNodes)) {
      if (isCell(child)) {
        cells.push(child);
      }
    }
    return cells;
  };

  const emitCellRun = (
    children: readonly QuotePreviewNode[],
    start: number,
  ): number => {
    const cells: QuotePreviewNode[] = [];
    let index = start;
    while (index < children.length) {
      const candidate = children[index];
      if (isCell(candidate)) {
        cells.push(candidate);
        index += 1;
        continue;
      }
      if (!isWhitespace(candidate)) {
        break;
      }
      index += 1;
    }
    emitRow(cells);
    return index;
  };

  const visitChildren = (node: QuotePreviewNode): void => {
    const children = Array.from(node.childNodes);
    let index = 0;
    while (index < children.length) {
      const child = children[index];
      if (!isCell(child)) {
        visit(child);
        index += 1;
        continue;
      }
      index = emitCellRun(children, index);
    }
  };

  const visit = (node: QuotePreviewNode): void => {
    if (node.nodeType === 3) {
      current += node.data ?? "";
      return;
    }
    if (node.nodeType !== 1 && node.nodeType !== 11) {
      return;
    }
    const tag = tagNameOf(node);
    if (tag === "TR") {
      emitRow(cellsIn(node));
      return;
    }
    const block = BLOCK_TAGS.has(tag);
    if (block) {
      flush();
    }
    visitChildren(node);
    if (block) {
      flush();
    }
  };

  visit(fragment);
  flush();

  if (!sawMultiCellRow || lines.length === 0) {
    return null;
  }

  // Never show text that cannot be reconciled with the stored anchor. Both
  // sides are compared without structural newlines because the index adds
  // them where the DOM text has none.
  const expected = normalizeQuoteText(canonicalText);
  const actual = normalizeQuoteText(rawText(fragment));
  if (expected.replace(/\n/g, "") !== actual.replace(/\n/g, "")) {
    return null;
  }

  const compact = normalizeQuoteText(lines.join("\n"));
  if (
    compact.length === 0 ||
    compact.split("\n").length >= expected.split("\n").length
  ) {
    return null;
  }
  return compact;
}

/**
 * Row detection is heuristic: previewText is a flat string, so every prose
 * `" | "` is indistinguishable from a table-cell separator.
 */
function isPreviewTableLine(line: string): boolean {
  const value = line.trim();
  return (
    value.includes(CELL_SEPARATOR) ||
    value.endsWith(" |") ||
    value.startsWith("| ") ||
    value === "|"
  );
}

function splitPreviewTableCells(line: string): string[] {
  const trimmed = line.trim();
  const startsWithPipe = trimmed.startsWith("|");
  const endsWithPipe = trimmed.endsWith("|");
  let body = trimmed;
  if (startsWithPipe) {
    body = body.slice(1).trimStart();
  }
  if (endsWithPipe) {
    body = body.slice(0, -1).trimEnd();
  }
  const cells =
    body.length === 0
      ? []
      : body.split(CELL_SEPARATOR).map((cell) => cell.trim());
  if (startsWithPipe) {
    cells.unshift("");
  }
  if (endsWithPipe) {
    cells.push("");
  }
  return cells;
}

/**
 * Splits compact table preview text into display blocks. The parser is
 * deliberately display-only: capture, navigation, and transport continue to
 * use the untouched anchor fields.
 */
export function parseQuotePreviewBlocks(
  text: string,
  enableTables: boolean,
): QuotePreviewBlock[] {
  if (!enableTables) {
    return [{ kind: "text", text }];
  }

  const lines = text.split("\n");
  const blocks: QuotePreviewBlock[] = [];
  const textLines: string[] = [];
  const flushText = (): void => {
    if (textLines.length === 0) {
      return;
    }
    blocks.push({ kind: "text", text: textLines.join("\n") });
    textLines.length = 0;
  };

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!isPreviewTableLine(line)) {
      textLines.push(line);
      index += 1;
      continue;
    }

    flushText();
    const rows: string[][] = [];
    while (index < lines.length && isPreviewTableLine(lines[index])) {
      rows.push(splitPreviewTableCells(lines[index]));
      index += 1;
    }
    const width = rows.reduce((widest, row) => Math.max(widest, row.length), 0);
    blocks.push({
      kind: "table",
      rows: rows.map((row) => [
        ...row,
        ...Array.from({ length: width - row.length }, () => ""),
      ]),
    });
  }

  flushText();
  return blocks;
}
