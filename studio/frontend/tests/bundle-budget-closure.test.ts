// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The startup budget is only worth having if it can fail. These pin the part that
 * decides what counts: a gate that silently measured nothing would pass forever.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { BUDGET, eagerChunksFromHtml } from "../scripts/check-bundle-budget.ts";

const HTML = `<!doctype html><html><head>
<link rel="modulepreload" crossorigin href="/assets/react-DYHJPYbT.js">
<link rel="modulepreload" crossorigin href="/assets/katex-QVq56Mr3.js">
<link rel="stylesheet" href="/assets/index-abc.css">
<script type="module" crossorigin src="/assets/index-DZBgT93Y.js"></script>
</head><body></body></html>`;

test("the eager set is the entry script plus its preloaded chunks", () => {
  assert.deepEqual(eagerChunksFromHtml(HTML), [
    "index-DZBgT93Y.js",
    "react-DYHJPYbT.js",
    "katex-QVq56Mr3.js",
  ]);
});

test("a chunk reached only by import() is not charged to startup", () => {
  // Vite emits no preload link for a dynamic-only chunk, which is the whole
  // mechanism this budget rewards. Naming one in the document body must not
  // pull it in.
  const withDynamic = HTML.replace(
    "</head>",
    '</head><body><script>import("/assets/settings-lazy.js")</script>',
  );
  assert.ok(!eagerChunksFromHtml(withDynamic).includes("settings-lazy.js"));
});

test("a non-module script is not the entry", () => {
  const html = '<script src="/assets/theme-boot.js"></script>';
  assert.deepEqual(eagerChunksFromHtml(html), []);
});

test("stylesheets and non-asset hrefs are not counted", () => {
  const html =
    '<link rel="modulepreload" href="https://cdn.example/x.js">' +
    '<link rel="stylesheet" href="/assets/index.css">';
  assert.deepEqual(eagerChunksFromHtml(html), []);
});

test("a chunk preloaded twice is counted once", () => {
  const html = `${HTML}<link rel="modulepreload" href="/assets/react-DYHJPYbT.js">`;
  const names = eagerChunksFromHtml(html);
  assert.equal(new Set(names).size, names.length);
});

test("an unrecognisable document yields nothing, so the gate reports a shape change", () => {
  // check-bundle-budget exits 2 on an empty set rather than passing at 0 bytes.
  assert.deepEqual(eagerChunksFromHtml("<!doctype html><html></html>"), []);
});

test("the budget is a real number, not a placeholder", () => {
  assert.ok(BUDGET.gzipBytes > 0 && BUDGET.rawBytes > BUDGET.gzipBytes);
});

test("the chunk count is not budgeted", () => {
  // Splitting a page out of the entry raises the count while lowering the bytes.
  // A cap here would fail the very change the gate exists to encourage.
  assert.ok(!("chunks" in BUDGET));
});
