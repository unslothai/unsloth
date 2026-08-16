// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The startup budget is only worth having if it can fail. These pin the part that
 * decides what counts: a gate that silently measured nothing would pass forever.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BUDGET,
  eagerChunksFromHtml,
  eagerSetFromHtml,
} from "../scripts/check-bundle-budget.ts";

const HTML = `<!doctype html><html><head>
<link rel="modulepreload" crossorigin href="/assets/react-DYHJPYbT.js">
<link rel="modulepreload" crossorigin href="/assets/katex-QVq56Mr3.js">
<link rel="stylesheet" href="/assets/index-abc.css">
<script type="module" crossorigin src="/assets/index-DZBgT93Y.js"></script>
</head><body></body></html>`;

test("the eager set is the entry script plus its preloaded chunks", () => {
  assert.deepEqual(eagerChunksFromHtml(HTML), [
    "assets/index-DZBgT93Y.js",
    "assets/react-DYHJPYbT.js",
    "assets/katex-QVq56Mr3.js",
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
  assert.ok(
    !eagerChunksFromHtml(withDynamic).includes("assets/settings-lazy.js"),
  );
});

test("a classic script is charged to startup, but not as the entry", () => {
  // Parser-blocking, so the browser fetches and runs it before the module graph.
  // public/theme-boot.js is one, and it is not under assets/.
  const html = '<script src="/theme-boot.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html), {
    entry: [],
    preloads: [],
    blocking: ["theme-boot.js"],
  });
});

test("a deferred script is on the startup path, an async one is not", () => {
  // defer runs after parsing but before DOMContentLoaded, in document order with
  // the module entry, which is deferred too. async has no such relationship, and
  // async wins when a tag carries both.
  const html =
    '<script defer src="/late.js"></script>' +
    '<script async src="/whenever.js"></script>' +
    '<script async defer src="/also-whenever.js"></script>';
  assert.deepEqual(eagerChunksFromHtml(html), ["late.js"]);
});

test("a script type is judged on whether the browser runs it", () => {
  const html =
    '<script type="application/javascript" src="/legacy.js"></script>' +
    '<script type="importmap" src="/map.js"></script>' +
    '<script type="application/json" src="/data.js"></script>';
  assert.deepEqual(eagerChunksFromHtml(html), ["legacy.js"]);
});

test("a cross-origin classic script is not ours to budget", () => {
  const html =
    '<script src="https://cdn.example/x.js"></script>' +
    '<script src="//cdn.example/y.js"></script>';
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
  assert.ok(BUDGET.transferBytes > 0 && BUDGET.rawBytes > BUDGET.transferBytes);
});

test("the entry and its preloads stay distinguishable", () => {
  // check-bundle-budget refuses to report a number when either half is missing,
  // which it cannot do if the two are flattened together before it looks.
  assert.deepEqual(eagerSetFromHtml(HTML), {
    entry: ["assets/index-DZBgT93Y.js"],
    preloads: ["assets/react-DYHJPYbT.js", "assets/katex-QVq56Mr3.js"],
    blocking: [],
  });
});

test("a build with no preload links is not mistaken for a one-chunk app", () => {
  // `build.modulePreload: false` emits exactly this: the entry, no links. Read as
  // a flat list it looks like a 424 KB startup path with 4.8 MB to spare.
  const entryOnly = HTML.replace(/<link rel="modulepreload"[^>]*>\n?/g, "");
  assert.deepEqual(eagerSetFromHtml(entryOnly).preloads, []);
  assert.equal(eagerSetFromHtml(entryOnly).entry.length, 1);
});

test("tag and attribute matching is case-insensitive, as HTML is", () => {
  const shouty = HTML.replace(
    /<link rel="modulepreload" crossorigin href="([^"]+)">/g,
    '<LINK REL="MODULEPRELOAD" CROSSORIGIN HREF="$1">',
  ).replace(/<script type="module"/, '<SCRIPT TYPE="Module"');
  assert.deepEqual(eagerChunksFromHtml(shouty), eagerChunksFromHtml(HTML));
});

test("attribute order, quoting and self-closing syntax do not matter", () => {
  const rewritten = HTML.replace(
    /<link rel="modulepreload" crossorigin href="([^"]+)">/g,
    "<link href='$1' crossorigin rel='modulepreload' />",
  ).replace(
    /<script type="module" crossorigin src="([^"]+)"><\/script>/,
    "<script crossorigin src=$1 type=module></script>",
  );
  assert.deepEqual(eagerChunksFromHtml(rewritten), eagerChunksFromHtml(HTML));
});

test("rel is a token list, so a second token does not hide the preload", () => {
  const html = '<link rel="preload modulepreload" href="/assets/react-x.js">';
  assert.deepEqual(eagerSetFromHtml(html).preloads, ["assets/react-x.js"]);
});

test("a tag broken across lines is still read", () => {
  const html = '<link\n  rel="modulepreload"\n  href="/assets/react-x.js"\n>';
  assert.deepEqual(eagerSetFromHtml(html).preloads, ["assets/react-x.js"]);
});

test("the chunk count is not budgeted", () => {
  // Splitting a page out of the entry raises the count while lowering the bytes.
  // A cap here would fail the very change the gate exists to encourage.
  assert.ok(!("chunks" in BUDGET));
});
