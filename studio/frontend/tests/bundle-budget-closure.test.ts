// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  const withDynamic = HTML.replace(
    "</head>",
    '</head><body><script>import("/assets/settings-lazy.js")</script>',
  );
  assert.ok(
    !eagerChunksFromHtml(withDynamic).includes("assets/settings-lazy.js"),
  );
});

test("a classic script is charged to startup, but not as the entry", () => {
  const html = '<script src="/theme-boot.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html), {
    entry: [],
    preloads: [],
    blocking: ["theme-boot.js"],
  });
});

test("a deferred script is on the startup path, an async one is not", () => {
  // async wins when a tag carries both async and defer.
  const html =
    '<script defer src="/late.js"></script>' +
    '<script async src="/whenever.js"></script>' +
    '<script async defer src="/also-whenever.js"></script>';
  assert.deepEqual(eagerChunksFromHtml(html), ["late.js"]);
});

test("an async script that blocks rendering is on the startup path", () => {
  // blocking="render" holds first paint until the script runs, so it counts despite async.
  const html =
    '<script async blocking="render" src="/theme-boot.js"></script>' +
    '<script async src="/whenever.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, ["theme-boot.js"]);
});

test("the blocking attribute is a token list, matched like the spec matches it", () => {
  const html =
    '<script async blocking="  RENDER  " src="/shouty.js"></script>' +
    '<script async blocking="render full" src="/two-tokens.js"></script>' +
    '<script async blocking="rendering" src="/not-a-token.js"></script>' +
    '<script async blocking="prerender" src="/also-not.js"></script>' +
    '<script async blocking="" src="/empty.js"></script>' +
    '<script async data-blocking="render" src="/decoy.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, [
    "shouty.js",
    "two-tokens.js",
  ]);
});

test("blocking=render on a non-async script changes nothing", () => {
  const html =
    '<script blocking="render" src="/theme-boot.js"></script>' +
    '<script blocking="render" src="/theme-boot.js"></script>' +
    '<script type="application/json" blocking="render" src="/data.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, ["theme-boot.js"]);
});

test("a script type is judged on whether the browser runs it", () => {
  const html =
    '<script type="application/javascript" src="/legacy.js"></script>' +
    '<script type="importmap" src="/map.js"></script>' +
    '<script type="application/json" src="/data.js"></script>';
  assert.deepEqual(eagerChunksFromHtml(html), ["legacy.js"]);
});

test("a MIME type with parameters is not a script the browser runs", () => {
  // type is matched on MIME essence, so a parameter makes browsers skip the script.
  const html = '<script type="text/javascript; charset=utf-8" src="/never.js">';
  assert.deepEqual(eagerChunksFromHtml(html), []);
});

test("every JavaScript MIME essence the browser still runs is counted", () => {
  // Chromium executes every legacy JavaScript MIME spelling.
  const essences = [
    "application/ecmascript",
    "application/javascript",
    "application/x-ecmascript",
    "application/x-javascript",
    "text/ecmascript",
    "text/javascript",
    "text/javascript1.0",
    "text/javascript1.1",
    "text/javascript1.2",
    "text/javascript1.3",
    "text/javascript1.4",
    "text/javascript1.5",
    "text/jscript",
    "text/livescript",
    "text/x-ecmascript",
    "text/x-javascript",
  ];
  for (const [i, essence] of essences.entries()) {
    const html = `<script type="${essence}" src="/boot-${i}.js"></script>`;
    assert.deepEqual(
      eagerChunksFromHtml(html),
      [`boot-${i}.js`],
      `${essence} should be counted`,
    );
    assert.deepEqual(
      eagerChunksFromHtml(html.replace(essence, essence.toUpperCase())),
      [`boot-${i}.js`],
      `${essence} should match case-insensitively`,
    );
  }
});

test("a type that only looks like JavaScript is not counted", () => {
  for (const type of [
    "text/javascript1.6",
    "application/javascript-ish",
    "text/jscript.encode",
    "application/json",
    "importmap",
  ]) {
    assert.deepEqual(
      eagerChunksFromHtml(`<script type="${type}" src="/nope.js"></script>`),
      [],
      `${type} should not be counted`,
    );
  }
});

test("a decoy data- attribute cannot shadow the real one", () => {
  const html =
    '<script data-type="metadata" type="module" src="/assets/entry.js"></script>' +
    '<script data-src="/decoy.js" type="module" src="/assets/second.js"></script>' +
    '<link data-rel="x" rel="modulepreload" data-href="/assets/no.js" href="/assets/yes.js">';
  assert.deepEqual(eagerSetFromHtml(html), {
    entry: ["assets/entry.js", "assets/second.js"],
    preloads: ["assets/yes.js"],
    blocking: [],
  });
});

test("`async` inside a value is not an async attribute", () => {
  const html =
    '<script data-mode="load async later" src="/theme-boot.js"></script>' +
    '<script data-flags="async defer" src="/also-blocking.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, [
    "theme-boot.js",
    "also-blocking.js",
  ]);
});

test("an attribute whose name ends in async is not async", () => {
  const html = '<script data-async src="/theme-boot.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, ["theme-boot.js"]);
});

test("a quoted value containing `>` does not end the tag", () => {
  const html =
    '<script data-note="a > b" type="module" src="/assets/entry.js"></script>' +
    "<script data-note='b > c' src='/theme-boot.js'></script>" +
    '<link data-note="x > y" rel="modulepreload" href="/assets/react-x.js">';
  assert.deepEqual(eagerSetFromHtml(html), {
    entry: ["assets/entry.js"],
    preloads: ["assets/react-x.js"],
    blocking: ["theme-boot.js"],
  });
});

test("an unquoted value ends at whitespace, not at the first likely-looking token", () => {
  const html =
    "<script type=module src=/assets/entry.js></script>" +
    "<link rel=modulepreload href=/assets/react-x.js>";
  assert.deepEqual(eagerSetFromHtml(html), {
    entry: ["assets/entry.js"],
    preloads: ["assets/react-x.js"],
    blocking: [],
  });
});

test("a tag name is matched whole, so scriptx is not a script", () => {
  const html =
    '<scriptx src="/nope.js"></scriptx><linkx rel="modulepreload" href="/assets/nope.js">';
  assert.deepEqual(eagerChunksFromHtml(html), []);
});

test("a commented-out tag is not downloaded, so it is not charged", () => {
  const html =
    '<!-- <script src="/disabled.js"></script> -->' +
    '<script src="/theme-boot.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, ["theme-boot.js"]);
});

test("a tag written inside an inline script is text, not a tag", () => {
  const html =
    '<script>document.write(\'<link rel="modulepreload" href="/assets/nope.js">\')</script>' +
    '<link rel="modulepreload" href="/assets/react-x.js">';
  assert.deepEqual(eagerSetFromHtml(html).preloads, ["assets/react-x.js"]);
});

test("a truncated file does not invent a tag out of its last line", () => {
  assert.deepEqual(eagerChunksFromHtml('<script src="/half-written.js'), []);
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
  assert.deepEqual(eagerChunksFromHtml("<!doctype html><html></html>"), []);
});

test("the budget is a real number, not a placeholder", () => {
  assert.ok(BUDGET.transferBytes > 0 && BUDGET.rawBytes > BUDGET.transferBytes);
});

test("the entry and its preloads stay distinguishable", () => {
  assert.deepEqual(eagerSetFromHtml(HTML), {
    entry: ["assets/index-DZBgT93Y.js"],
    preloads: ["assets/react-DYHJPYbT.js", "assets/katex-QVq56Mr3.js"],
    blocking: [],
  });
});

test("a build with no preload links is not mistaken for a one-chunk app", () => {
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
  // No chunk-count cap: splitting a page out raises the count while lowering bytes.
  assert.ok(!("chunks" in BUDGET));
});
