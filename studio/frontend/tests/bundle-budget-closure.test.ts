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

test("an async script that blocks rendering is on the startup path", () => {
  // `blocking="render"` holds the first paint until the script has been fetched and
  // run, which is the timeline this budgets, and it is the documented way to take a
  // boot script off the parser without letting the unthemed page paint. Measured in
  // Chromium 151 and WebKit 26.5: a two-second script moved first contentful paint
  // from ~20 ms to ~2,020 ms. Excluded for being async, it could grow without limit.
  const html =
    '<script async blocking="render" src="/theme-boot.js"></script>' +
    '<script async src="/whenever.js"></script>';
  assert.deepEqual(eagerSetFromHtml(html).blocking, ["theme-boot.js"]);
});

test("the blocking attribute is a token list, matched like the spec matches it", () => {
  // "converted to ASCII lowercase... split on ASCII whitespace", then the set is
  // asked whether it contains "render". So case and neighbouring tokens do not
  // matter, and a token that merely CONTAINS render is a different token.
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
  // It is already counted for being parser-blocking; saying so twice must not
  // duplicate it, and the attribute must not pull in a type the browser never runs.
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
  // The type attribute is matched on JavaScript MIME type ESSENCE, so a parameter
  // makes it match nothing. Chromium, Firefox and WebKit all decline to fetch it,
  // and bytes the browser never requests are not startup cost.
  const html = '<script type="text/javascript; charset=utf-8" src="/never.js">';
  assert.deepEqual(eagerChunksFromHtml(html), []);
});

test("every JavaScript MIME essence the browser still runs is counted", () => {
  // The legacy spellings are not dead letters: measured in Chromium 151, every one
  // of these executes. Counting only the four anyone writes today dropped the rest
  // while the entry and preloads kept the shape guard satisfied.
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
  // The guard against the list above turning into "anything with `script` in it".
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
  // A hyphen is a word boundary, so a `\b`-anchored `type` reads `data-type`
  // first, drops the entry, and leaves the preloads to satisfy the shape guard.
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
  // An attribute begins only after whitespace, so `async` can only be read from a
  // NAME. Searching the tag text found it in this value, dropped the only
  // parser-blocking script, and left the entry and preloads to pass the shape
  // guard -- so theme-boot.js could grow without ever showing up.
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
  // `>` is ordinary text inside a quoted value; the tag really does continue. Read
  // as the end of the tag, `type` and `src` fall off the entry, and the preloads
  // alone still satisfy the shape guard -- a pass measured without the entry chunk.
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
  // EOF inside a tag is a parse error and the tag token is never emitted, so the
  // browser does not fetch it either.
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
