// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

import { katexWoff2Only, keepWoff2Sources } from "../vite-plugin-katex-woff2.ts";

const require = createRequire(import.meta.url);
const KATEX_CSS_PATH = require.resolve("katex/dist/katex.min.css");
const KATEX_CSS = readFileSync(KATEX_CSS_PATH, "utf8");

const FONT_FACE = /@font-face\{[^}]*\}/g;

test("every KaTeX @font-face keeps exactly its woff2 source and nothing else changes", () => {
  const { css, rules } = keepWoff2Sources(KATEX_CSS);
  const before = KATEX_CSS.match(FONT_FACE) ?? [];
  const after = css.match(FONT_FACE) ?? [];
  assert.ok(before.length >= 20, `expected the KaTeX font set, found ${before.length} rules`);
  assert.equal(rules, before.length);
  assert.equal(after.length, before.length);
  for (const [index, rule] of before.entries()) {
    const expected = rule.replace(
      /,url\(fonts\/[\w-]+\.woff\) format\("woff"\),url\(fonts\/[\w-]+\.ttf\) format\("truetype"\)/,
      "",
    );
    assert.notEqual(expected, rule, `rule ${index} lost no fallback: ${rule}`);
    assert.equal(after[index], expected);
  }
  assert.equal(css.replace(FONT_FACE, ""), KATEX_CSS.replace(FONT_FACE, ""));
  assert.doesNotMatch(css, /\.woff\)|\.ttf\)|format\("woff"\)|format\("truetype"\)/);
});

test("quoted, spaced and upper-case woff2 sources are kept as written", () => {
  const css =
    "@font-face{font-family:A;src:url('a.woff2') format('woff2'),url(a.woff) format(\"woff\")}" +
    "@font-face{font-family:B;src:url( \"b.woff2\" ) FORMAT( \"WOFF2\" ),url(b.ttf) format(\"truetype\")}" +
    ".x{src:url(a.woff)}";
  assert.equal(
    keepWoff2Sources(css).css,
    "@font-face{font-family:A;src:url('a.woff2') format('woff2')}" +
      "@font-face{font-family:B;src:url( \"b.woff2\" ) FORMAT( \"WOFF2\" )}" +
      ".x{src:url(a.woff)}",
  );
});

test("a font-face without a single woff2 source fails the build instead of dropping the font", () => {
  assert.throws(() => keepWoff2Sources("@font-face{font-family:A;src:url(a.woff) format(\"woff\")}"), /0 woff2/);
  assert.throws(
    () =>
      keepWoff2Sources(
        "@font-face{font-family:A;src:url(a.woff2) format(\"woff2\"),url(b.woff2) format(\"woff2\")}",
      ),
    /2 woff2/,
  );
});

test("the plugin runs before vite:css and only on KaTeX's stylesheet", () => {
  const plugin = katexWoff2Only();
  assert.equal(plugin.enforce, "pre");
  const transform = plugin.transform as {
    filter: { id: RegExp; code: string };
    handler: (code: string, id: string) => { code: string } | null;
  };
  const { id } = transform.filter;
  for (const path of [
    "/repo/studio/frontend/node_modules/katex/dist/katex.min.css",
    "C:\\repo\\studio\\frontend\\node_modules\\katex\\dist\\katex.min.css",
    "/repo/node_modules/katex/dist/katex.css?direct",
    KATEX_CSS_PATH,
  ]) {
    assert.match(path, id, path);
  }
  for (const path of [
    "/repo/studio/frontend/src/index.css",
    "/repo/node_modules/katex/dist/contrib/copy-tex.css",
    "/repo/node_modules/not-katex/dist/katex.min.css",
  ]) {
    assert.doesNotMatch(path, id, path);
  }
  const result = transform.handler(KATEX_CSS, KATEX_CSS_PATH);
  assert.ok(result);
  assert.equal(result.code, keepWoff2Sources(KATEX_CSS).css);
  assert.throws(() => transform.handler(".katex{color:red}", KATEX_CSS_PATH), /no KaTeX @font-face/);
});
