// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { rehypeAutoDirection } from "../src/components/assistant-ui/rehype-auto-direction.ts";
import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";

function render(markdown: string): string {
  return renderToStaticMarkup(
    createElement(Streamdown, {
      mode: "static",
      children: markdown,
      rehypePlugins: withDataImageSupport({}, [rehypeAutoDirection]),
    }),
  );
}

test("text blocks get dir=auto through the chat's sanitize and harden pipeline", () => {
  const html = render(
    "## سرفصل\n\nبرای Unsloth راست به چپ\n\n| ستون |\n| --- |\n| خانه |",
  );
  for (const tag of ["h2", "p", "th", "td"]) {
    assert.match(html, new RegExp(`<${tag}[^>]* dir="auto"`), tag);
  }
});

test("lists and quotes flip only when their text starts right to left", () => {
  const rtl = render(
    "1. **مورد** اول\n2. دوم\n\n- مورد\n\n  با پاراگراف\n\n> نقل",
  );
  for (const tag of ["ol", "ul", "blockquote"]) {
    assert.match(rtl, new RegExp(`<${tag}[^>]* dir="rtl"`), tag);
  }
  const ltr = render("- `کد` item\n\n> quote سلام");
  assert.match(ltr, /<ul[^>]* dir="ltr"/);
  assert.match(ltr, /<blockquote[^>]* dir="ltr"/);
});

test("an English list nested in an RTL list stays left to right", () => {
  const html = render("- سلام\n  - English item.");
  assert.match(html, /<ul[^>]* dir="rtl"[\s\S]*<ul[^>]* dir="ltr"/);
});

test("a tight item's text gets its own direction, the item keeps the list's", () => {
  const html = render("- سلام\n- English item.\n  - nested");
  assert.match(html, /<ul[^>]* dir="rtl"/);
  assert.match(html, /<li[^>]*><span dir="auto">English item\.\s*<\/span>/);
  assert.doesNotMatch(html, /<li[^>]* dir=/);
});

test("code is pinned LTR in the stylesheet", () => {
  const css = readFileSync(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  assert.match(
    css,
    /\[data-streamdown="code-block"\], \[data-streamdown="inline-code"\][^{]*\{\s*direction: ltr;\s*unicode-bidi: isolate;/,
  );
});

test("code keeps its own direction", () => {
  const html = render("```python\nprint('سلام')\n```\n\nمتن `code` متن");
  assert.doesNotMatch(html, /<(pre|code)[^>]* dir=/);
});
