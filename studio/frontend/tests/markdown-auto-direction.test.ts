// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { rehypeAutoDirection } from "../src/components/assistant-ui/rehype-auto-direction.ts";
import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";

import { readSrc } from "./helpers/kit.ts";

function render(markdown: string): string {
  return renderToStaticMarkup(
    createElement(Streamdown, {
      mode: "static",
      children: markdown,
      rehypePlugins: withDataImageSupport({}, [rehypeAutoDirection]),
    }),
  );
}

function openingTags(html: string, tag: string): string[] {
  return html.match(new RegExp(`<${tag}(?:\\s[^>]*)?>`, "g")) ?? [];
}

test("each text block lays itself out from its own first strong character", () => {
  const html = render(
    [
      "## سلام",
      "",
      "این یک پاسخ فارسی است.",
      "",
      "An English paragraph.",
      "",
      "- مورد اول",
      "- مورد دوم",
      "",
      "1. one",
      "",
      "| ستون | column |",
      "| --- | --- |",
      "| مقدار | value |",
    ].join("\n"),
  );
  for (const tag of ["h2", "p", "ul", "ol", "th", "td"]) {
    const tags = openingTags(html, tag);
    assert.ok(tags.length > 0, `no <${tag}> rendered`);
    for (const opening of tags) {
      assert.match(opening, /\sdir="auto"/, opening);
    }
  }
  // The list decides for its items, so an item never carries its bullet away from the others.
  for (const opening of openingTags(html, "li")) {
    assert.doesNotMatch(opening, /\sdir=/, opening);
  }
});

test("code keeps the document's direction", () => {
  const html = render(
    "```python\n# توضیح\nprint('سلام')\n```\n\nبا `کد` درون متن.",
  );
  for (const tag of ["pre", "code"]) {
    const tags = openingTags(html, tag);
    assert.ok(tags.length > 0, `no <${tag}> rendered`);
    for (const opening of tags) {
      assert.doesNotMatch(opening, /\sdir=/, opening);
    }
  }
  assert.match(html, /<p dir="auto"/);
});

test("a direction the reply wrote in raw HTML is kept", () => {
  const html = render('<p dir="ltr">سلام</p>');
  assert.deepEqual(openingTags(html, "p"), ['<p dir="ltr">']);
});

test("chat replies, sent prompts and right-to-left lists are wired to it", () => {
  // The renderer every reply and thinking fragment goes through.
  assert.match(
    readSrc("components/assistant-ui/markdown-text.tsx"),
    /withDataImageSupport\(STREAMDOWN_ALLOWED_TAGS, \[[^\]]*\][^\]]*rehypeAutoDirection,\s*\]\)/,
  );
  // A sent prompt reads as it did in the composer, which is dir="auto" too.
  assert.match(
    readSrc("components/assistant-ui/thread.tsx"),
    /<div\s+dir="auto"\s+className="aui-user-message-content /,
  );
  // A physical padding-left would leave a right-to-left list's bullets hanging off the right edge,
  // so no list rule may use one, the wider indents of long ordered lists included.
  const css = readSrc("index.css");
  for (const list of ["unordered-list", "ordered-list"]) {
    const rule = css.match(
      new RegExp(`\\[data-streamdown="${list}"\\] \\{([^}]*)\\}`),
    )?.[1];
    assert.ok(rule, list);
    assert.match(rule, /padding-inline-start:/, list);
    for (const [, body] of css.matchAll(
      new RegExp(`\\[data-streamdown="${list}"\\][^{]*\\{([^}]*)\\}`, "g"),
    )) {
      assert.doesNotMatch(body, /padding-left:/, list);
    }
  }
});
