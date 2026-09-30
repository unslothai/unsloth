// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";

const ALLOWED_TAGS = { "search-image": ["token"] };

function render(markdown: string, mode: "static" | "streaming" = "static") {
  return renderToStaticMarkup(
    createElement(Streamdown, {
      mode,
      children: markdown,
      allowedTags: ALLOWED_TAGS,
      rehypePlugins: withDataImageSupport(ALLOWED_TAGS),
    }),
  );
}

function text(html: string) {
  return html
    .replace(/<[^>]*>/g, "")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}

test("placeholders and generic types in prose stay visible", () => {
  for (const mode of ["static", "streaming"] as const) {
    for (const line of [
      "Replace <your-api-key> with the key from your dashboard.",
      "Usage: git clone <repo-url> <directory>",
      "Use Vec<T> here, or List<String> in Java.",
      "Save it to /home/<user>/models/<model-name>/ then restart.",
      "The <script> tag loads JS. More text follows.",
      "For all 0<x and x>1 we have a bound.",
      "<your-api-key>",
    ]) {
      assert.equal(text(render(line, mode)).trim(), line, `${mode}: ${line}`);
    }
  }
  assert.match(
    render("Set the key:\n\n<your-api-key>\n\nThen restart."),
    /<p>&lt;your-api-key&gt;<\/p>/,
  );
  for (const [block, line] of [
    ["<div>Use <your-api-key> here</div>", "Use <your-api-key> here"],
    ["<div>Use <code_snippet> here</div>", "Use <code_snippet> here"],
    [
      "<details>\n<summary>Setup</summary>\nReplace <your-api-key> in /home/<user>/.env\n</details>",
      "Replace <your-api-key> in /home/<user>/.env",
    ],
  ]) {
    const html = render(block);
    assert.ok(text(html).includes(line), html);
    assert.doesNotMatch(html, /<(your-api-key|code_snippet)/);
  }
});

test("allowed HTML tags still render as elements", () => {
  assert.match(
    render("Press <kbd>Ctrl</kbd>+<kbd>C</kbd>."),
    /<kbd[^>]*>Ctrl<\/kbd>/,
  );
  assert.match(render("H<sub>2</sub>O"), /<sub[^>]*>2<\/sub>/);
  assert.match(render("| a |\n| - |\n| x<br>y |"), /x<br[^>]*>y/);
  const details = render(
    "<details>\n<summary>More</summary>\n\nHidden\n\n</details>",
  );
  assert.match(details, /<details[^>]*>/);
  assert.match(details, /<summary[^>]*>More<\/summary>/);
  assert.match(
    render('<search-image token="t1"></search-image>'),
    /<search-image token="t1"/,
  );
  assert.doesNotMatch(render("a <script>alert(1)</script> b"), /<script>/);
});

test("literal tags keep their source position so streamed blocks re-render", () => {
  const [plugin, tagNames] = withDataImageSupport(ALLOWED_TAGS)[0] as [
    (names: string[]) => (tree: unknown) => void,
    string[],
  ];
  const position = {
    start: { line: 3, column: 1, offset: 9 },
    end: { line: 4, column: 20, offset: 43 },
  };
  const tree = {
    type: "root",
    children: [
      {
        type: "raw",
        value: "<your-api-key>\nwhere the key comes from",
        position,
      },
      {
        type: "element",
        tagName: "p",
        properties: {},
        children: [{ type: "raw", value: "<T>", position }],
      },
    ],
  };
  plugin(tagNames)(tree);
  const [block, paragraph] = tree.children as {
    position?: unknown;
    children: { type: string; position?: unknown }[];
  }[];
  assert.deepEqual(block.position, position);
  assert.deepEqual(block.children[0].position, position);
  assert.equal(paragraph.children[0].type, "text");
  assert.deepEqual(paragraph.children[0].position, position);
});
