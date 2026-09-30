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
