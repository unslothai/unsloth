// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";

const HORIZONTAL_RULE = 'data-streamdown="horizontal-rule"';
const UNORDERED_LIST = 'data-streamdown="unordered-list"';
const STRONG = 'data-streamdown="strong"';

function render(markdown: string): string {
  return renderToStaticMarkup(
    createElement(Streamdown, { mode: "streaming" }, markdown),
  );
}

test("buffers the thematic-break prefix of a bold asterisk list item", () => {
  const markdown = "* **";
  assert.ok(render(markdown).includes(HORIZONTAL_RULE));

  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  assert.equal(stabilized, "");
  assert.ok(!render(stabilized).includes(HORIZONTAL_RULE));
});

test("buffers triple-emphasis list prefixes", () => {
  for (const markdown of ["* ***", "* ****", "* *****"]) {
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), "");
  }

  const markdown = "* ***Heading***";
  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  assert.equal(stabilized, markdown);
  assert.ok(render(stabilized).includes(UNORDERED_LIST));
  assert.ok(!render(stabilized).includes(HORIZONTAL_RULE));
});

test("buffers tab-separated list prefixes", () => {
  for (const separator of ["\t", " \t", "  \t", "\t ", " \t "]) {
    const markdown = `*${separator}**`;
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), "");

    const completed = `*${separator}**Heading**`;
    const stabilized = stabilizeStreamingMarkdown(completed, true);
    const html = render(stabilized);
    assert.equal(stabilized, completed);
    assert.ok(html.includes(UNORDERED_LIST));
    assert.ok(html.includes(STRONG));
    assert.ok(!html.includes(HORIZONTAL_RULE));
  }
});

test("keeps tab-indented code prefixes unchanged", () => {
  for (const markdown of ["*\t\t**", "*   \t**", "*\t  **"]) {
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }
});

test("reveals the list as soon as bold content arrives", () => {
  const markdown = "* **Heading";
  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  const html = render(stabilized);

  assert.equal(stabilized, markdown);
  assert.ok(html.includes(UNORDERED_LIST));
  assert.ok(html.includes(STRONG));
  assert.ok(!html.includes(HORIZONTAL_RULE));
});

test("keeps completed thematic breaks unchanged", () => {
  const markdown = "* **";
  assert.equal(stabilizeStreamingMarkdown(markdown, false), markdown);
  assert.ok(render(markdown).includes(HORIZONTAL_RULE));
});

test("buffers only the ambiguous trailing line", () => {
  assert.equal(stabilizeStreamingMarkdown("* First\n* **", true), "* First\n");
  assert.equal(
    stabilizeStreamingMarkdown("* Parent\n  * **", true),
    "* Parent\n",
  );
  assert.equal(
    stabilizeStreamingMarkdown("- Parent\n    * **", true),
    "- Parent\n",
  );
  assert.equal(
    stabilizeStreamingMarkdown("- Parent\n        * **", true),
    "- Parent\n        * **",
  );
});

test("buffers ambiguous items relative to their blockquote container", () => {
  for (const markdown of [
    "> * **",
    "> > * **",
    "   > * **",
    "- Parent\n    > * **",
  ]) {
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(
      stabilizeStreamingMarkdown(markdown, true),
      markdown.includes("\n") ? "- Parent\n" : "",
    );
  }

  const markdown = "> * **Heading";
  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  const html = render(stabilized);
  assert.equal(stabilized, markdown);
  assert.ok(html.includes(UNORDERED_LIST));
  assert.ok(html.includes(STRONG));
  assert.ok(!html.includes(HORIZONTAL_RULE));
});

test("recognizes CommonMark line endings", () => {
  for (const [markdown, expected] of [
    ["First\r* **", "First\r"],
    ["First\r\n* **", "First\r\n"],
    ["First\n* **", "First\n"],
  ] as const) {
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), expected);
  }
});

test("preserves ambiguous-looking content in list-contained fences", () => {
  const unchanged = [
    "- ~~~markdown\n  * **",
    "1. ```markdown\n   * **",
    "> - ~~~markdown\n>   * **",
  ];

  for (const markdown of unchanged) {
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }
});

test("preserves ambiguous-looking content in raw HTML blocks", () => {
  const unchanged = [
    "<pre>\n* **",
    "<SCRIPT>\n* **",
    "<!-- example\n* **",
    "<div>\n* **",
  ];

  for (const markdown of unchanged) {
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }

  assert.equal(
    stabilizeStreamingMarkdown("<div>\ntext\n\n* **", true),
    "<div>\ntext\n\n",
  );
});

test("does not reinterpret adjacent Markdown constructs", () => {
  const unchanged = [
    "- **",
    "+ **",
    "1. **",
    "***",
    "* *",
    "* __",
    "released in\n1976.",
    "    * **",
    "```markdown\n* **",
    "~~~markdown\n* **",
    "$$\n* **",
  ];

  for (const markdown of unchanged) {
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }
});
