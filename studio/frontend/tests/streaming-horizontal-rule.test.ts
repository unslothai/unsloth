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

test("buffers an ambiguous item after a long accumulated response", () => {
  const preceding = `${"word ".repeat(20_000)}\n\n`;
  const markdown = `${preceding}* **`;

  assert.ok(render(markdown).includes(HORIZONTAL_RULE));
  assert.equal(stabilizeStreamingMarkdown(markdown, true), preceding);
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

test("preserves GFM footnote content that does not render a rule", () => {
  // An empty footnote definition renders no content at all, so buffering its
  // indented prefix is invisible. Assert the rendered output, not the string.
  const markdown = "See[^1]\n\n[^1]:\n    * **";
  const html = render(markdown);

  assert.ok(!html.includes(HORIZONTAL_RULE));
  assert.equal(render(stabilizeStreamingMarkdown(markdown, true)), html);

  const completed = `${markdown}Heading**`;
  const completedHtml = render(completed);
  assert.ok(completedHtml.includes(UNORDERED_LIST));
  assert.ok(completedHtml.includes(STRONG));
  assert.ok(!completedHtml.includes(HORIZONTAL_RULE));
});

test("buffers a populated GFM footnote that does render a rule", () => {
  const markdown = "See[^1]\n\n[^1]: text\n\n    * **";
  assert.ok(render(markdown).includes(HORIZONTAL_RULE));

  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  assert.equal(stabilized, "See[^1]\n\n[^1]: text\n\n");
  assert.ok(!render(stabilized).includes(HORIZONTAL_RULE));
});

test("is not defeated by literal dollar signs", () => {
  // Dollar parity alone mistook prices, shell vars and code spans for math and
  // silently disabled buffering for the rest of the response.
  for (const preceding of [
    "Price is $5",
    "Use `$HOME`",
    "```sh\necho $HOME\n```",
  ]) {
    const markdown = `${preceding}\n\n* **`;
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(
      stabilizeStreamingMarkdown(markdown, true),
      `${preceding}\n\n`,
    );
  }

  // Real math is still left alone. `render` here has no math plugin, unlike the
  // app, so only the decision can be asserted, not this helper's output.
  assert.equal(stabilizeStreamingMarkdown("$$\n* **", true), "$$\n* **");
});

test("keeps lines that Streamdown's incomplete-Markdown repair already fixes", () => {
  // remend closes the open construct, so these render as a list already and
  // buffering them would hide a line that was displaying correctly.
  for (const markdown of ["[open\n* **", "`open\n* **", "[open](\n* **"]) {
    assert.ok(!render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }

  // An unclosed bold still leaves a real rule to buffer.
  assert.ok(render("**open\n* **").includes(HORIZONTAL_RULE));
  assert.equal(stabilizeStreamingMarkdown("**open\n* **", true), "**open\n");
});

test("does not parse a degenerate punctuation run", () => {
  // Parsing a long run is quadratic, so a runaway line is left alone. Measured
  // rather than assumed: unbounded, 100k dashes took over 8 seconds.
  for (const run of ["-", "*", "_"]) {
    const markdown = `Intro.\n\n* ${run.repeat(100_000)}`;
    const started = performance.now();
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
    assert.ok(performance.now() - started < 250);
  }

  // A run a user could plausibly see is still well inside the bound.
  assert.equal(stabilizeStreamingMarkdown(`* ${"*".repeat(60)}`, true), "");
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

test("buffers dash and underscore runs, not only asterisks", () => {
  // A dash item whose content starts with dashes is the same ambiguity: CLI
  // flag lists ("- --verbose: ...") stream through a thematic-break frame.
  for (const markdown of [
    "- --",
    "- ---",
    "+ ---",
    "* ---",
    "- ___",
    "* ___",
  ]) {
    assert.ok(render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), "");
  }

  const markdown = "- --verbose";
  const stabilized = stabilizeStreamingMarkdown(markdown, true);
  const html = render(stabilized);
  assert.equal(stabilized, markdown);
  assert.ok(html.includes(UNORDERED_LIST));
  assert.ok(!html.includes(HORIZONTAL_RULE));
});

test("keeps runs that do not currently render a rule", () => {
  // Two characters never make a break, so nothing is flashing and nothing may
  // be hidden. Mixed runs are not breaks either.
  for (const markdown of ["- **", "+ **", "* --", "- *-*", "* -_-", "+ ++"]) {
    assert.ok(!render(markdown).includes(HORIZONTAL_RULE));
    assert.equal(stabilizeStreamingMarkdown(markdown, true), markdown);
  }
});
