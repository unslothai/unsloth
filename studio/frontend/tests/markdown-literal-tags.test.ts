// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import {
  withDataImageSupport,
  withLiteralUnknownTags,
} from "../src/lib/markdown-data-images.ts";

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

function renderDocument(markdown: string) {
  return renderToStaticMarkup(
    createElement(Streamdown, {
      mode: "static",
      children: markdown,
      rehypePlugins: withLiteralUnknownTags(),
    }),
  );
}

function escaped(text: string) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

test("placeholders and generic types in prose stay visible", () => {
  for (const mode of ["static", "streaming"] as const) {
    for (const line of [
      "Replace <your-api-key> with the key from your dashboard.",
      "Usage: git clone <repo-url> <directory>",
      "Use Vec<T> here, or List<String> in Java.",
      "Save it to /home/<user>/models/<model-name>/ then restart.",
      "The <script> tag loads JS. More text follows.",
      "The <small> tag shrinks text, <u> underlines it.",
      "Wrap the image in a <figure> with a <figcaption>.",
      "For all 0<x and x>1 we have a bound.",
      "<your-api-key>",
      "Use Vec<T> and close </T> here.",
      "   <your-api-key>",
    ]) {
      const html = render(line, mode);
      assert.ok(html.includes(`>${escaped(line)}<`), `${mode}: ${html}`);
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
      "<div>Use <T>x</T> in Vec<Vec<T>> here</div>",
      "Use <T>x</T> in Vec<Vec<T>> here",
    ],
    [
      "<details>\n<summary>Setup</summary>\nReplace <your-api-key> in /home/<user>/.env\n</details>",
      "Replace <your-api-key> in /home/<user>/.env",
    ],
  ]) {
    const html = render(block);
    assert.ok(html.includes(escaped(line)), html);
    assert.doesNotMatch(html, /<(your-api-key|code_snippet)/);
  }
});

test("allowed HTML tags still render as elements", () => {
  assert.match(
    render("Press <kbd>Ctrl</kbd>+<kbd>C</kbd>."),
    /<kbd[^>]*>Ctrl<\/kbd>/,
  );
  assert.match(render("H<sub>2</sub>O"), /<sub[^>]*>2<\/sub>/);
  assert.match(render("Press <KBD>Ctrl</KBD>."), /<kbd[^>]*>Ctrl<\/kbd>/);
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
});

test("documents unwrap formatting tags used as markup", () => {
  assert.match(
    renderDocument("MMBench<sub><small>EN-DEV</small></sub> and <U>under</U>"),
    /<p>MMBench<sub[^>]*>EN-DEV<\/sub> and under<\/p>/,
  );
  for (const [block, line] of [
    ["<figure>\n<figcaption>Caption</figcaption>\n</figure>", "Caption"],
    ["<center>Use <your-api-key> here</center>", "Use <your-api-key> here"],
    ["<center>\nCentered text", "Centered text"],
    ["Intro\n\n<figure>\n<img src=x>", "Intro"],
    ["<table><tr><td>MMBench<sub><small>EN</td></tr></table>", "MMBench"],
    [
      '<div>The <abbr title="x">API</abbr> takes <T></div>',
      "The API takes <T>",
    ],
  ]) {
    const html = renderDocument(block);
    assert.ok(html.includes(escaped(line)), html);
    assert.doesNotMatch(
      html,
      /(<|&lt;)\/?(figure|figcaption|center|abbr|small)\b/,
      html,
    );
  }
  for (const [markdown, expected] of [
    [
      "Use <mark>text</mark>, e.g. <mark>.",
      "<p>Use text, e.g. &lt;mark&gt;.</p>",
    ],
    ["The <u> tag, as in <u>x</u>.", "<p>The &lt;u&gt; tag, as in x.</p>"],
    ["<u>a <u>b</u> c</u> and </u>", "<p>a b c and &lt;/u&gt;</p>"],
    [
      "The <small> tag shrinks text.",
      "<p>The &lt;small&gt; tag shrinks text.</p>",
    ],
    [
      "The <small> tag.\n\nLater: </small> here.",
      "<p>The &lt;small&gt; tag.</p>\n<p>Later: &lt;/small&gt; here.</p>",
    ],
  ]) {
    const html = renderDocument(markdown);
    assert.ok(html.includes(expected), html);
  }
});

test("chat replies keep formatting tags as text", () => {
  for (const line of [
    "Use <cite>Title</cite> for works.",
    "Open it with <small> and close it with </small>.",
  ]) {
    const html = render(line);
    assert.ok(html.includes(`>${escaped(line)}<`), html);
  }
  for (const [block, line] of [
    [
      "<div>The <small> tag shrinks text.</div>",
      "The <small> tag shrinks text.",
    ],
    [
      "<p>Wrap it in <figure> with a <figcaption>.</p>",
      "Wrap it in <figure> with a <figcaption>.",
    ],
  ]) {
    const html = render(block);
    assert.ok(html.includes(escaped(line)), html);
  }
});

test("withLiteralUnknownTags merges caller tags into the schema", () => {
  const html = renderToStaticMarkup(
    createElement(Streamdown, {
      mode: "static",
      children:
        '<video src="https://example.com/v.mp4" controls></video>\n\nPad with <unk> tokens.',
      rehypePlugins: withLiteralUnknownTags({ video: ["src", "controls"] }),
    }),
  );
  assert.match(html, /<video src="https:\/\/example\.com\/v\.mp4" controls/);
  assert.match(html, /Pad with &lt;unk&gt; tokens\./);
});

test("hostile markup never renders as markup", () => {
  for (const markdown of [
    "a <script>alert(1)</script> b",
    "<ScRiPt\n>alert(1)</script >",
    "x <img src=x onerror=alert(1)> y",
    'x <iframe src="javascript:alert(1)"></iframe> y',
    "<svg onload=alert(1)><circle r=5></circle></svg>",
    '<a href="javascript:alert(1)">x</a>',
    "<style>*{display:none}</style>",
    "<div><script>alert(1)</script><img src=x onerror=alert(1)><svg onload=alert(1)></div>",
    '<details open ontoggle=alert(1)><summary>s</summary><iframe srcdoc="<script>alert(1)</script>"></iframe></details>',
    "<custom-wrapper>\n<b>b</b> <img src=x onerror=alert(1)>\n</custom-wrapper>",
    '<p style="position:fixed;inset:0">overlay</p>',
    '<div>x <kbd style="position:fixed">k</kbd> <T></div>',
  ]) {
    for (const mode of ["static", "streaming"] as const) {
      const html = render(markdown, mode);
      assert.doesNotMatch(
        html,
        /<(script|iframe|svg|style|object|embed|form|math)\b/i,
        html,
      );
      assert.doesNotMatch(
        html,
        /<[a-z][^>]*\s(on[a-z]+|srcdoc|style)=|<[a-z][^>]*="\s*javascript:/i,
        html,
      );
    }
  }
});

test("literal tags keep their source position so streamed blocks re-render", () => {
  const [plugin, options] = withDataImageSupport(ALLOWED_TAGS)[0] as [
    (options: unknown) => (tree: unknown) => void,
    unknown,
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
  plugin(options)(tree);
  const [block, paragraph] = tree.children as {
    position?: unknown;
    children: { type: string; position?: unknown }[];
  }[];
  assert.deepEqual(block.position, position);
  assert.deepEqual(block.children[0].position, position);
  assert.equal(paragraph.children[0].type, "text");
  assert.deepEqual(paragraph.children[0].position, position);
});
