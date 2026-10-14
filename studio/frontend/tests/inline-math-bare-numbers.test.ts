// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createMathPlugin } from "@streamdown/math";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import { normalizeEscapedInlineMath } from "../src/lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const math = createMathPlugin({ singleDollarTextMath: true });

function render(source: string): string {
  return renderToStaticMarkup(
    React.createElement(
      Streamdown,
      { mode: "streaming", isAnimating: false, plugins: { math } },
      stabilizeStreamingMarkdown(
        preprocessLaTeX(normalizeEscapedInlineMath(source)),
        false,
      ),
    ),
  );
}

function renderedMath(html: string): string[] {
  return [
    ...html.matchAll(
      /<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g,
    ),
  ].map((match) => match[1]);
}

function visibleText(html: string): string {
  return html
    .replace(/<span class="katex">[\s\S]*?<\/annotation>/g, "")
    .replace(/<[^>]+>/g, "")
    .replaceAll("&#x27;", "'")
    .replaceAll("&#39;", "'")
    .replaceAll("&quot;", '"');
}

test("numbers written as inline maths render as maths", () => {
  const cases: Array<[string, string[]]> = [
    ["There are $3$ red balls and $5$ blue balls.", ["3", "5"]],
    ["The answer is $42$.", ["42"]],
    ["The answer is $42$", ["42"]],
    ["The answer is $42$!", ["42"]],
    ["Is it $3$?", ["3"]],
    ["It's $3$!", ["3"]],
    ["Users' answer is $3$!", ["3"]],
    ["The answer is $42$[source](https://example.com).", ["42"]],
    ["The answer is $42$[^1].\n\n[^1]: Source.", ["42"]],
    ["Step 2: $8$ times $2$ is $16$.", ["8", "2", "16"]],
    [
      "Janet has $16$ eggs. She eats $3$ and bakes with $4$, so $16 - 3 - 4 = 9$ remain.",
      ["16", "3", "4", "16 - 3 - 4 = 9"],
    ],
    ["About $1,000$ rows and $0.5$ of them.", ["1,000", "0.5"]],
  ];
  for (const [source, expected] of cases) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), expected, source);
    assert.ok(!visibleText(html).includes("$"), `${source} showed a dollar`);
  }
});

test("currency next to numbers stays text", () => {
  for (const source of [
    "Prices run from $5 to $10 in that range.",
    "It costs $5, $10 or $15.",
    "Between $5-$10 per run.",
    "It costs $1,000 today.",
    "Pay $3.50 for one and $1.25 for the next.",
    "Around $5K then $3.5M.",
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
  }
});

test("a shell parameter right after a positional one stays text", () => {
  for (const [source, literal] of [
    ['Run echo "$1$HOME" to print both.', "$1$HOME"],
    ["Use $1$ext in the replacement.", "$1$ext"],
    ['Run echo "$1$x" to print both.', "$1$x"],
    ['Run echo "$1$_" to print both.', "$1$_"],
    ['Run echo "$1$_name" to print both.', "$1$_name"],
    ['Run echo "$1$?" to print both.', "$1$?"],
    ['Run echo "$1$#" to print both.', "$1$#"],
    ['Run echo "$1$@" to print both.', "$1$@"],
    ['Run echo "$1$*" to print both.', "$1$*"],
    ['Run echo "$1$-" to print both.', "$1$-"],
    ['Run echo "$1$!" to print both.', "$1$!"],
    ['Run echo "$1$0" to print both.', "$1$0"],
    ['Run echo "$1$$" to print both.', "$1$$"],
    ['Run echo "$1${2}" to print both.', "$1${2}"],
    ['Run echo "$1${?}" to print both.', "$1${?}"],
    ['Run echo "$1${_}" to print both.', "$1${_}"],
    ['Run echo "$1${foo_bar}" to print both.', "$1${foo_bar}"],
    ['Run echo "$1${foo:-${bar}}" to print both.', "$1${foo:-${bar}}"],
    ['Run echo "$1$(date)" to print both.', "$1$(date)"],
    ['Run echo "$1$((n + 1))" to print both.', "$1$((n + 1))"],
    ['Run echo "$1$[n + 1]" to print both.', "$1$[n + 1]"],
    [`Run echo "$1$'value'" to print both.`, `$1$'value'`],
    [`Run echo '$1$"localized"' to print both.`, `$1$"localized"`],
    ["Run echo $1$# to print both.", "$1$#"],
    ["Run echo $1$@ to print both.", "$1$@"],
    ["Run echo $1$* to print both.", "$1$*"],
    ["Run echo $1$- to print both.", "$1$-"],
    ["Run echo $1$[n+1] to print both.", "$1$[n+1]"],
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    assert.ok(visibleText(html).includes(literal), source);
  }

  const combined =
    'Shell examples "$1$x", "$1$?", "$1${2}", and "$1$$" stay literal.';
  const html = render(combined);
  assert.deepEqual(renderedMath(html), [], combined);
  for (const literal of ["$1$x", "$1$?", "$1${2}", "$1$$"]) {
    assert.ok(
      visibleText(html).includes(literal),
      `${combined} lost ${literal}`,
    );
  }
});

test("raw-text element bodies keep adjacent shell syntax byte-for-byte", () => {
  for (const source of [
    '<script>const value = "$1$x";</script>',
    '<style>var { value: "$1${2}"; }</style>',
  ]) {
    assert.equal(preprocessLaTeX(source), source);
  }
});

test("a closing shell-word quote does not consume later Markdown", () => {
  const source = 'echo "$1$" and **then** echo "ok"';
  const html = render(source);
  assert.deepEqual(renderedMath(html), ["1"]);
  assert.match(html, /data-streamdown="strong">then<\/span>/);
});

test("shell quote state carries across long and multiline words", () => {
  for (const source of [
    `echo "${"x".repeat(512)}$1$?"`,
    `echo "first line\n${"x".repeat(512)}$1$?"`,
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    assert.ok(visibleText(html).includes("$1$?"), source);
  }
});

test("a trailing numeric closer stays literal while streaming", () => {
  assert.equal(preprocessLaTeX("$1$", true), "\\$1$");
  assert.equal(preprocessLaTeX("$1$", false), "$1$");
});

test("many unterminated substitutions stay within the streaming budget", () => {
  const source = "$1$(".repeat(51_200);
  const started = performance.now();
  preprocessLaTeX(source, true);
  assert.ok(performance.now() - started < 250);
});
