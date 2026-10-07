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

function render(source: string, isStreaming = false): string {
  return renderToStaticMarkup(
    React.createElement(
      Streamdown,
      {
        mode: "streaming",
        isAnimating: isStreaming,
        plugins: { math },
      },
      stabilizeStreamingMarkdown(
        preprocessLaTeX(normalizeEscapedInlineMath(source)),
        isStreaming,
      ),
    ),
  );
}

function renderedMath(html: string): string[] {
  return [
    ...html.matchAll(
      /<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g,
    ),
  ].map((match) =>
    match[1]
      .replaceAll("&amp;", "&")
      .replaceAll("&lt;", "<")
      .replaceAll("&gt;", ">")
      .replaceAll("&quot;", '"')
      .replaceAll("&#x27;", "'"),
  );
}

const SHELL_VARIABLE_REPLIES: Array<[string, string[]]> = [
  [
    "Add the bin folder to $PATH, then check $HOME/.bashrc.",
    ["$PATH, then check $HOME/.bashrc."],
  ],
  [
    "Export $CUDA_HOME and $LD_LIBRARY_PATH before building.",
    ["$CUDA_HOME and $LD_LIBRARY_PATH"],
  ],
  ["Run echo $PATH\nthen echo $HOME", ["$PATH", "$HOME"]],
  ["Check $HOME, $PATH, and $USER first.", ["$HOME, $PATH, and $USER"]],
  ['Quote them: "$HOME" and "$PATH".', ["$HOME", "$PATH"]],
  ["Files live under $HOME/$USER/data.", ["$HOME/$USER/data"]],
  ["Use ${HOME} and ${PATH} in scripts.", ["${HOME} and ${PATH}"]],
  ["Set it to $PATH:$HOME/bin now.", ["$PATH:$HOME/bin"]],
  ["#define HOME $HOME\nthen $PATH", ["$HOME", "$PATH"]],
  ["PHP reads $_GET and $_POST.", ["$_GET and $"]],
  [
    "Add it to $PATH, then run `echo $HOME`.",
    ["$PATH, then run", "echo $HOME"],
  ],
];

test("shell variables in a reply render as text, not maths", () => {
  for (const [source, texts] of SHELL_VARIABLE_REPLIES) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    for (const text of texts) {
      assert.ok(html.includes(text), `${source} lost ${text}`);
    }
  }
});

test("shell variables never flash as maths while a reply streams", () => {
  for (const [source] of SHELL_VARIABLE_REPLIES) {
    for (let end = 1; end <= source.length; end += 1) {
      const prefix = source.slice(0, end);
      assert.deepEqual(renderedMath(render(prefix, true)), [], prefix);
    }
  }
});

const MATH_REPLIES: Array<[string, string[]]> = [
  ["$x$", ["x"]],
  ["In triangle $ABC$, the side $AB = 5$.", ["ABC", "AB = 5"]],
  ["The slope is $dy/dx$ and the line is $ax + b$.", ["dy/dx", "ax + b"]],
  ["Use $\\alpha + 1$ and $\\alpha$.", ["\\alpha + 1", "\\alpha"]],
  ["Lines $AB$ and $CD$ meet at $P$.", ["AB", "CD", "P"]],
  ["Segments $AB and CD$ are equal.", ["AB and CD"]],
  ["The $n$th element and the $k$th one.", ["n", "k"]],
  [
    "We have $sin x$ and $n log n$ and $sin theta$.",
    ["sin x", "n log n", "sin theta"],
  ],
  ["Points $x_1, x_2$ and $v_s$ and $a_{ij}$.", ["x_1, x_2", "v_s", "a_{ij}"]],
  ["So $E = mc^2$ and $f(x)$ and $O(n)$.", ["E = mc^2", "f(x)", "O(n)"]],
  ["Then $P(A and B)$ holds.", ["P(A and B)"]],
  ["Pick ${n \\choose k}$ ways.", ["{n \\choose k}"]],
  ["Write $ab/cd$ or $AB/CD$.", ["ab/cd", "AB/CD"]],
  ["Then $\\text{if } x > 0$ and $|x|$.", ["\\text{if } x > 0", "|x|"]],
  ["An angle of $30^\\circ$ here.", ["30^\\circ"]],
  ["**$90 - x$** is the rest.", ["90 - x"]],
  ["Let $x \\in A$ and $AB $ hold.", ["x \\in A", "AB "]],
  ["The value \\(\\beta\\) and $\\gamma$.", ["\\beta", "\\gamma"]],
  ["Sum $a +\nb$ over lines.", ["a +\nb"]],
  ["Segment $AB'$ and the derivative $uv'$.", ["AB'", "uv'"]],
  [
    "About $\\sim$2x faster: the $k$th token and the $n$th layer.",
    ["\\sim", "k", "n"],
  ],
  [
    "| Item | Price ($) |\n|---|---|\n| the $n$th row and the $m$th column | 5 |",
    ["n", "m"],
  ],
  ["# Cost in $\nThe $n$th row and the $m$th column.", ["n", "m"]],
  ["- Ends with $\n- The $n$th row and the $m$th column", ["n", "m"]],
  ["We have $sin theta $ and $AB / CD $ here.", ["sin theta ", "AB / CD "]],
  ["Use $HOME/$USER and $\\alpha$ here.", ["\\alpha"]],
];

test("real maths still renders", () => {
  for (const [source, expected] of MATH_REPLIES) {
    assert.deepEqual(renderedMath(render(source)), expected, source);
  }
});

test("maths and currency next to shell variables keep rendering", () => {
  const withMath = render("Set $x$ from $PATH, then $HOME.");
  assert.deepEqual(renderedMath(withMath), ["x"]);
  assert.ok(withMath.includes("$PATH, then $HOME."));

  const withCurrency = render("It costs $5 to set $PATH, then $HOME.");
  assert.deepEqual(renderedMath(withCurrency), []);
  assert.ok(withCurrency.includes("$5 to set $PATH, then $HOME."));
});
