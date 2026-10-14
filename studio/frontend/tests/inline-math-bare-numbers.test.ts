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
      { mode: "streaming", isAnimating: false, plugins: { math } },
      stabilizeStreamingMarkdown(
        preprocessLaTeX(normalizeEscapedInlineMath(source), isStreaming),
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
    ["$$42$$", ["42"]],
    ["The answer is $42$", ["42"]],
    ["The answer is $42$!", ["42"]],
    ["Is it $3$?", ["3"]],
    ["It's $3$!", ["3"]],
    ["Users' answer is $3$!", ["3"]],
    ['He said "The answer is $42$!"', ["42"]],
    ['He said ("The answer is $42$!")', ["42"]],
    ['She asked "Is it $3$?"', ["3"]],
    ['the answer is "$42$!"', ["42"]],
    ['consider this "$3$?"', ["3"]],
    ['- please note "The answer is $42$!"', ["42"]],
    ['- chapter 1.2 notes "The answer is $42$!"', ["42"]],
    ["The mass is $42$kg.", ["42"]],
    ["a $10$-fold increase", ["10"]],
    ["Range $1$-$2$", ["1", "2"]],
    ["The number $42$'s factors are 2, 3, and 7.", ["42"]],
    ["There are $3$*red* balls.", ["3"]],
    ["There are $3$_red_ balls.", ["3"]],
    ["There are $3$**red** balls.", ["3"]],
    ["There are $3$__red__ balls.", ["3"]],
    ["*$3$*", ["3"]],
    ["_$3$_", ["3"]],
    ["*There are $3$*", ["3"]],
    ["_There are $3$_", ["3"]],
    ["**There are $3$**", ["3"]],
    ["__There are $3$__", ["3"]],
    ["| value |\n| --- |\n| $42$! |", ["42"]],
    ["| value |\n| --- |\n| $10$-fold |", ["10"]],
    ["| case | value |\n| --- | --- |\n| answer | $42$! |", ["42"]],
    ["answer = $42$!", ["42"]],
    ['- consider "Is it $3$?"', ["3"]],
    ["The result is $42$(the answer).", ["42"]],
    ["The answer is $42$[source](https://example.com).", ["42"]],
    ["The answer is $42$[^1].\n\n[^1]: Source.", ["42"]],
    ['He said "The answer is $42$[source](https://example.com)."', ["42"]],
    [
      'Reference: "$42$[source](https://example.com)." Then he said "The answer is $42$!"',
      ["42", "42"],
    ],
    ['He said "$42$[^1]."\n\n[^1]: Source.', ["42"]],
    ["The answer is $42$[x+1][ref].\n\n[ref]: https://example.com", ["42"]],
    [
      "The answer is $42$[source-link].\n\n[source-link]: https://example.com",
      ["42"],
    ],
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
  assert.match(render("There are $3$*red* balls."), /<em>red<\/em>/);
  assert.match(render("There are $3$_red_ balls."), /<em>red<\/em>/);
  assert.match(
    render("There are $3$**red** balls."),
    /data-streamdown="strong">red<\/span>/,
  );
  assert.match(
    render("There are $3$__red__ balls."),
    /data-streamdown="strong">red<\/span>/,
  );
  assert.match(render("*$3$*"), /<em><span class="katex">/);
  assert.match(render("_$3$_"), /<em><span class="katex">/);
  assert.match(render("*There are $3$*"), /<em>There are <span class="katex">/);
  assert.match(
    render("**There are $3$**"),
    /data-streamdown="strong">There are <span class="katex">/,
  );
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
    ['cat "$1$?" to print both.', "$1$?"],
    ['cat -n "$1$?" to print both.', "$1$?"],
    ['widget --mode fast "$1$?" to print both.', "$1$?"],
    ['- Run echo "$1$?" to print both.', "$1$?"],
    ['- [ ] Run echo "$1$?" to print both.', "$1$?"],
    ['- [x] Run echo "$1$?" to print both.', "$1$?"],
    ['> echo "$1$?" to print both.', "$1$?"],
    ['> 1. Run echo "$1$?" to print both.', "$1$?"],
    ['Run echo "$1$#" to print both.', "$1$#"],
    ['Run echo "$1$@" to print both.', "$1$@"],
    ['Run echo "$1$*" to print both.', "$1$*"],
    ['echo "$1$*glob*" now.', "$1$*glob*"],
    ['echo "$1$_name_" now.', "$1$_name_"],
    ['echo prefix"$1$?" now.', "$1$?"],
    ["echo prefix'$1$?' now.", "$1$?"],
    ["echo prefix$1$(date) now.", "$1$(date)"],
    ["echo status:$1$? now.", "$1$?"],
    ["echo status+$1$? now.", "$1$?"],
    ["echo status,$1$? now.", "$1$?"],
    ['git show "$1$?" now.', "$1$?"],
    ['git show HEAD "$1$?" now.', "$1$?"],
    ['kubectl get pods "$1$?" now.', "$1$?"],
    ['widget --mode HEAD "$1$?" now.', "$1$?"],
    ['python script.py | grep -n "$1$?" now.', "$1$?"],
    ["Run echo \\" + '\n  "$1$?"', "$1$?"],
    ['sudo -u root echo "$1$?" now.', "$1$?"],
    ['env MODE=fast echo "$1$?" now.', "$1$?"],
    ['MODE=fast echo "$1$?" now.', "$1$?"],
    ['$ echo "$1$?" now.', "$1$?"],
    ["echo *$1$* now.", "*$1$*"],
    ["echo _$1$_ now.", "_$1$_"],
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
    ["Run echo $1$? to print both.", "$1$?"],
    ["value=$1$?", "$1$?"],
    ["echo x | $1$?", "$1$?"],
    ["Run echo $1$! to print both.", "$1$!"],
    ["Run echo $1$[n] to print both.", "$1$[n]"],
    ["Run echo $1$[1] to print both.", "$1$[1]"],
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

test("unterminated shell expansions stop at Markdown block boundaries", () => {
  const source = "Run echo $1${oops\n\n**later**";
  const html = render(source);
  assert.deepEqual(renderedMath(html), []);
  assert.ok(visibleText(html).includes("$1${oops"));
  assert.match(html, /data-streamdown="strong">later<\/span>/);
});

test("protected regions do not leak shell quote state into later prose", () => {
  for (const source of [
    '```sh\necho ok # \'\n```\n\nThe answer is "$42$" and **done**.',
    '<script>const fragment = "\'";</script>\n\nThe answer is "$42$" and **done**.',
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), ["42"], source);
    assert.match(html, /data-streamdown="strong">done<\/span>/, source);
  }
});

test("link destinations do not reset surrounding prose quote state", () => {
  const source =
    'There are $3$ red balls and $5$ blue balls, so $3 + 5 = 8$ in total. The answer is $8$. Shell examples "$1$x", "$1$?", "$1${2}", "$1$$", "$1$(date)", and "$1$((n+1))" stay literal. The answer is $42$! Is it $3$? Quote check: echo "$1$" and **then** echo "ok". Reference check: "$42$[source](https://example.com)." Prose check: He said "The answer is $42$!" after a $10$-fold increase across $1$-$2$.';
  const html = render(source);
  assert.deepEqual(renderedMath(html), [
    "3",
    "5",
    "3 + 5 = 8",
    "8",
    "42",
    "3",
    "1",
    "42",
    "42",
    "10",
    "1",
    "2",
  ]);
  assert.ok(visibleText(html).includes("$1$x"));
  assert.ok(visibleText(html).includes("$1$?"));
  assert.match(html, /data-streamdown="strong">then<\/span>/);
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
    `echo "first line\r\n${"x".repeat(512)}$1$?"`,
    '> Run echo "first line\n> $1$?"',
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    assert.ok(visibleText(html).includes("$1$?"), source);
  }
});

test("shell quote state resets at Markdown block boundaries", () => {
  for (const source of [
    '- Run echo "oops\n- The answer is $42$!',
    'Run echo "oops\n\nThe answer is $42$!',
    'Run echo "oops\n# The answer is $42$!',
    'Run echo "oops\n> The answer is $42$!',
    'Run echo "oops\n| value |\n| $42$! |',
    'Run echo "oops\n***\nThe answer is $42$!',
    'Run echo "oops\n---\nThe answer is $42$!',
    'Run echo "oops\n_ _ _\nThe answer is $42$!',
    'Run echo "oops\n===\nThe answer is $42$!',
    'Run echo "oops\r\rThe answer is $42$!',
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), ["42"], source);
  }
});

test("shell comments do not leak inert quotes into later prose", () => {
  const source = 'echo ok # "\nThe answer is $42$!';
  const html = render(source);
  assert.deepEqual(renderedMath(html), ["42"]);
});

test("a trailing numeric closer stays literal while streaming", () => {
  assert.equal(preprocessLaTeX("$1$", true), "\\$1$");
  assert.equal(preprocessLaTeX("$1$", false), "$1$");
  const source = "Run echo $1$[n+1]";
  for (
    let length = source.indexOf("$1$[") + 4;
    length < source.length;
    length++
  ) {
    const prefix = source.slice(0, length);
    const html = render(prefix, true);
    assert.deepEqual(renderedMath(html), [], prefix);
    assert.ok(visibleText(html).includes("$1$["), prefix);
  }
});

test("many unterminated substitutions stay within the streaming budget", () => {
  for (const source of ["$1$(".repeat(51_200), "$1$[".repeat(200_000)]) {
    const started = performance.now();
    preprocessLaTeX(source, true);
    // linear runs take ~200 ms on a loaded runner; a quadratic scan takes minutes.
    assert.ok(performance.now() - started < 2000);
  }
});

test("spaced inline math at a line start is not a shell prompt", () => {
  for (const [source, expected] of [
    ["$ x^2 + 1 $", ["x^2 + 1"]],
    ["The equation is:\n$ x = 3 $\nDone.", ["x = 3"]],
    ["$ x = 3 $!", ["x = 3"]],
    ["$ x = 3 $?", ["x = 3"]],
  ] as Array<[string, string[]]>) {
    assert.deepEqual(renderedMath(render(source)), expected, source);
  }
  const prompt = render("$ ls -la\n$ cd /tmp");
  assert.deepEqual(renderedMath(prompt), []);
  assert.ok(visibleText(prompt).includes("$ ls -la"));
});

test("nested emphasis at offset zero does not hang the emphasis scan", () => {
  for (const source of ["*There **are** $3$*", "_There __are__ $3$_"]) {
    const started = performance.now();
    preprocessLaTeX(source);
    assert.ok(performance.now() - started < 2000, source);
  }
});
