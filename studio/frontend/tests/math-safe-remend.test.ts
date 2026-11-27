// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FINISHED_TEXT_REMEND,
  IncrementalMarkdownCache,
  LITERAL_LINK_REMEND,
  repairStreamingMarkdown,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { mathSafeRemendOptions } from "../src/lib/math-safe-remend.ts";
import remend from "remend";

const remendMathSafe = (text: string, options?: Parameters<typeof remend>[1]) =>
  remend(text, mathSafeRemendOptions(options));

const KEPT = [
  "$$p_\\theta(x) = \\prod_i p_\\theta(x_i \\mid x_{<i})$$",
  "Intro:\n\n$$p_\\theta(x) = \\prod_i p_\\theta(x_i \\mid x_{<i})$$",
  "Inline $x_{<i}$",
  "$$\nx_{<i}\n$$",
  "\\(x_{<i}\\)",
  "\\[x_{</i}\\]",
  "$$a<b",
  "```\n$HOME\n```\n\n$$x<y$$",
  "`$x` and $y<z$",
  "use `a<b",
  "x `<span",
  "use `a<b` and <div",
  "```html\n<div class",
  "The $ sign.\n\nThen $x_{<i}$ ok",
  "The $ sign.\n- Then $x_{<i}$ ok",
  "The $ sign.\n1. Then $x_{<i}$ ok",
  "## What does $ mean?\nThen $x_{<i}$ ok",
  "Intro.\n## What $ is\nThen $x_{<i}$ ok",
  "The $ sign.\n# Then $x_{<i}$ ok",
  "- Prices in US$.\n---\nThen $x_{<i}$ ok",
  "Prices in US$.\n***\nThen $x_{<i}$ ok",
  "Prices in US$.\n___\nThen $x_{<i}$ ok",
  "- Let $y\n=\nx_{<i}$ ok",
  "Let $a\n-*-\nb_{<i}$ ok",
  "Let $a\n**\nb_{<i}$ ok",
  "Let $a\n__\nb_{<i}$ ok",
  "- Let $y\n--\nx_{<i}$ ok",
  "| Item | Cost ($) |\n|---|---|\n| A | $x_{<i}$ |",
  "Prices in US$.\n$x_{<i}$ | Q\n--- | ---\n1 | 2",
  "| Metric\n| ---\n| US$\n| $x_{<i}$",
  "## Condition $p(x_{<i}|y)$\n|---|---|",
  "| Model | Factorization |\n|---|---|\n| AR | $p(x_i|x_{<i})$ |\n\nDone.",
  "Use $$ for display math and $x_{<i}$ for inline math.",
  "| Conditional $p(x_{<i}|y)$ | Value |\n| --- | --- |\n| a | 1 |",
  "| a | b |\n|---|---|\n| US$ | 1 |\n| 2 | $x_{<i}$ |",
  "Intro.\n\n| a | b |\n|---|---|\n| US$ | 1 |\n| 2 | $x_{<i}$ |",
  "a | b\n--- | ---\n1 | US$\n$x_{<i}$ | 2",
  "| a |\n|---|\n| 1 |\n\nLet $a +\nx_{<i}$ ok",
  "Let $a\n***b***\nc_{<i}$ ok",
  "Let $a +\n#tag\nx_{<i}$ ok",
  "The $ sign.\n```\nx\n```\nThen $x_{<i}$ ok",
  "> The $ sign.\n>\n> Then $x_{<i}$ ok",
  "## T\n\nLet $a +\nx_{<i}$ ok",
  "Let $a +\n    - x_{<i}$ ok",
  "- Let $a_{<i}\n  + \n  b_i$ ok",
  "Let $a\n    >\n    b_{<i}$ ok",
  "> Let $a\n>     >\n>     b_{<i}$ ok",
  "> > The $ sign.\n> >\n> > Then $x_{<i}$ ok",
  "A dollar sign ($) opens it.\n> Then $x_{<i}$ ok",
  "> Let $a\nlazy\n> b_{<i}$ ok",
  "Intro.\n\n> Let $a\nlazy\n> b_{<i}$ ok",
  "1. > Let $p(x_i\n   > \\mid x_{<i})$ ok",
  "Intro.\n\n1. > Let $p(x_i\n   > \\mid x_{<i})$ ok",
  "Wrap display math in $$.\n\n$$\nx_{<i}\n$$",
  "Wrap display math in $$.\n$$\nx_{<i}\n$$",
  "Wrap display math in $$.\n$$\\displaystyle\nx_{<i}\n$$",
  "The terms are $$a+b\n$$ and $$x_{<i}$$ ok",
  "> $$\n> a\n>\n> x_{<i} $$",
  "- $$\n  a\n\n  x_{<i}\n  $$",
  "Let $a +\n    # b\nx_{<i}$ ok",
  "~~~bash\necho $HOME\n~~~\nThen $x_{<i}$ ok",
  "Use $x_{<i}$`c` ok",
  "Let $a +\nx_{<i}$ ok",
  "Let $a \\le\n|x_{<i}|$ ok",
  "Let $a =\n2. x_{<i}$ ok",
  "> Let $p(x_i\n> \\mid x_{<i})$ ok",
  "It costs \\$5. Then $x_{<i}$ ok",
  "$$a$$ and $x_{<i}$ ok",
  "~~~\necho $HOME\n~~~\n\n\\(x_{<i}\\) ok",
  "$$\na\n\nx_{<i}\n$$",
];

const STRIPPED: Array<[string, string]> = [
  ["text <custom", "text"],
  ["$$a$$ then <div", "$$a$$ then"],
  ["price \\$5 and <span", "price \\$5 and"],
  ["```\n$HOME\n```\n\n<div class", "```\n$HOME\n```"],
  ["`$x` then <b", "`$x` then"],
  ["$$x_{<i}$$ then <div", "$$x_{<i}$$ then"],
  ["$x<y$ and $a<b$ <span", "$x<y$ and $a<b$"],
  ["\\(a<b\\) and <em", "\\(a<b\\) and"],
];

test("a < inside math is not cut as an incomplete HTML tag", () => {
  for (const text of KEPT) {
    for (const options of [
      undefined,
      FINISHED_TEXT_REMEND,
      LITERAL_LINK_REMEND,
    ]) {
      const out = remendMathSafe(text, options);
      assert.ok(
        out.startsWith(text),
        `${JSON.stringify(text)} -> ${JSON.stringify(out)}`,
      );
    }
    assert.ok(
      repairStreamingMarkdown(text).startsWith(text),
      JSON.stringify(text),
    );
  }
});

test("a streamed formula with < survives the incremental tail repair", () => {
  for (const text of [
    "Intro:\n\n$$p(x) = \\prod_i p(x_i \\mid x_{<i})$$",
    `**Context.**${"\n\nShort paragraph.".repeat(16)}\n\nThe term $x_{<i}$.`,
  ]) {
    const cache = new IncrementalMarkdownCache();
    let markdown = "";
    for (let n = 1; n <= text.length; n++) {
      markdown = cache.update(text.slice(0, n)).markdown;
    }
    assert.ok(markdown.endsWith(text.slice(-8)), JSON.stringify(markdown));
  }
});

test("a trailing HTML tag outside math is still stripped", () => {
  for (const [text, expected] of STRIPPED) {
    assert.equal(remendMathSafe(text), expected);
    assert.equal(
      remend(text, mathSafeRemendOptions(FINISHED_TEXT_REMEND)),
      expected,
    );
  }
});

test("output matches remend's own pass whenever the cut is not inside math", () => {
  for (const text of [
    "text <custom",
    "a <b>bold</b> and <i",
    "use `a<b` and <div",
    "```\n<div\n",
    "line\n\n<details><summary>Hi",
    "costs \\$5 <span",
    "- item <br",
    "Run `cat $HOME/x < in.txt` then echo $PATH <done",
    "Inline $x_{<i",
    "and \\[a<b",
    "and \\(a<b",
    "price $5 and <span",
    "## Using $$ for display\n\nadd <span class=",
  ]) {
    for (const options of [undefined, FINISHED_TEXT_REMEND, LITERAL_LINK_REMEND]) {
      assert.equal(remendMathSafe(text, options), remend(text, options));
    }
  }
});

test("an explicit htmlTags: false is kept", () => {
  assert.equal(
    remend("text <custom", mathSafeRemendOptions({ htmlTags: false })),
    "text <custom",
  );
});
