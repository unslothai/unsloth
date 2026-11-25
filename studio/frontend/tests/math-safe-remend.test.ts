// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FINISHED_TEXT_REMEND,
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
