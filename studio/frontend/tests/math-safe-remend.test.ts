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
];

const STRIPPED: Array<[string, string]> = [
  ["text <custom", "text"],
  ["$$a$$ then <div", "$$a$$ then"],
  ["price \\$5 and <span", "price \\$5 and"],
  ["```\n$HOME\n```\n\n<div class", "```\n$HOME\n```"],
  ["`$x` then <b", "`$x` then"],
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

test("an explicit htmlTags: false is kept", () => {
  assert.equal(
    remend("text <custom", mathSafeRemendOptions({ htmlTags: false })),
    "text <custom",
  );
});
