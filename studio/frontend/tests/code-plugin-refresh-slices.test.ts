// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import { createHighlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

import {
  createCodePlugin,
  MIN_INCREMENTAL_CHARS,
  TOKENIZE_LIMITS,
} from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
const LANGUAGE = "typescript" as HighlightOptions["language"];

const SOURCE = `${Array.from(
  { length: 260 },
  (_, index) =>
    `export const value_${index} = { id: ${index}, label: "row ${index}" };`,
).join("\n")}\n`;

const text = (result: HighlightResult): string =>
  result.tokens.map((line) => line.map((token) => token.content).join("")).join("\n");

test("a refresh that falls behind the stream is tokenized in slices and still settles exact", async () => {
  const reference = await createHighlighter({
    themes: THEMES,
    langs: ["typescript"],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  const expected = reference.codeToTokens(SOURCE, {
    lang: "typescript",
    themes: { light: "github-light", dark: "github-dark" },
    ...TOKENIZE_LIMITS,
  }).tokens;
  const plugin = createCodePlugin({ themes: THEMES });
  const prefix = SOURCE.slice(0, SOURCE.indexOf("\n", MIN_INCREMENTAL_CHARS + 500) + 1);
  await new Promise<void>((resolve) => {
    const immediate = plugin.highlight(
      { code: prefix, language: LANGUAGE, themes: THEMES },
      () => resolve(),
    );
    if (immediate) resolve();
  });

  const realNow = performance.now.bind(performance);
  const base = realNow();
  let ticks = 0;
  performance.now = () => base + (ticks += 3);
  const deliveries: HighlightResult[] = [];
  try {
    const settled = await new Promise<HighlightResult>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("the sliced refresh never settled")), 10_000);
      const immediate = plugin.highlight(
        { code: SOURCE, language: LANGUAGE, themes: THEMES },
        (result) => {
          deliveries.push(result);
          if (result.tokens.length === expected.length && JSON.stringify(result.tokens) === JSON.stringify(expected)) {
            clearTimeout(timer);
            resolve(result);
          }
        },
      );
      assert.ok(immediate, "a throttled frame still answers at once");
      assert.equal(text(immediate), SOURCE, "the throttled frame renders every character");
    });
    assert.deepEqual(settled.tokens, expected);
  } finally {
    performance.now = realNow;
  }

  assert.ok(deliveries.length > 2, `the refresh ran as ${deliveries.length} slice(s), not several`);
  for (const result of deliveries) {
    assert.equal(text(result), SOURCE, "every slice renders every character, coloured or plain");
  }
});
