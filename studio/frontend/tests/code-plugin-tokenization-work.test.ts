// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Pins that a streaming fence is tokenized once, not per update. Waits out REFRESH_MS and counts
 * characters the plugin's own shiki import sees.
 */

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import { createHighlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

register("./shiki-tokenization-resolver.mjs", import.meta.url);
const { createCodePlugin, MIN_INCREMENTAL_CHARS, TOKENIZE_LIMITS } =
  await import("../src/components/assistant-ui/code-plugin.ts");
const { tokenized } = await import("./shiki-tokenization-counter.mts");

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
const LANGUAGE = "typescript" as HighlightOptions["language"];
const SETTLE_MS = 300;

const SOURCE = `${Array.from(
  { length: 260 },
  (_, index) =>
    `export const value_${index} = { id: ${index}, label: "row ${index}" };`,
).join("\n")}\n`;

const settle = () => new Promise((resolve) => setTimeout(resolve, SETTLE_MS));

const highlightOnce = (
  plugin: ReturnType<typeof createCodePlugin>,
  code: string,
): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const immediate = plugin.highlight(
      { code, language: LANGUAGE, themes: THEMES },
      resolve,
    );
    if (immediate) resolve(immediate);
  });

test("a streaming fence is tokenized once, not once per update", async () => {
  assert.ok(
    SOURCE.length > 4 * MIN_INCREMENTAL_CHARS,
    "the fixture must leave room to stream well past the incremental threshold",
  );

  const plugin = createCodePlugin({ themes: THEMES });
  const start = MIN_INCREMENTAL_CHARS + 500;
  const step = Math.ceil((SOURCE.length - start) / 15);

  await highlightOnce(plugin, SOURCE.slice(0, start));
  const streamed = SOURCE.length - start;
  tokenized.characters = 0;
  tokenized.calls = 0;

  let updates = 0;
  let last: HighlightResult | null = null;
  for (let length = start + step; length <= SOURCE.length; length += step) {
    await settle();
    last = await highlightOnce(
      plugin,
      SOURCE.slice(0, Math.min(length, SOURCE.length)),
    );
    updates += 1;
  }
  await settle();
  last = await highlightOnce(plugin, SOURCE);

  assert.ok(
    updates >= 12,
    `the stream needs enough updates to tell the two apart, got ${updates}`,
  );

  assert.ok(
    tokenized.characters >= streamed,
    `only ${tokenized.characters} characters reached shiki for ${streamed} characters of new source; the updates never left the throttled approximation and this test measured nothing`,
  );

  // Incremental work lands just above `streamed`; full re-tokenization is ~10x more.
  assert.ok(
    tokenized.characters <= 3 * streamed,
    `${tokenized.characters} characters were tokenized to stream ${streamed} new ones over ${updates} updates: the fence is being re-tokenized whole instead of incrementally`,
  );

  const reference = await createHighlighter({
    themes: THEMES,
    langs: ["typescript"],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  assert.deepEqual(
    last?.tokens,
    reference.codeToTokens(SOURCE, {
      lang: "typescript",
      themes: { light: "github-light", dark: "github-dark" },
      ...TOKENIZE_LIMITS,
    }).tokens,
  );
});

/* Shiki's tokenizeTimeLimit degrades lines on slow hosts; the wall clock must not reach output. */
test("tokenization does not degrade when the tokenizer overruns the wall clock", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const prefix = SOURCE.slice(0, MIN_INCREMENTAL_CHARS + 500);

  await highlightOnce(plugin, prefix);
  await settle();

  const realNow = Date.now;
  let result: HighlightResult;
  try {
    let elapsed = 0;
    Date.now = () => realNow() + (elapsed += 60_000);
    result = plugin.highlight({
      code: SOURCE,
      language: LANGUAGE,
      themes: THEMES,
    }) as HighlightResult;
  } finally {
    Date.now = realNow;
  }

  assert.ok(result, "the grown fence should tokenize synchronously here");
  const reference = await createHighlighter({
    themes: THEMES,
    langs: ["typescript"],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  assert.deepEqual(
    result.tokens,
    reference.codeToTokens(SOURCE, {
      lang: "typescript",
      themes: { light: "github-light", dark: "github-dark" },
      ...TOKENIZE_LIMITS,
    }).tokens,
  );
});
