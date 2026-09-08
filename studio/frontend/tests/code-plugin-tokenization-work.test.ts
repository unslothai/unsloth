// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The one claim `code-plugin.ts` exists to make: a streaming fence is tokenized
 * once, not once per update.
 *
 * Every other test here compares tokens, and tokens are identical either way --
 * re-tokenizing the whole fence on every frame produces exactly the same
 * output, just O(updates x length) work instead of O(length). So dropping the
 * committed lines and the grammar state, and calling shiki with the whole block
 * again, would pass the entire rest of the suite. This is the only test that
 * would notice.
 *
 * Two things it has to get right, or it measures nothing:
 *
 *  - It must wait out REFRESH_MS between updates. Inside that window a grown
 *    fence past MIN_INCREMENTAL_CHARS returns `approximateResult` and cancels
 *    the refresh the previous frame queued, so no tokenizer runs at all and the
 *    character count stays near zero however the tokenizer is written. The
 *    lower bound below fails outright if that happens.
 *  - It must count what the plugin's own highlighter sees. The plugin builds
 *    that highlighter from a static `shiki` import and an ES module namespace
 *    cannot be patched, so the resolver hook redirects that single import to a
 *    counting re-export. The reference highlighter below is imported from the
 *    real `shiki` and is not counted.
 *
 * The bound is on characters, not on wall-clock time, so it is deterministic
 * under any CI load; a slow machine only sleeps longer, which keeps every
 * update on the path being measured.
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
// Longer than REFRESH_MS, so every update takes the tokenizing path.
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

  // The first frame loads the grammar and tokenizes the prefix whole; only what
  // the fence costs from here on is the thing under test.
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

  // Lower bound: the fence really was tokenized. Without it a plugin that never
  // reached shiki at all -- every frame throttled, or the whole test sitting on
  // the approximation path -- would satisfy the upper bound trivially.
  assert.ok(
    tokenized.characters >= streamed,
    `only ${tokenized.characters} characters reached shiki for ${streamed} characters of new source; the updates never left the throttled approximation and this test measured nothing`,
  );

  // Upper bound. Incremental work is the new source once, plus the unterminated
  // tail of each frame, so it lands just above `streamed`. Re-tokenizing the
  // whole fence every frame is ~sum(length) over the updates, an order of
  // magnitude more; 3x separates them with room for either to drift.
  assert.ok(
    tokenized.characters <= 3 * streamed,
    `${tokenized.characters} characters were tokenized to stream ${streamed} new ones over ${updates} updates: the fence is being re-tokenized whole instead of incrementally`,
  );

  // And it is still correct: the count above must not be bought with wrong tokens.
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

/*
 * WHAT MADE THE ASSERTION ABOVE FAIL ON WINDOWS ONE RUN IN TWENTY.
 *
 * Shiki abandons a line once `tokenizeTimeLimit` of wall clock has gone by and emits the rest of it
 * as one uncoloured token. Dual themes are two passes with two budgets and only the first compiles
 * the grammar's regexes, so a slow enough host returns the light theme plain and the dark theme
 * correct -- and the plugin then commits that line and never tokenizes it again. Neither side of the
 * comparison was safe: CI produced diffs with the plugin degraded, diffs with the reference
 * degraded, and diffs with both on different lines.
 *
 * Racing `Date.now` reproduces an overrun on any machine, without waiting for one: every elapsed
 * check clears any finite limit, and `tokenizeTimeLimit: 0` skips the check entirely. Which tokens
 * a real overrun loses depends on where in the line it lands, so this pins the invariant rather
 * than one signature -- the wall clock must not reach the output at all. The plugin's own throttle
 * reads `performance.now`, so it is unaffected.
 */
test("tokenization does not degrade when the tokenizer overruns the wall clock", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const prefix = SOURCE.slice(0, MIN_INCREMENTAL_CHARS + 500);

  await highlightOnce(plugin, prefix);
  // Leave the throttle window so the grown fence takes the tokenizing path.
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
