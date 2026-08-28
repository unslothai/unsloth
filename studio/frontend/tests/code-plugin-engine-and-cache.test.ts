// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Two guards on `code-plugin.ts` that nothing else covers.
 *
 * 1. The regex engine. The desktop app's CSP is `default-src 'self'` with no
 *    `wasm-unsafe-eval` (studio/src-tauri/tauri.conf.json), and the packaged
 *    frontend is the same bundle the browser gets. Shiki's default Oniguruma
 *    engine is WebAssembly, so swapping to it would work in every browser and
 *    every test here, and be blocked at runtime in the packaged desktop app.
 *    No CI job builds the desktop bundle and drives its webview, so this is
 *    the only thing standing between that change and a shipped regression.
 *
 * 2. The cache budget. `MAX_CACHED_CHARACTERS` is 512,000 and the largest
 *    fixture in the sibling files is ~36 KB, so the character-budget branch,
 *    its "never evict the only fence" carve-out, and the running character
 *    count were all unreachable. A miscount there degrades silently: either
 *    the cache stops evicting and grows without bound, or it evicts fences it
 *    should have kept and quietly reverts to full re-tokenization.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  TOKENIZE_LIMITS,
} from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

const PLUGIN_SOURCE = readFileSync(
  new URL("../src/components/assistant-ui/code-plugin.ts", import.meta.url),
  "utf8",
);

const highlightOnce = (
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const immediate = plugin.highlight(options, resolve);
    if (immediate) resolve(immediate);
  });

// ── 1. the engine the desktop CSP allows ───────────────────────────────

test("the highlighter uses the JavaScript regex engine, not the WASM one", () => {
  // Assert on a boolean, not on PLUGIN_SOURCE: assert.match prints the whole
  // subject, so matching the file directly buries the message under ~10 KB.
  assert.ok(
    /createJavaScriptRegexEngine/.test(PLUGIN_SOURCE),
    "code-plugin must build its highlighter with shiki's JavaScript regex engine",
  );
  assert.ok(
    !/shiki\/wasm|loadWasm|createOnigurumaEngine/.test(PLUGIN_SOURCE),
    // This is a source-text assertion, not a behavioural one: it pins the name,
    // so an alias would trip it and a same-named wrapper would slip past. That
    // is the accepted cost of covering a constraint no runtime here can check,
    // since nothing in CI drives the packaged webview.
    "shiki's Oniguruma engine is WebAssembly, which the desktop app's CSP " +
      "(default-src 'self', no wasm-unsafe-eval) blocks; the packaged app ships " +
      "this same bundle and no CI job would catch it",
  );
});

// ── 2. the character budget ────────────────────────────────────────────

/** Distinct, cheap-to-tokenize source of roughly `chars` characters. */
const sourceOfSize = (id: number, chars: number): string => {
  const lines: string[] = [];
  let length = 0;
  for (let i = 0; length < chars; i += 1) {
    const line = `const value_${id}_${i} = ${i};`;
    lines.push(line);
    length += line.length + 1;
  }
  return `${lines.join("\n")}\n`;
};

/**
 * Whether the plugin still holds the entry that produced `previous`.
 *
 * Re-requesting is not enough on its own: a miss re-tokenizes and re-caches, so
 * asking twice always agrees with itself. The only signal is whether the entry
 * from before is still the one being served, so compare against the object the
 * cache handed out earlier.
 */
const stillCached = async (
  plugin: ReturnType<typeof createCodePlugin>,
  previous: HighlightResult,
  code: string,
  language: HighlightOptions["language"],
): Promise<boolean> =>
  (await highlightOnce(plugin, { code, language, themes: THEMES })) === previous;

test("the character budget evicts even while the fence count is under its limit", async () => {
  // 100 fences of ~6 KB is 600 KB, past MAX_CACHED_CHARACTERS, while staying
  // well under MAX_FENCES. Only the character branch can evict here, so this
  // fails outright if that branch is unreachable or miscounted.
  const plugin = createCodePlugin({ themes: THEMES });
  const first = sourceOfSize(0, 6_000);
  const held = await highlightOnce(plugin, {
    code: first,
    language: "typescript",
    themes: THEMES,
  });
  assert.equal(
    await stillCached(plugin, held, first, "typescript"),
    true,
    "the fence was not cached to begin with, so the check below proves nothing",
  );

  for (let id = 1; id < 100; id += 1) {
    await highlightOnce(plugin, {
      code: sourceOfSize(id, 6_000),
      language: "typescript",
      themes: THEMES,
    });
  }

  assert.equal(
    await stillCached(plugin, held, first, "typescript"),
    false,
    "600 KB of fences did not evict the oldest; the character budget is not being enforced",
  );
});

test("a fence larger than the whole budget is kept rather than evicting itself", async () => {
  // The carve-out at the eviction loop: stop when one fence is left, so a
  // single oversized fence is retained. Without it the only fence in the cache
  // would be dropped the moment it exceeded the budget, and every refresh of a
  // long block would re-tokenize from zero, which is the case this PR exists
  // to remove.
  const plugin = createCodePlugin({ themes: THEMES });
  const huge = sourceOfSize(1, 600_000);
  assert.ok(huge.length > 512_000, "fixture must exceed MAX_CACHED_CHARACTERS");

  const held = await highlightOnce(plugin, {
    code: huge,
    language: "typescript",
    themes: THEMES,
  });
  assert.equal(
    await stillCached(plugin, held, huge, "typescript"),
    true,
    "the only fence in the cache was evicted for being over budget",
  );
});

test("a fence evicted mid-stream still tokenizes correctly when it resumes", async () => {
  // Eviction drops the committed lines and the grammar state, so the fence
  // restarts from offset 0 and rebuilds incremental state as it keeps growing.
  // That is a performance loss by design; it must not be a correctness one.
  //
  // Two things this test needs that are easy to get wrong, both found by
  // measurement rather than assumed:
  //
  //  - It must wait out REFRESH_MS between updates. Past MIN_INCREMENTAL_CHARS,
  //    two updates inside that window return the throttled approximation, which
  //    renders the uncommitted tail plain and never reads the grammar state.
  //    Without the wait, 22 of 23 comparisons took that path and the test could
  //    not fail however the resume was broken.
  //  - The fixture must be one whose tokens actually change when the resumed
  //    state is dropped. A python triple-quoted string is not: dropping the
  //    state left its tokens byte-identical. HTML with an embedded script is,
  //    because the grammar is several levels deep at the resume point.
  const chunk = [
    "  <section>",
    '    <div class="card" data-note="a > b">text</div>',
    "    <script>",
    "      /* a block comment",
    "         that stays open across lines */",
    "      const total = items.reduce((sum, item) => sum + item.n, 0);",
    "      console.log(`total ${total}`);",
    "    </script>",
    "    <style>",
    "      .card { color: #333; /* comment",
    "         spanning lines */ }",
    "    </style>",
    "  </section>",
  ].join("\n");
  let source = "<!doctype html>\n<html>\n<body>\n";
  for (let i = 0; i < 12; i += 1) source += `${chunk}\n`;
  source += "</body>\n</html>\n";
  assert.ok(source.length > 4_000, "fixture must clear MIN_INCREMENTAL_CHARS");

  const highlighter = await createHighlighter({
    themes: THEMES,
    langs: ["html"],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  const oracle = (code: string) =>
    highlighter.codeToTokens(code, {
      lang: "html",
      themes: { light: "github-light", dark: "github-dark" },
      // Same tokenizer limits as the plugin, so neither side can degrade.
      ...TOKENIZE_LIMITS,
    }).tokens;

  const settle = () => new Promise((r) => setTimeout(r, 260));
  const plugin = createCodePlugin({ themes: THEMES });
  const half = Math.floor(source.length / 2);

  // Grow past the incremental threshold so committed state exists to lose.
  for (let length = 2_100; length <= half; length += 700) {
    await settle();
    await highlightOnce(plugin, {
      code: source.slice(0, length),
      language: "html",
      themes: THEMES,
    });
  }

  // Evict it.
  for (let id = 0; id < 100; id += 1) {
    await highlightOnce(plugin, {
      code: sourceOfSize(id, 6_000),
      language: "html",
      themes: THEMES,
    });
  }

  // Keep streaming. Every prefix must still match the whole document.
  for (let length = half; length <= source.length; length += 700) {
    await settle();
    const code = source.slice(0, length);
    const streamed = await highlightOnce(plugin, {
      code,
      language: "html",
      themes: THEMES,
    });
    assert.deepEqual(
      streamed.tokens,
      oracle(code),
      `a fence that lost its cache mid-stream diverged at ${length} of ${source.length}`,
    );
  }

  await settle();
  const final = await highlightOnce(plugin, {
    code: source,
    language: "html",
    themes: THEMES,
  });
  assert.deepEqual(final.tokens, oracle(source));
});

// ── 3. the compact cache key is a hint, not the answer ─────────────────

/**
 * `codeKey` hashes a fence down to `key + length + first 32 + last 32` so an
 * update does not rehash the whole block. That is a lossy digest, so two
 * different fences can share one. `findFence` therefore treats a hit as a
 * candidate and confirms it with `exact.code === code` before serving it.
 *
 * Drop that confirmation and the second fence is served the first one's
 * tokens: the reader sees the wrong code, with no error anywhere. The upstream
 * @streamdown/code cache this file replaces keys on the same shape (length
 * plus the first and last 100 characters) and does not confirm, so the pair
 * below is rendered wrong by it today. Two config or import blocks that share
 * an opening and a closing but differ in the middle are the realistic shape.
 */
test("two fences that share a compact cache key are not served each other's tokens", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  // Same length, same first 32 and same last 32 characters, different middle.
  const head = "const cfg = {\n  alpha: 1,\n  beta: 2,\n";
  const tail = "\n  omega: 26,\n};\nexport default cfg;\n";
  const first = `${head}  middle: 'AAAA',\n${tail}`;
  const second = `${head}  middle: 'BBBB',\n${tail}`;

  assert.equal(first.length, second.length, "fixture must share a length");
  assert.equal(
    first.slice(0, 32),
    second.slice(0, 32),
    "fixture must share the first 32 characters codeKey samples",
  );
  assert.equal(
    first.slice(-32),
    second.slice(-32),
    "fixture must share the last 32 characters codeKey samples",
  );
  assert.notEqual(first, second, "fixture fences must actually differ");

  const rendered = async (code: string): Promise<string> => {
    const result = await highlightOnce(plugin, {
      code,
      language: "ts",
      themes: THEMES,
    });
    return result.tokens
      .map((line) => line.map((token) => token.content).join(""))
      .join("\n");
  };

  assert.match(await rendered(first), /AAAA/);
  const secondText = await rendered(second);
  assert.doesNotMatch(
    secondText,
    /AAAA/,
    "the second fence was served the first fence's tokens, so the reader sees code that is not in the message",
  );
  assert.match(
    secondText,
    /BBBB/,
    "the second fence did not render its own contents",
  );
});
