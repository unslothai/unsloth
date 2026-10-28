// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The desktop CSP forbids WASM, so shiki must use the JS regex engine; also exercises the
 * character-budget eviction branch, unreachable from other fixtures.
 */

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
  TOKENIZE_LIMITS,
} from "../src/components/assistant-ui/code-plugin.ts";

import { readSrc } from "./helpers/kit.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

const PLUGIN_SOURCE = readSrc("components/assistant-ui/code-plugin.ts");

const highlightOnce = (
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const immediate = plugin.highlight(options, resolve);
    if (immediate) resolve(immediate);
  });

test("the highlighter uses the JavaScript regex engine, not the WASM one", () => {
  // assert.match prints the whole subject, burying the message, so assert a boolean.
  assert.ok(
    /createJavaScriptRegexEngine/.test(PLUGIN_SOURCE),
    "code-plugin must build its highlighter with shiki's JavaScript regex engine",
  );
  assert.ok(
    !/shiki\/wasm|loadWasm|createOnigurumaEngine/.test(PLUGIN_SOURCE),
    "shiki's Oniguruma engine is WebAssembly, which the desktop app's CSP " +
      "(default-src 'self', no wasm-unsafe-eval) blocks; the packaged app ships " +
      "this same bundle and no CI job would catch it",
  );
});

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

const stillCached = async (
  plugin: ReturnType<typeof createCodePlugin>,
  previous: HighlightResult,
  code: string,
  language: HighlightOptions["language"],
): Promise<boolean> =>
  (await highlightOnce(plugin, { code, language, themes: THEMES })) === previous;

test("the character budget evicts even while the fence count is under its limit", async () => {
  // ~600 KB over 100 fences exceeds MAX_CACHED_CHARACTERS but not MAX_FENCES.
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
  // Waits out REFRESH_MS, and uses HTML with embedded script since its resumed state matters.
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
      ...TOKENIZE_LIMITS,
    }).tokens;

  const settle = () => new Promise((r) => setTimeout(r, 260));
  const plugin = createCodePlugin({ themes: THEMES });
  const half = Math.floor(source.length / 2);

  for (let length = 2_100; length <= half; length += 700) {
    await settle();
    await highlightOnce(plugin, {
      code: source.slice(0, length),
      language: "html",
      themes: THEMES,
    });
  }

  for (let id = 0; id < 100; id += 1) {
    await highlightOnce(plugin, {
      code: sourceOfSize(id, 6_000),
      language: "html",
      themes: THEMES,
    });
  }

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

/** codeKey is lossy, so a hit must be confirmed by exact code before serving. */
test("two fences that share a compact cache key are not served each other's tokens", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
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
