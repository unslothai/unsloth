// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type { HighlightOptions, HighlightResult, ThemeInput } from "@streamdown/code";

register("./shiki-tokenization-resolver.mjs", import.meta.url);
const { createCodePlugin } = await import("../src/components/assistant-ui/code-plugin.ts");

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
const LANGUAGE = "typescript" as HighlightOptions["language"];
const options = (code: string): HighlightOptions => ({ code, language: LANGUAGE, themes: THEMES });

const highlighted = (plugin: ReturnType<typeof createCodePlugin>, code: string): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const now = plugin.highlight(options(code), resolve);
    if (now) resolve(now);
  });

const SEEDED = "const a = 1;\nexport function f() {\n  return `x${a}`;\n}";

test("the exact path never answers with the throttled approximation of a fence that extends another", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const reference = createCodePlugin({ themes: THEMES });
  const base = Array.from({ length: 120 }, (_, i) => `const value${i} = ${i} * 2;`).join("\n") + "\n";
  assert.ok(base.length > 2000);
  const longer = `${base}export function tail() {\n  return value1 + value2;\n}\n`;
  await highlighted(plugin, base);
  const approximate = plugin.highlight(options(longer), () => {});
  const exact = plugin.highlightExact(options(longer));
  const expected = await highlighted(reference, longer);
  assert.ok(approximate && exact);
  assert.notDeepEqual(approximate.tokens, expected.tokens, "the ordinary path still throttles a large growing fence");
  assert.deepEqual(exact.tokens, expected.tokens, "the exact path tokenizes the whole body");
  assert.equal(plugin.cached(options(longer)), exact, "and caches it as the fence's exact result");
});

test("a seeded result is served for its exact source only, never as a prefix for a longer one", async () => {
  const fresh = createCodePlugin({ themes: THEMES });
  const seeded = createCodePlugin({ themes: THEMES });
  const result = await highlighted(fresh, SEEDED);
  const extended = `${SEEDED}\nconst more = a + 1;\n`;
  const expected = await highlighted(fresh, extended);
  seeded.seed(options(SEEDED), result);
  assert.equal(seeded.cached(options(SEEDED)), result, "an exact hit hands back the seeded object");
  assert.equal(seeded.highlight(options(SEEDED), () => {}), result, "and the ordinary highlight path too");
  const continued = await highlighted(seeded, extended);
  assert.deepEqual(continued.tokens, expected.tokens, "a longer body is tokenized afresh, not continued from the seeded lines");
  assert.equal(continued.tokens.length, expected.tokens.length);
});
