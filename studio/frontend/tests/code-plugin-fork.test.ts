// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type { HighlightOptions, HighlightResult, ThemeInput } from "@streamdown/code";

register("./shiki-tokenization-resolver.mjs", import.meta.url);
const { createCodePlugin } = await import("../src/components/assistant-ui/code-plugin.ts");
const { tokenized } = await import("./shiki-tokenization-counter.mts");

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
const LANGUAGE = "typescript" as HighlightOptions["language"];
const options = (code: string): HighlightOptions => ({ code, language: LANGUAGE, themes: THEMES });
type Plugin = ReturnType<typeof createCodePlugin>;

const exactly = (plugin: Plugin, code: string): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const now = plugin.highlightExact(options(code), resolve);
    if (now) resolve(now);
  });

const lines = (name: string, count: number) =>
  Array.from({ length: count }, (_, i) => `const ${name}${i} = ${i} * 2;`).join("\n") + "\n";

const a = lines("a", 120);
const b = a + lines("b", 6);
const c = b + lines("c", 6);

test("settled fences that extend one another each get exact tokens once, and the shorter keeps its own entry", async () => {
  const reference = createCodePlugin({ themes: THEMES });
  const expected = [await exactly(reference, a), await exactly(reference, b), await exactly(reference, c)];
  const plugin = createCodePlugin({ themes: THEMES });
  await exactly(plugin, "const warm = 1;\n");
  assert.ok(a.length > 2000, "the family is past the size where the throttled path would answer");
  tokenized.characters = 0;
  const warmPass = [a, b, c].map((code) => plugin.highlightExact(options(code)));
  const effectPass = [a, b, c].map((code) => plugin.highlightExact(options(code)));
  for (let i = 0; i < 3; i += 1) {
    assert.ok(warmPass[i], `fence ${i} answered synchronously`);
    assert.deepEqual(warmPass[i]!.tokens, expected[i].tokens, `fence ${i} has every line coloured`);
    assert.equal(effectPass[i], warmPass[i], `fence ${i} is served from its own entry on the second pass`);
  }
  assert.ok(
    tokenized.characters <= c.length + 64,
    `${tokenized.characters} characters were tokenized for a family whose longest fence is ${c.length}; a sibling must reuse the committed lines instead of re-tokenizing the fence it extends`,
  );
});

test("an exact request leaves a streaming sibling's queued refresh in place", async () => {
  const reference = createCodePlugin({ themes: THEMES });
  const expected = await exactly(reference, b);
  const plugin = createCodePlugin({ themes: THEMES });
  await exactly(plugin, a);
  // Refresh slices can land partial on a slow runner; the same callback fires again until complete.
  const refined = new Promise<HighlightResult>((resolve) => {
    const approximate = plugin.highlight(options(b), (result) => {
      if (JSON.stringify(result.tokens) === JSON.stringify(expected.tokens)) resolve(result);
    });
    assert.ok(approximate, "the growing fence is answered with the throttled approximation");
    assert.notDeepEqual(approximate!.tokens, expected.tokens);
  });
  const sibling = plugin.highlightExact(options(a + lines("z", 3)));
  assert.ok(sibling);
  const late = await Promise.race([
    refined,
    new Promise<null>((resolve) => setTimeout(() => resolve(null), 2000)),
  ]);
  assert.ok(late, "the streaming fence still receives its refresh");
  assert.deepEqual(late!.tokens, expected.tokens);
});
