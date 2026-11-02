// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The highlighter registers only the theme names Studio passes; shiki's full map emitted one
 * unreachable lazy chunk per bundled theme. A newly named theme must be added to the map.
 */

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import { bundledThemesInfo } from "shiki/themes";

import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";

import { readSrc } from "./helpers/kit.ts";

const PLUGIN_SOURCE = readSrc("components/assistant-ui/code-plugin.ts");
const SRC_ROOT = fileURLToPath(new URL("../src/", import.meta.url));

const sourceFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return /\.(ts|tsx)$/.test(entry.name) ? [path] : [];
  });

const registeredNames = (): Set<string> => {
  const map = /const NAMED_THEMES = \{([\s\S]*?)\n\};/.exec(PLUGIN_SOURCE);
  assert.ok(map, "code-plugin must declare its NAMED_THEMES map");
  return new Set(
    Array.from(map[1].matchAll(/^\s*"([a-z0-9-]+)":/gm), (m) => m[1]),
  );
};

test("the highlighter does not register shiki's full themes map", () => {
  assert.ok(
    !/^import\s+(?!type\b)[^;]*?\bfrom\s+"shiki\/themes";/m.test(PLUGIN_SOURCE),
    "bundledThemes emits a lazy chunk per bundled theme; list only the names Studio passes",
  );
  assert.ok(PLUGIN_SOURCE.includes("themes: NAMED_THEMES,"));
});

test("every bundled theme name a highlighting call site passes is registered", () => {
  const ids = bundledThemesInfo.map((theme) => theme.id);
  const literal = new RegExp(`["'\`](${ids.join("|")})["'\`]`, "g");
  const registered = registeredNames();
  const used = new Set<string>();
  for (const file of sourceFiles(SRC_ROOT)) {
    const text = readFileSync(file, "utf8");
    if (!/shikiTheme|createCodePlugin|ThemeInput/.test(text)) continue;
    for (const match of text.matchAll(literal)) used.add(match[1]);
  }
  // Streamdown's default shikiTheme, used by every Streamdown that passes none.
  used.add("github-light");
  used.add("github-dark");
  assert.deepEqual(
    [...used].filter((name) => !registered.has(name)),
    [],
  );
  assert.deepEqual([...registered].sort(), [...used].sort());
});

const highlightOnce = (
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const immediate = plugin.highlight(options, resolve);
    if (immediate) resolve(immediate);
  });

test("the registered names still highlight", async () => {
  const themes: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
  const result = await highlightOnce(createCodePlugin({ themes }), {
    code: "def f(x):\n    return x + 1\n",
    language: "python" as HighlightOptions["language"],
    themes,
  });
  const colours = new Set(
    result.tokens
      .flat()
      .flatMap((token) => Object.values(token.htmlStyle ?? {})),
  );
  assert.ok(colours.size > 2, "expected real token colours for both themes");
});
