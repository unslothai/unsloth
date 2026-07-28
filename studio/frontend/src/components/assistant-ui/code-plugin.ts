// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  createCodePlugin as createShikiCodePlugin,
  type CodeHighlighterPlugin,
  type CodePluginOptions,
  type HighlightOptions,
  type HighlightResult,
} from "@streamdown/code";
import type { BundledLanguage } from "shiki";

// Common fence tags shiki doesn't expose as aliases.
// Keys: lower-cased input; values: canonical shiki language ids.
const LANGUAGE_ALIAS_OVERRIDES: Record<string, BundledLanguage> = {
  objectivec: "objective-c",
  "obj-c": "objective-c",
  objectivecpp: "objective-cpp",
  "objective-cplusplus": "objective-cpp",
  objcpp: "objective-cpp",
  "c++": "cpp",
  cplusplus: "cpp",
  "c#": "csharp",
  cs: "csharp",
  "f#": "fsharp",
  "c-sharp": "csharp",
  "f-sharp": "fsharp",
  golang: "go",
  rs: "rust",
  rb: "ruby",
  py: "python",
  sh: "shellscript",
  bash: "shellscript",
  zsh: "shellscript",
  shell: "shellscript",
  yml: "yaml",
  ts: "typescript",
  js: "javascript",
  kt: "kotlin",
  rsx: "rust",
  "vue-html": "vue",
};

const normalizeLanguage = (language: string): BundledLanguage => {
  const key = language.trim().toLowerCase();
  const override = LANGUAGE_ALIAS_OVERRIDES[key];
  return (override ?? (key as BundledLanguage));
};

// A streaming fence re-enters highlight() every frame with the whole block, so
// Shiki re-tokenizes it from scratch ~60x/sec: O(length) per frame. Past
// MIN_INCREMENTAL_CHARS, reuse the cached tokens and append the new tail
// unstyled, re-tokenizing in full at most every REFRESH_MS.
const MIN_INCREMENTAL_CHARS = 2000;
const REFRESH_MS = 250;

type TokenLine = HighlightResult["tokens"][number];
type CacheEntry = {
  /** Code the cached tokens were produced from. */
  code: string;
  result: HighlightResult | null;
  /** When inner.highlight() was last dispatched for this key. */
  dispatchedAt: number;
};

export function createCodePlugin(
  options: CodePluginOptions = {},
): CodeHighlighterPlugin {
  const inner = createShikiCodePlugin(options);
  const cache = new Map<string, CacheEntry>();

  // Copies an existing token so dual-theme `variants` stay well-formed.
  const plainLine = (text: string, template?: TokenLine[number]): TokenLine =>
    [{ ...(template ?? {}), content: text, offset: 0 }] as TokenLine;

  return {
    ...inner,
    supportsLanguage: (language) =>
      inner.supportsLanguage(normalizeLanguage(language)),
    highlight: (
      opts: HighlightOptions,
      callback?: (result: HighlightResult) => void,
    ) => {
      const language = normalizeLanguage(opts.language);
      if (opts.code.length < MIN_INCREMENTAL_CHARS) {
        return inner.highlight({ ...opts, language }, callback);
      }

      const key = `${language} ${JSON.stringify(opts.themes)}`;
      const cached = cache.get(key);
      const now = Date.now();
      const grewFrom =
        cached?.result != null &&
        opts.code.length > cached.code.length &&
        opts.code.startsWith(cached.code)
          ? cached
          : null;

      // Shiki returns null synchronously and resolves via callback, so
      // throttling the dispatch is what removes the per-frame work.
      if (grewFrom && now - grewFrom.dispatchedAt < REFRESH_MS) {
        const previous = grewFrom.result as HighlightResult;
        // Drop the cached final line: it may have been cut mid-token.
        const keptLines = previous.tokens.slice(
          0,
          Math.max(0, grewFrom.code.split("\n").length - 1),
        );
        const template = previous.tokens[0]?.[0];
        const tail = opts.code.split("\n").slice(keptLines.length);
        return {
          ...previous,
          tokens: [
            ...keptLines,
            ...tail.map((line) => plainLine(line, template)),
          ],
        };
      }

      cache.set(key, {
        code: opts.code,
        result: cached?.result ?? null,
        dispatchedAt: now,
      });
      return inner.highlight({ ...opts, language }, (result) => {
        const entry = cache.get(key);
        // Ignore stale results from a superseded dispatch.
        if (entry && entry.code === opts.code) {
          entry.result = result;
        }
        callback?.(result);
      });
    },
  };
}
