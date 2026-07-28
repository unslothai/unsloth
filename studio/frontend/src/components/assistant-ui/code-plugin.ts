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
type Dispatch = {
  opts: HighlightOptions;
  language: BundledLanguage;
  callback?: (result: HighlightResult) => void;
};
type CacheEntry = {
  /** Code the cached tokens were produced from. */
  code: string;
  result: HighlightResult | null;
  /** When inner.highlight() was last dispatched for this key. */
  dispatchedAt: number;
  /** Trailing re-tokenize that closes out a run of reused results. */
  trailing: ReturnType<typeof setTimeout> | null;
  latest: Dispatch | null;
};

// Unstyled line: no colour fields, so it renders in the default foreground
// rather than inheriting a neighbouring token's colour.
const plainLine = (text: string): TokenLine =>
  [{ content: text, offset: 0 }] as unknown as TokenLine;

export function createCodePlugin(
  options: CodePluginOptions = {},
): CodeHighlighterPlugin {
  const inner = createShikiCodePlugin(options);
  const cache = new Map<string, CacheEntry>();

  const dispatch = (entry: CacheEntry, { opts, language, callback }: Dispatch) => {
    entry.code = opts.code;
    entry.dispatchedAt = Date.now();
    return inner.highlight({ ...opts, language }, (result) => {
      // Ignore stale results from a superseded dispatch.
      if (entry.code === opts.code) entry.result = result;
      callback?.(result);
    });
  };

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
      let entry = cache.get(key);
      if (!entry) {
        entry = { code: "", result: null, dispatchedAt: 0, trailing: null, latest: null };
        cache.set(key, entry);
      }
      const now = Date.now();
      const elapsed = now - entry.dispatchedAt;
      // Kept lines are byte-identical to the cached run (strict prefix), so
      // their tokens are still correct; only the tail is unstyled.
      const canReuse =
        entry.result != null &&
        opts.code.length > entry.code.length &&
        opts.code.startsWith(entry.code) &&
        elapsed < REFRESH_MS;

      if (!canReuse) {
        return dispatch(entry, { opts, language, callback });
      }

      // Always close out a reused run, so the block cannot be left showing an
      // unstyled tail if this turns out to be the final render.
      entry.latest = { opts, language, callback };
      if (entry.trailing === null) {
        const wait = Math.max(0, REFRESH_MS - elapsed);
        entry.trailing = setTimeout(() => {
          const e = entry as CacheEntry;
          e.trailing = null;
          const pending = e.latest;
          e.latest = null;
          if (pending) dispatch(e, pending);
        }, wait);
      }

      const previous = entry.result as HighlightResult;
      // Drop the cached final line: it may have been cut mid-token.
      const keptLines = previous.tokens.slice(
        0,
        Math.max(0, entry.code.split("\n").length - 1),
      );
      const tail = opts.code.split("\n").slice(keptLines.length);
      return { ...previous, tokens: [...keptLines, ...tail.map(plainLine)] };
    },
  };
}
