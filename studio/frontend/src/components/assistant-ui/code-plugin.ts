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
// One slot per fence in flight. A message can hold several large fences, and
// Streamdown revisits all of them on every render.
const MAX_SLOTS_PER_KEY = 8;

type TokenLine = HighlightResult["tokens"][number];
type Dispatch = {
  opts: HighlightOptions;
  language: BundledLanguage;
  callback?: (result: HighlightResult) => void;
};
type Slot = {
  /** Code that produced `result`. Only ever set together with it. */
  code: string;
  result: HighlightResult | null;
  /** Code of the dispatch awaiting a callback. */
  inFlight: string | null;
  lastDispatchAt: number;
  trailing: ReturnType<typeof setTimeout> | null;
  pending: Dispatch | null;
};

// Unstyled line: no colour fields, so it renders in the default foreground
// rather than inheriting a neighbouring token's colour.
const plainLine = (text: string): TokenLine =>
  [{ content: text, offset: 0 }] as unknown as TokenLine;

export function createCodePlugin(
  options: CodePluginOptions = {},
): CodeHighlighterPlugin {
  const inner = createShikiCodePlugin(options);
  const slotsByKey = new Map<string, Slot[]>();

  const clearTrailing = (slot: Slot) => {
    if (slot.trailing !== null) clearTimeout(slot.trailing);
    slot.trailing = null;
    slot.pending = null;
  };

  const dispatch = (slot: Slot, d: Dispatch) => {
    slot.inFlight = d.opts.code;
    slot.lastDispatchAt = Date.now();
    return inner.highlight({ ...d.opts, language: d.language }, (result) => {
      // Adopt only the dispatch still in flight, and move code/result together
      // so a reuse can never slice one against the other.
      if (slot.inFlight === d.opts.code) {
        slot.code = d.opts.code;
        slot.result = result;
        slot.inFlight = null;
      }
      d.callback?.(result);
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
      let slots = slotsByKey.get(key);
      if (!slots) {
        slots = [];
        slotsByKey.set(key, slots);
      }

      // Match this fence to its own slot by longest prefix, so sibling fences
      // do not evict each other.
      let slot: Slot | null = null;
      let bestLength = -1;
      for (const candidate of slots) {
        const anchor = candidate.code || candidate.inFlight || "";
        if (!anchor || !opts.code.startsWith(anchor)) continue;
        if (anchor.length > bestLength) {
          slot = candidate;
          bestLength = anchor.length;
        }
      }
      if (!slot) {
        slot = { code: "", result: null, inFlight: null, lastDispatchAt: 0, trailing: null, pending: null };
        slots.unshift(slot);
        for (const dropped of slots.splice(MAX_SLOTS_PER_KEY)) clearTrailing(dropped);
      }

      // Finished fence re-rendered unchanged: serve it, never re-tokenize.
      if (slot.result && slot.code === opts.code) return slot.result;

      const elapsed = Date.now() - slot.lastDispatchAt;
      const grew = slot.result !== null && opts.code.length > slot.code.length;
      if (!grew || elapsed >= REFRESH_MS) {
        clearTrailing(slot);
        return dispatch(slot, { opts, language, callback });
      }

      // Always close out a reused run, so the fence cannot be left showing an
      // unstyled tail if this turns out to be its final render.
      slot.pending = { opts, language, callback };
      if (slot.trailing === null) {
        const target = slot;
        target.trailing = setTimeout(() => {
          target.trailing = null;
          const next = target.pending;
          target.pending = null;
          if (next) dispatch(target, next);
        }, Math.max(0, REFRESH_MS - elapsed));
      }

      const previous = slot.result as HighlightResult;
      // Drop the cached final line: it may have been cut mid-token.
      const keptLines = previous.tokens.slice(
        0,
        Math.max(0, slot.code.split("\n").length - 1),
      );
      const tail = opts.code.split("\n").slice(keptLines.length);
      return { ...previous, tokens: [...keptLines, ...tail.map(plainLine)] };
    },
  };
}
