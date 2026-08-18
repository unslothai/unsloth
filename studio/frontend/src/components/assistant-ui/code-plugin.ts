// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  createCodePlugin as createShikiCodePlugin,
  type CodeHighlighterPlugin,
  type CodePluginOptions,
  type HighlightOptions,
  type HighlightResult,
  type ThemeInput,
} from "@streamdown/code";
import {
  bundledLanguages,
  bundledLanguagesInfo,
  createHighlighter,
  type BundledLanguage,
  type Highlighter,
  type SpecialLanguage,
} from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import { createTokenCache } from "./code-token-cache.ts";

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

// @streamdown/code 1.1.1 memoises every tokenisation in a module-level Map whose key embeds the
// code's LENGTH plus its first and last 100 characters, and it has no eviction path at all: no
// size cap, no clear, no unmount hook. A STREAMED fence is handed to it once per refresh window
// with a longer prefix each time, so a single reply mints one full dual-theme tokenisation per
// window and every one of them is kept for the life of the tab. Measured on a 32 KB python fence
// streamed over about five seconds: +10.8 MB of retained V8 heap per fence, still there after the
// reply is unmounted and the heap is force-collected.
//
// Studio therefore drives Shiki itself, with the same engine, the same highlighter arguments and
// the same asynchronous contract as the package, and holds the result in a cache that is bounded
// and that DROPS THE PREFIXES of a fence as the fence grows. Same tokens on screen; a streaming
// fence occupies one entry instead of one per window.

// Aliases Shiki does publish, resolved the way the package resolves them, so a fence tag that
// worked before still lands on the same grammar.
const SHIKI_ALIASES: Record<string, string> = Object.fromEntries(
  bundledLanguagesInfo.flatMap((entry) =>
    (entry.aliases ?? []).map((alias) => [alias, entry.id]),
  ),
);
const BUNDLED_LANGUAGE_IDS = new Set(Object.keys(bundledLanguages));

// Unknown tags fall back to "text", which is what the package does; without it `createHighlighter`
// rejects and the fence is left permanently unstyled.
type ResolvedLanguage = BundledLanguage | SpecialLanguage;

const canonicalLanguage = (language: string): ResolvedLanguage => {
  const normalized = normalizeLanguage(language);
  const resolved = SHIKI_ALIASES[normalized] ?? normalized;
  return BUNDLED_LANGUAGE_IDS.has(resolved)
    ? (resolved as BundledLanguage)
    : "text";
};

const themeName = (theme: ThemeInput): string =>
  typeof theme === "string" ? theme : (theme.name ?? "custom");

// The JavaScript regex engine, not Oniguruma. Same choice the package makes, and it is not
// optional here: studio/src-tauri/tauri.conf.json ships a CSP without `wasm-unsafe-eval`, so the
// WASM engine cannot instantiate inside the packaged desktop app.
const regexEngine = createJavaScriptRegexEngine({ forgiving: true });

// One highlighter per language and theme pair, exactly as the package keys them. Bounded by the
// number of languages a session actually shows.
const highlighters = new Map<string, Promise<Highlighter>>();
const highlighterFor = (
  language: ResolvedLanguage,
  themes: [ThemeInput, ThemeInput],
): Promise<Highlighter> => {
  const key = `${language}-${themeName(themes[0])}-${themeName(themes[1])}`;
  const existing = highlighters.get(key);
  if (existing) return existing;
  const created = createHighlighter({
    themes,
    langs: [language],
    engine: regexEngine,
  });
  highlighters.set(key, created);
  return created;
};

// Budget for the token cache. Tokens cost roughly 17x their source in retained heap here (a
// 32 KB fence tokenises to about 0.55 MB), so 512,000 characters of source caps this at single
// digit megabytes while still holding far more finished fences than a thread shows at once.
const MAX_CACHED_CHARS = 512_000;
const MAX_CACHED_ENTRIES = 64;

const tokenCache = createTokenCache<HighlightResult>({
  maxChars: MAX_CACHED_CHARS,
  maxEntries: MAX_CACHED_ENTRIES,
});
const pendingTokenisations = new Map<
  string,
  Set<(result: HighlightResult) => void>
>();

const groupKey = (
  language: ResolvedLanguage,
  themes: [ThemeInput, ThemeInput],
): string => `${language} ${themeName(themes[0])} ${themeName(themes[1])}`;

// Same contract as the package's `highlight`: a cached result synchronously, or null plus a
// callback once Shiki has answered.
const tokenize = (
  opts: HighlightOptions,
  callback?: (result: HighlightResult) => void,
): HighlightResult | null => {
  const language = canonicalLanguage(opts.language);
  const group = groupKey(language, opts.themes);
  const key = `${group} ${opts.code}`;
  const hit = tokenCache.get(group, opts.code);
  if (hit) return hit;

  const waiting = pendingTokenisations.get(key);
  if (waiting) {
    if (callback) waiting.add(callback);
    return null;
  }
  const waiters = new Set<(result: HighlightResult) => void>();
  if (callback) waiters.add(callback);
  pendingTokenisations.set(key, waiters);

  highlighterFor(language, opts.themes)
    .then((highlighter) => {
      const loaded = highlighter.getLoadedLanguages().includes(language)
        ? language
        : "text";
      const result = highlighter.codeToTokens(opts.code, {
        lang: loaded,
        themes: {
          light: themeName(opts.themes[0]),
          dark: themeName(opts.themes[1]),
        },
      });
      tokenCache.set(group, opts.code, result);
      pendingTokenisations.delete(key);
      for (const waiter of waiters) waiter(result);
    })
    .catch((error: unknown) => {
      console.error("[Studio Code] Failed to highlight code:", error);
      pendingTokenisations.delete(key);
    });
  return null;
};

// Test seam: the retention probe reads this to assert the cache stays bounded while a fence
// streams. Not used by the app.
export const __tokenCacheStats = () => ({
  ...tokenCache.stats(),
  pending: pendingTokenisations.size,
});

// A streaming fence re-enters highlight() every frame with the whole block, so
// Shiki re-tokenizes it in full ~60x/sec. Past MIN_INCREMENTAL_CHARS, reuse the
// cached tokens with an unstyled tail, re-tokenizing at most every REFRESH_MS.
export const MIN_INCREMENTAL_CHARS = 2000;
const REFRESH_MS = 250;
// Wall-clock Date.now() can step backwards (NTP, sleep resume) and make
// `elapsed` negative; the throttle only needs elapsed time, so stay monotonic.
const monotonicNow = (): number =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

// One slot per fence: a message can hold several large fences, and Streamdown
// revisits all of them on every render.
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

// No colour fields, so it renders in the default foreground instead of
// inheriting a neighbouring token's colour.
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

  const adopt = (slot: Slot, code: string, result: HighlightResult) => {
    // Write code and result together so a reuse cannot slice one against the other.
    slot.code = code;
    slot.result = result;
    slot.inFlight = null;
  };

  const dispatch = (slot: Slot, d: Dispatch) => {
    slot.inFlight = d.opts.code;
    slot.lastDispatchAt = monotonicNow();
    const immediate = tokenize({ ...d.opts, language: d.language }, (result) => {
      if (slot.inFlight === d.opts.code) {
        adopt(slot, d.opts.code, result);
      }
      d.callback?.(result);
    });
    // A cache hit is answered synchronously and never invokes the callback, so
    // adopt here too or the slot keeps older tokens.
    if (immediate) {
      adopt(slot, d.opts.code, immediate);
    }
    return immediate;
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
        return tokenize({ ...opts, language }, callback);
      }

      const key = `${language} ${JSON.stringify(opts.themes)}`;
      let slots = slotsByKey.get(key);
      if (!slots) {
        slots = [];
        slotsByKey.set(key, slots);
      }

      // Longest-prefix match, so sibling fences do not evict each other.
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

      const elapsed = monotonicNow() - slot.lastDispatchAt;
      const grew = slot.result !== null && opts.code.length > slot.code.length;
      if (!grew || elapsed >= REFRESH_MS) {
        clearTrailing(slot);
        return dispatch(slot, { opts, language, callback });
      }

      // Close out a reused run, so a final render is never left unstyled.
      slot.pending = { opts, language, callback };
      if (slot.trailing === null) {
        const target = slot;
        target.trailing = setTimeout(() => {
          target.trailing = null;
          const next = target.pending;
          target.pending = null;
          if (!next) return;
          const immediate = dispatch(target, next);
          // Nothing consumes this return value, so hand a synchronous cache
          // hit to the callback or the fence keeps its unstyled tail.
          if (immediate) next.callback?.(immediate);
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
