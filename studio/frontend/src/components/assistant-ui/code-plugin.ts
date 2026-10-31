// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CodeHighlighterPlugin,
  CodePluginOptions,
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import type { GrammarState, ThemedToken } from "shiki";
import { createBundledHighlighter } from "shiki/core";
import {
  createJavaScriptRegexEngine,
  defaultJavaScriptRegexConstructor,
} from "shiki/engine/javascript";
import {
  type BundledLanguage,
  bundledLanguages,
  bundledLanguagesInfo,
} from "shiki/langs";
import { type BundledTheme, bundledThemes } from "shiki/themes";

// shiki's own createHighlighter, minus its default Oniguruma engine: every call below passes the
// JavaScript engine, so the WASM engine never ran and only cost startup bytes.
const createHighlighter = createBundledHighlighter<
  BundledLanguage,
  BundledTheme
>({
  langs: bundledLanguages,
  themes: bundledThemes,
  engine: () => createJavaScriptRegexEngine({ forgiving: true }),
});

// Fence tags shiki does not alias: lower-cased input to canonical shiki id.
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

const SHIKI_LANGUAGE_ALIASES: Record<string, BundledLanguage> =
  Object.fromEntries(
    bundledLanguagesInfo.flatMap((info) =>
      (info.aliases ?? []).map((alias) => [alias, info.id as BundledLanguage]),
    ),
  );
const SUPPORTED_LANGUAGES = new Set(Object.keys(bundledLanguages));
const SUPPORTED_LANGUAGE_LIST = Array.from(
  SUPPORTED_LANGUAGES,
) as BundledLanguage[];
const PLAIN_TEXT = "text" as BundledLanguage;

/** Exported so per-grammar caches key on the same identity as `highlight`. */
export const normalizeLanguage = (language: string): BundledLanguage => {
  const key = language.trim().toLowerCase();
  const alias = LANGUAGE_ALIAS_OVERRIDES[key] ?? SHIKI_LANGUAGE_ALIASES[key];
  return alias ?? (key as BundledLanguage);
};

// Past this size, cache completed lines and grammar state so a streaming fence tokenizes only new text.
export const MIN_INCREMENTAL_CHARS = 2000;
const REFRESH_MS = 250;
const REFRESH_SLICE_MS = 8;
// Shiki's own tokenizer guard is a wall clock: a line that takes longer than `tokenizeTimeLimit`
// (default 500 ms) is abandoned part-way and the rest of it is emitted as one uncoloured token.
// Dual themes are two passes with two separate budgets, and only the first pays to compile the
// grammar's regexes, so a loaded machine drops the light theme to plain and keeps the dark one
// correct -- for the same line, in the same result. Committed lines are never re-tokenized, so that
// half-plain line is then cached for the life of the fence. Bound the work by line length instead:
// same protection against a minified blob (~120 ms at this length, growing quadratically past it,
// which is VS Code's `editor.maxTokenizationLineLength` default for the same reason), but the
// output depends on the source rather than on the host.
export const TOKENIZE_LIMITS = {
  tokenizeTimeLimit: 0,
  tokenizeMaxLineLength: 20_000,
} as const;
// An unvirtualized thread mounts every fence. Bound both their count and source
// size; token data measured roughly 30 bytes per source character.
export const MAX_FENCES = 512;
export const MAX_CACHED_CHARACTERS = 512_000;

// Date.now() can step backwards (NTP, sleep resume), so use a monotonic clock.
const monotonicNow = (): number =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

type Highlighter = Awaited<ReturnType<typeof createHighlighter>>;
type TokenLine = HighlightResult["tokens"][number];
type ResultMeta = Omit<HighlightResult, "tokens">;
type ThemeNames = { light: string; dark: string };
type HighlightCallback = (result: HighlightResult) => void;
type Pending = { code: string; callbacks: Set<HighlightCallback> };

type Fence = {
  key: string;
  code: string;
  result: HighlightResult | null;
  meta: ResultMeta | null;
  lines: TokenLine[];
  committedLength: number;
  state: GrammarState | undefined;
  liveTokens: TokenLine | null;
  lastTokenizedAt: number;
  trailing: ReturnType<typeof setTimeout> | null;
  pending: Pending | null;
  seeded: boolean;
};

const plainLine = (content: string): TokenLine =>
  [{ content, offset: 0 } as unknown as ThemedToken] as TokenLine;

const themeName = (theme: ThemeInput): string =>
  typeof theme === "string" ? theme : (theme.name ?? "custom");

// Custom themes can share or lack names, so key on a short id per distinct definition.
const themeIds = new Map<string, string>();
const themeIdByInput = new WeakMap<object, string>();

const themeKey = (theme: ThemeInput): string => {
  if (typeof theme === "string") return theme;
  const known = themeIdByInput.get(theme);
  if (known !== undefined) return known;
  const definition = JSON.stringify(theme);
  let id = themeIds.get(definition);
  if (id === undefined) {
    id = `${themeName(theme)}#${themeIds.size}`;
    themeIds.set(definition, id);
  }
  themeIdByInput.set(theme, id);
  return id;
};

const stripTokens = (result: HighlightResult): ResultMeta => ({
  bg: result.bg,
  fg: result.fg,
  themeName: result.themeName,
  rootStyle: result.rootStyle,
  grammarState: result.grammarState,
});

const shiftLine = (line: TokenLine, offset: number): TokenLine =>
  offset === 0
    ? line
    : line.map((token) => ({ ...token, offset: token.offset + offset }));

// Adjacent-token coalescing measured zero (dual tokens 537013 -> merged 537013); Shiki already
// coalesces. Reproduce with scripts/coal-span-census.mjs.

// Markdown reports a closing fence as body until it recognizes it, so that line (possibly
// a partial run) is all a fence can lose.
const CLOSING_FENCE = /^ {0,3}(?:`+|~+)[ \t]*$/;

const shedsClosingRun = (shorter: string, longer: string): boolean =>
  (shorter === "" || shorter.endsWith("\n")) &&
  longer.startsWith(shorter) &&
  CLOSING_FENCE.test(longer.slice(shorter.length));

export type FenceCover = {
  uncovered: number;
  result: HighlightResult | null;
};

export type UnslothCodePlugin = CodeHighlighterPlugin & {
  cached: (opts: HighlightOptions) => HighlightResult | null;
  cover: (opts: HighlightOptions) => FenceCover;
  seed: (opts: HighlightOptions, result: HighlightResult) => void;
  highlight: (
    opts: HighlightOptions,
    callback?: (result: HighlightResult) => void,
    exact?: boolean,
  ) => HighlightResult | null;
  highlightExact: (
    opts: HighlightOptions,
    callback?: (result: HighlightResult) => void,
  ) => HighlightResult | null;
};

const fenceKeyOf = (opts: HighlightOptions): string =>
  `${normalizeLanguage(opts.language)} ${themeKey(opts.themes[0])} ${themeKey(opts.themes[1])}`;

const javaScriptCore =
  typeof navigator !== "undefined"
  && navigator.userAgent.includes("AppleWebKit/")
  && !/Chrom(?:e|ium)\//.test(navigator.userAgent);

export function createCodePlugin(
  options: CodePluginOptions = {},
): UnslothCodePlugin {
  const defaultThemes: [ThemeInput, ThemeInput] = options.themes ?? [
    "github-light",
    "github-dark",
  ];
  const engine = createJavaScriptRegexEngine(
    javaScriptCore
      ? {
          forgiving: true,
          regexConstructor: (pattern) => {
            try {
              return defaultJavaScriptRegexConstructor(pattern, {
                target: "ES2018",
                accuracy: "strict",
              });
            } catch {
              return defaultJavaScriptRegexConstructor(pattern);
            }
          },
        }
      : { forgiving: true },
  );
  const highlighters = new Map<string, { highlighter: Highlighter | null }>();
  // Most recently used first.
  const fences: Fence[] = [];
  const fencesByCode = new Map<string, Fence>();
  let cachedCharacters = 0;

  // Avoid hashing the whole source on each update; verify compact-key hits.
  const codeKey = (key: string, code: string): string =>
    `${key}\n${code.length}\n${code.slice(0, 32)}\n${code.slice(-32)}`;

  // Identical fences can share an index key.
  const dropCodeIndex = (fence: Fence): void => {
    const indexKey = codeKey(fence.key, fence.code);
    if (fencesByCode.get(indexKey) === fence) fencesByCode.delete(indexKey);
  };

  const clearTrailing = (fence: Fence): void => {
    if (fence.trailing !== null) clearTimeout(fence.trailing);
    fence.trailing = null;
    fence.pending = null;
  };

  const queuePending = (
    fence: Fence,
    code: string,
    callback?: HighlightCallback,
  ): void => {
    if (!fence.pending || fence.pending.code !== code) {
      fence.pending = { code, callbacks: new Set() };
    }
    if (callback) fence.pending.callbacks.add(callback);
  };

  const notifyPending = (
    pending: Pending,
    result: HighlightResult,
  ): void => {
    for (const callback of pending.callbacks) callback(result);
  };

  const dropFence = (fence: Fence): void => {
    const index = fences.indexOf(fence);
    if (index < 0) return;
    fences.splice(index, 1);
    cachedCharacters -= fence.code.length;
    dropCodeIndex(fence);
    clearTrailing(fence);
  };

  const evict = (): void => {
    while (
      fences.length > MAX_FENCES ||
      (cachedCharacters > MAX_CACHED_CHARACTERS && fences.length > 1)
    ) {
      // Keep pending callbacks until their grammar resolves.
      let dropIndex = fences.length - 1;
      while (dropIndex >= 0 && fences[dropIndex].pending !== null) {
        dropIndex -= 1;
      }
      if (dropIndex < 0) return;
      dropFence(fences[dropIndex]);
    }
  };

  const forkFence = (fence: Fence): Fence => {
    const fork: Fence = {
      ...fence,
      lines: [...fence.lines],
      liveTokens: null,
      trailing: null,
      pending: null,
      seeded: false,
    };
    fences.unshift(fork);
    cachedCharacters += fence.code.length;
    return fork;
  };

  const promote = (fence: Fence): Fence => {
    const index = fences.indexOf(fence);
    if (index > 0) {
      fences.splice(index, 1);
      fences.unshift(fence);
    }
    return fence;
  };

  const reachInto = (fence: Fence, code: string): number => {
    if (fence.result === null || fence.seeded) return -1;
    const anchor = fence.code;
    // Lost more than its closing delimiter: a different fence, so do not share the entry.
    const reaches =
      code.startsWith(anchor) ||
      (code.length >= fence.committedLength && shedsClosingRun(code, anchor));
    return anchor && reaches ? Math.min(anchor.length, code.length) : -1;
  };

  /** The fence whose cached code reaches furthest into `code`. */
  const findFence = (key: string, code: string): Fence => {
    const exact = fencesByCode.get(codeKey(key, code));
    if (exact && exact.code === code) return promote(exact);
    let match: Fence | null = null;
    let matchLength = -1;
    for (const fence of fences) {
      if (fence.key !== key) continue;
      // Prefix-related pending calls may belong to different blocks.
      if (fence.result === null) {
        if (fence.pending?.code === code) return promote(fence);
        continue;
      }
      const reach = reachInto(fence, code);
      if (reach <= matchLength) continue;
      match = fence;
      matchLength = reach;
    }
    if (match) return promote(match);
    const fence: Fence = {
      key,
      code: "",
      result: null,
      meta: null,
      lines: [],
      committedLength: 0,
      state: undefined,
      liveTokens: null,
      lastTokenizedAt: 0,
      trailing: null,
      pending: null,
      seeded: false,
    };
    fences.unshift(fence);
    return fence;
  };

  const tokenize = (
    fence: Fence,
    highlighter: Highlighter,
    code: string,
    language: BundledLanguage,
    themes: ThemeNames,
    budgetMs: number,
  ): HighlightResult => {
    const lang = highlighter.getLoadedLanguages().includes(language)
      ? language
      : PLAIN_TEXT;
    const tokenizeFrom = (text: string) =>
      highlighter.codeToTokens(text, {
        lang,
        themes,
        grammarState: fence.state,
        ...TOKENIZE_LIMITS,
      });

    const lastNewline = code.lastIndexOf("\n");
    if (lastNewline >= fence.committedLength && budgetMs < Number.POSITIVE_INFINITY) {
      const deadline = monotonicNow() + budgetMs;
      while (fence.committedLength <= lastNewline) {
        const newline = code.indexOf("\n", fence.committedLength);
        const end =
          newline > fence.committedLength && code[newline - 1] === "\r"
            ? newline - 1
            : newline;
        const completed = tokenizeFrom(code.slice(fence.committedLength, end));
        for (const line of completed.tokens) {
          fence.lines.push(shiftLine(line, fence.committedLength));
        }
        fence.state = completed.grammarState;
        fence.committedLength = newline + 1;
        fence.liveTokens = null;
        fence.meta = stripTokens(completed);
        if (monotonicNow() >= deadline) break;
      }
      if (fence.committedLength <= lastNewline) {
        fence.lastTokenizedAt = monotonicNow();
        const tail = code.slice(fence.committedLength).split("\n");
        return {
          ...(fence.meta as ResultMeta),
          tokens: [...fence.lines, ...tail.map(plainLine)],
        };
      }
    }

    if (lastNewline >= fence.committedLength) {
      // Shiki omits CR from CRLF token content.
      const completedEnd =
        lastNewline > 0 && code[lastNewline - 1] === "\r"
          ? lastNewline - 1
          : lastNewline;
      const completed = tokenizeFrom(
        code.slice(fence.committedLength, completedEnd),
      );
      for (const line of completed.tokens) {
        fence.lines.push(shiftLine(line, fence.committedLength));
      }
      fence.state = completed.grammarState;
      fence.committedLength = lastNewline + 1;
      fence.liveTokens = null;
      fence.meta = stripTokens(completed);
    }

    const live = code.slice(fence.committedLength);
    const liveResult = tokenizeFrom(live);
    fence.meta = stripTokens(liveResult);
    fence.liveTokens = shiftLine(
      liveResult.tokens[0] ?? ([] as unknown as TokenLine),
      fence.committedLength,
    );
    fence.lastTokenizedAt = monotonicNow();
    return { ...fence.meta, tokens: [...fence.lines, fence.liveTokens] };
  };

  const approximateResult = (fence: Fence, code: string): HighlightResult => {
    const exact = fence.result as HighlightResult;
    // A live line may have grown mid-token, so render the whole tail plain.
    const keptLines = exact.tokens.slice(0, fence.lines.length);
    const tail = code.slice(fence.committedLength).split("\n");
    return { ...exact, tokens: [...keptLines, ...tail.map(plainLine)] };
  };

  const resetFence = (fence: Fence): void => {
    fence.lines = [];
    fence.committedLength = 0;
    fence.state = undefined;
    fence.liveTokens = null;
    fence.meta = null;
  };

  const update = (
    fence: Fence,
    highlighter: Highlighter,
    code: string,
    language: BundledLanguage,
    themes: ThemeNames,
    budgetMs = Number.POSITIVE_INFINITY,
  ): HighlightResult | null => {
    let result: HighlightResult;
    try {
      result = tokenize(fence, highlighter, code, language, themes, budgetMs);
    } catch (error) {
      console.error("[Unsloth Code] Failed to highlight code:", error);
      resetFence(fence);
      // A fence that never produced tokens has no anchor, so a failing block would strand one per render.
      if (fence.result === null) dropFence(fence);
      return null;
    }
    cachedCharacters += code.length - fence.code.length;
    dropCodeIndex(fence);
    fence.code = code;
    fence.result = result;
    fencesByCode.set(codeKey(fence.key, code), fence);
    evict();
    return result;
  };

  const settlePending = (
    fence: Fence,
    highlighter: Highlighter,
    language: BundledLanguage,
    themes: ThemeNames,
  ): void => {
    const pending = fence.pending;
    clearTrailing(fence);
    if (!pending) return;
    const refreshed = update(
      fence,
      highlighter,
      pending.code,
      language,
      themes,
    );
    if (refreshed) notifyPending(pending, refreshed);
  };

  const loadHighlighter = (
    key: string,
    language: BundledLanguage,
    themes: [ThemeInput, ThemeInput],
    resume: (highlighter: Highlighter) => void,
  ): Highlighter | null => {
    const loading = highlighters.get(key);
    if (loading) return loading.highlighter;
    const entry: { highlighter: Highlighter | null } = { highlighter: null };
    highlighters.set(key, entry);
    createHighlighter({
      themes,
      langs: [SUPPORTED_LANGUAGES.has(language) ? language : PLAIN_TEXT],
      engine,
    })
      .then((highlighter) => {
        entry.highlighter = highlighter;
        resume(highlighter);
      })
      .catch((error) => {
        console.error("[Unsloth Code] Failed to highlight code:", error);
        highlighters.delete(key);
        // Failed callbacks must not pin fences or fire after a later retry.
        for (const waiting of fences) {
          if (waiting.key === key) clearTrailing(waiting);
        }
        evict();
      });
    return null;
  };

  const plugin: UnslothCodePlugin = {
    name: "shiki",
    type: "code-highlighter",
    cached: (opts) => {
      const exact = fencesByCode.get(codeKey(fenceKeyOf(opts), opts.code));
      return exact && exact.code === opts.code && exact.result
        ? promote(exact).result
        : null;
    },
    cover: (opts) => {
      const key = fenceKeyOf(opts);
      const exact = fencesByCode.get(codeKey(key, opts.code));
      if (exact && exact.code === opts.code && exact.result) {
        return { uncovered: 0, result: exact.result };
      }
      let match: Fence | null = null;
      let matchLength = -1;
      for (const fence of fences) {
        if (fence.key !== key) continue;
        const reach = reachInto(fence, opts.code);
        if (reach <= matchLength) continue;
        match = fence;
        matchLength = reach;
      }
      if (match === null) return { uncovered: opts.code.length, result: null };
      return {
        uncovered:
          opts.code.length - Math.min(match.committedLength, opts.code.length),
        result: approximateResult(match, opts.code),
      };
    },
    seed: (opts, result) => {
      const key = fenceKeyOf(opts);
      const exact = fencesByCode.get(codeKey(key, opts.code));
      if (exact && exact.code === opts.code && exact.result) return;
      const fence: Fence = {
        key,
        code: opts.code,
        result,
        meta: stripTokens(result),
        lines: result.tokens,
        committedLength: opts.code.length,
        state: undefined,
        liveTokens: null,
        lastTokenizedAt: monotonicNow(),
        trailing: null,
        pending: null,
        seeded: true,
      };
      fences.unshift(fence);
      cachedCharacters += opts.code.length;
      fencesByCode.set(codeKey(key, opts.code), fence);
      evict();
    },
    getSupportedLanguages: () => SUPPORTED_LANGUAGE_LIST,
    getThemes: () => defaultThemes,
    supportsLanguage: (language) =>
      SUPPORTED_LANGUAGES.has(normalizeLanguage(language)),
    highlight: (
      opts: HighlightOptions,
      callback?: (result: HighlightResult) => void,
      exact = false,
    ): HighlightResult | null => {
      const language = normalizeLanguage(opts.language);
      const themes: ThemeNames = {
        light: themeName(opts.themes[0]),
        dark: themeName(opts.themes[1]),
      };
      const key = `${language} ${themeKey(opts.themes[0])} ${themeKey(opts.themes[1])}`;
      const fence = findFence(key, opts.code);

      const partial = fence.committedLength <= fence.code.lastIndexOf("\n");
      if (fence.result && fence.code === opts.code && !(exact && partial)) {
        const pending = fence.pending;
        if (
          pending !== null &&
          shedsClosingRun(opts.code, pending.code)
        ) {
          // Settle the queued body before serving the shorter exact hit, so neither caller loses state.
          const loaded = highlighters.get(key)?.highlighter;
          if (loaded) {
            const exact = fence.result;
            settlePending(fence, loaded, language, themes);
            return exact;
          }
        }
        return fence.result;
      }

      const highlighter = loadHighlighter(key, language, opts.themes, (ready) => {
        // Oldest-first snapshot because updates can evict fences.
        for (const waiting of [...fences].reverse()) {
          const pending = waiting.pending;
          if (waiting.key !== key || !pending) continue;
          clearTrailing(waiting);
          const resumed = update(waiting, ready, pending.code, language, themes);
          if (resumed) notifyPending(pending, resumed);
        }
      });
      if (!highlighter) {
        queuePending(fence, opts.code, callback);
        evict();
        return null;
      }
      if (exact && fence.result !== null && fence.code !== opts.code) {
        return update(forkFence(fence), highlighter, opts.code, language, themes);
      }

      const pending = fence.pending;
      if (pending && shedsClosingRun(opts.code, pending.code)) {
        settlePending(fence, highlighter, language, themes);
      } else {
        clearTrailing(fence);
      }
      const elapsed = monotonicNow() - fence.lastTokenizedAt;
      const grewLargeFence =
        fence.result !== null &&
        fence.code.length >= MIN_INCREMENTAL_CHARS &&
        opts.code.length > fence.code.length;
      if (!exact && grewLargeFence && elapsed < REFRESH_MS) {
        const result = approximateResult(fence, opts.code);
        queuePending(fence, opts.code, callback);
        const refresh = (): void => {
          const pending = fence.pending;
          fence.trailing = null;
          fence.pending = null;
          if (!pending) return;
          const refreshed = update(
            fence,
            highlighter,
            pending.code,
            language,
            themes,
            REFRESH_SLICE_MS,
          );
          if (refreshed) notifyPending(pending, refreshed);
          if (
            refreshed
            && fence.code === pending.code
            && fence.committedLength <= pending.code.lastIndexOf("\n")
          ) {
            fence.pending = pending;
            fence.trailing = setTimeout(refresh, 0);
          }
        };
        fence.trailing = setTimeout(
          refresh,
          partial ? 0 : Math.max(0, REFRESH_MS - elapsed),
        );
        return result;
      }
      return update(fence, highlighter, opts.code, language, themes);
    },
    highlightExact: (opts, callback) => plugin.highlight(opts, callback, true),
  };
  return plugin;
}
