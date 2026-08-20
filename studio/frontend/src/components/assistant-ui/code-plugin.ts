// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CodeHighlighterPlugin,
  CodePluginOptions,
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import {
  type BundledLanguage,
  type GrammarState,
  type ThemedToken,
  bundledLanguages,
  bundledLanguagesInfo,
  createHighlighter,
} from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import type { CodeHighlightGate } from "./code-highlight-gate";

/**
 * The attribute the plugin stamps on a gated block's first token span.
 *
 * The plugin is handed `{code, language, themes}` and nothing else -- no element, no ref, no key --
 * so it cannot ask where a block is, and a viewport gate has to be told. Streamdown spreads a
 * token's `htmlAttrs` onto the span it renders, which is the one channel from this layer into the
 * DOM, so the block's gate id rides out on its first token and a binder can find the element by
 * `[data-code-fence="..."]`. Only stamped when a gate is present: with the flag off no attribute
 * is emitted at all.
 */
export const CODE_FENCE_ATTRIBUTE = "data-code-fence";

let nextFenceId = 0;

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

const normalizeLanguage = (language: string): BundledLanguage => {
  const key = language.trim().toLowerCase();
  const alias = LANGUAGE_ALIAS_OVERRIDES[key] ?? SHIKI_LANGUAGE_ALIASES[key];
  return alias ?? (key as BundledLanguage);
};

// A streaming fence re-enters highlight() every frame with the whole block, so
// Shiki re-tokenizes it in full ~60x/sec. Past MIN_INCREMENTAL_CHARS, cache the
// completed lines and their grammar state so each refresh tokenizes only new
// text, keeping the 250 ms plain-tail cadence.
export const MIN_INCREMENTAL_CHARS = 2000;
const REFRESH_MS = 250;
// An unvirtualized thread mounts every fence. Bound both their count and source
// size; token data measured roughly 30 bytes per source character.
const MAX_FENCES = 512;
const MAX_CACHED_CHARACTERS = 512_000;

// Wall-clock Date.now() can step backwards (NTP, sleep resume) and make
// `elapsed` negative; the throttle only needs elapsed time, so stay monotonic.
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

type FenceContext = {
  language: BundledLanguage;
  themes: ThemeNames;
};

type Fence = {
  key: string;
  code: string;
  /** Gate id, assigned on the first gated call. Empty while ungated. */
  id: string;
  /**
   * Callbacks retained past the call that supplied them.
   *
   * Streamdown calls `highlight()` from an effect keyed on the code string, so a block that has
   * stopped streaming calls in exactly ONCE and never again. Its `useState` setter is then the
   * only way to change what it shows, which is what makes a later downgrade or re-highlight
   * possible at all. The setter is referentially stable per mounted component, so this holds one
   * entry per mounted block and is cleared when the fence is dropped.
   */
  subscribers: Set<HighlightCallback>;
  /** The most recent code seen for this fence, which may be ahead of `code` while gated plain. */
  live: string | null;
  /** What a later re-highlight needs to call the tokenizer with. */
  context: FenceContext | null;
  result: HighlightResult | null;
  meta: ResultMeta | null;
  /** Absolute-offset tokens for completed lines. */
  lines: TokenLine[];
  /** Source prefix covered by `lines`. */
  committedLength: number;
  state: GrammarState | undefined;
  liveTokens: TokenLine | null;
  lastTokenizedAt: number;
  trailing: ReturnType<typeof setTimeout> | null;
  pending: Pending | null;
};

// Tokens without colors preserve the previous plain-tail rendering.
const plainLine = (content: string): TokenLine =>
  [{ content, offset: 0 } as unknown as ThemedToken] as TokenLine;

/** The source text of one tokenized line. Shiki splits a line into runs, it does not alter it. */
const lineText = (line: TokenLine): string =>
  line.length === 1
    ? line[0].content
    : line.map((token) => token.content).join("");

/**
 * The same lines, with the colours taken off.
 *
 * Derived from the tokenized result rather than re-split from the source, so the plain rendering
 * has the same NUMBER of lines and the same TEXT on each line as the highlighted one by
 * construction. That is what makes the swap layout-neutral: `<pre>` does not wrap, so a block's
 * height is its line count times its line height, and neither changes here.
 */
const plainTokens = (tokens: TokenLine[]): TokenLine[] =>
  tokens.map((line) => plainLine(lineText(line)));

/**
 * Put `id` on the block's first rendered token.
 *
 * Skips lines the renderer turns into a bare newline (empty, or a single empty token), because
 * those emit no span to carry it. A block with no such line -- an empty fence -- goes unstamped
 * and is never located, which the gate reads as "not measured" and leaves highlighted.
 */
const stampFence = (result: HighlightResult, id: string): HighlightResult => {
  const index = result.tokens.findIndex(
    (line) => line.length > 0 && !(line.length === 1 && line[0].content === ""),
  );
  if (index < 0) return result;
  const [first, ...rest] = result.tokens[index];
  const tokens = [...result.tokens];
  tokens[index] = [
    { ...first, htmlAttrs: { ...first.htmlAttrs, [CODE_FENCE_ATTRIBUTE]: id } },
    ...rest,
  ] as TokenLine;
  return { ...result, tokens };
};

const themeName = (theme: ThemeInput): string =>
  typeof theme === "string" ? theme : (theme.name ?? "custom");

// Two custom themes can share a name, or carry none, so a cache key built from
// names alone would serve one theme's tokens for another. Reduce each distinct
// definition to a short id once: the key is compared on every lookup.
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

// Markdown reports a closing fence as body until it recognizes it, so that
// line is all a fence can lose: up to three spaces, one run of backticks or
// tildes, then spaces. It also starts a line, so the body it leaves behind ends
// at a newline. The run can be short, because the cached code is the last text
// tokenized and the run may still have been arriving then.
const CLOSING_FENCE = /^ {0,3}(?:`+|~+)[ \t]*$/;

/** Whether `longer` is one fence's body, `shorter`, plus its closing line. */
const shedsClosingRun = (shorter: string, longer: string): boolean =>
  (shorter === "" || shorter.endsWith("\n")) &&
  longer.startsWith(shorter) &&
  CLOSING_FENCE.test(longer.slice(shorter.length));

export type StudioCodePluginOptions = CodePluginOptions & {
  /**
   * When supplied, blocks the gate reports as far from the viewport render plain.
   *
   * Optional on purpose: with no gate the plugin takes the exact path it always has, every return
   * below is the object it always returned, and no attribute is stamped. That is the flag-off
   * build.
   */
  gate?: CodeHighlightGate;
};

export function createCodePlugin(
  options: StudioCodePluginOptions = {},
): CodeHighlighterPlugin {
  const defaultThemes: [ThemeInput, ThemeInput] = options.themes ?? [
    "github-light",
    "github-dark",
  ];
  const gate = options.gate;
  const engine = createJavaScriptRegexEngine({ forgiving: true });
  const highlighters = new Map<string, { highlighter: Highlighter | null }>();
  // Most recently used first.
  const fences: Fence[] = [];
  // Avoid scanning every fence for exact cache hits.
  const fencesByCode = new Map<string, Fence>();
  // Gated fences only, so a gate notification can find the block it names.
  const fencesById = new Map<string, Fence>();
  // Which fence currently owns each retained callback. A block whose code changes enough to start
  // a new fence must not be left subscribed to the old one, or a stale result can arrive last and
  // overwrite a newer one -- streamdown does no cancellation of its own.
  const subscriberOwners = new Map<HighlightCallback, Fence>();
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

  // Every result that leaves the plugin has to carry the stamp, not just the ones returned
  // synchronously: a block whose grammar arrived late is delivered through here, and an unstamped
  // delivery would erase the attribute that is how it gets located.
  const deliver = (fence: Fence, result: HighlightResult): HighlightResult =>
    gate === undefined || fence.id === ""
      ? result
      : stampFence(result, fence.id);

  const notifyPending = (
    fence: Fence,
    pending: Pending,
    result: HighlightResult,
  ): void => {
    for (const callback of pending.callbacks) callback(deliver(fence, result));
  };

  /** Push a result to every block that has this fence's content mounted. */
  const publish = (fence: Fence, result: HighlightResult): void => {
    for (const subscriber of fence.subscribers)
      subscriber(deliver(fence, result));
  };

  const dropFence = (fence: Fence): void => {
    const index = fences.indexOf(fence);
    if (index < 0) return;
    fences.splice(index, 1);
    cachedCharacters -= fence.code.length;
    dropCodeIndex(fence);
    clearTrailing(fence);
    // Retained callbacks hold their component alive, so a dropped fence must let go of them.
    for (const subscriber of fence.subscribers)
      subscriberOwners.delete(subscriber);
    fence.subscribers.clear();
    if (fence.id !== "") {
      fencesById.delete(fence.id);
      gate?.forget(fence.id);
    }
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

  const promote = (fence: Fence): Fence => {
    const index = fences.indexOf(fence);
    if (index > 0) {
      fences.splice(index, 1);
      fences.unshift(fence);
    }
    return fence;
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
      const anchor = fence.code;
      // A block that lost more than its closing delimiter is a different fence;
      // sharing this entry would cancel the refresh it has queued.
      const reaches =
        code.startsWith(anchor) ||
        (code.length >= fence.committedLength && shedsClosingRun(code, anchor));
      const reach = Math.min(anchor.length, code.length);
      if (!anchor || reach <= matchLength || !reaches) continue;
      match = fence;
      matchLength = reach;
    }
    if (match) return promote(match);
    const fence: Fence = {
      key,
      code: "",
      id: "",
      subscribers: new Set(),
      live: null,
      context: null,
      result: null,
      meta: null,
      lines: [],
      committedLength: 0,
      state: undefined,
      liveTokens: null,
      lastTokenizedAt: 0,
      trailing: null,
      pending: null,
    };
    fences.unshift(fence);
    return fence;
  };

  /** Tokenize what `fence` has not seen yet and return the whole fence. */
  const tokenize = (
    fence: Fence,
    highlighter: Highlighter,
    code: string,
    language: BundledLanguage,
    themes: ThemeNames,
  ): HighlightResult => {
    const lang = highlighter.getLoadedLanguages().includes(language)
      ? language
      : PLAIN_TEXT;
    const tokenizeFrom = (text: string) =>
      highlighter.codeToTokens(text, {
        lang,
        themes,
        grammarState: fence.state,
      });

    // Commit every newly completed line.
    const lastNewline = code.lastIndexOf("\n");
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
      liveResult.tokens[0] ?? [],
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

  // Leave the block plain if highlighting fails instead of breaking render.
  const update = (
    fence: Fence,
    highlighter: Highlighter,
    code: string,
    language: BundledLanguage,
    themes: ThemeNames,
  ): HighlightResult | null => {
    let result: HighlightResult;
    try {
      result = tokenize(fence, highlighter, code, language, themes);
    } catch (error) {
      console.error("[Studio Code] Failed to highlight code:", error);
      resetFence(fence);
      // A fence that never produced tokens has no anchor to match on, so a
      // block that keeps failing would strand a new one on every render.
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
    if (refreshed) notifyPending(fence, pending, refreshed);
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
        console.error("[Studio Code] Failed to highlight code:", error);
        highlighters.delete(key);
        // Failed callbacks must not pin fences or fire after a later retry.
        for (const waiting of fences) {
          if (waiting.key === key) clearTrailing(waiting);
        }
        evict();
      });
    return null;
  };

  /**
   * The plugin's own path, unchanged. Everything the gate does happens around this, never inside
   * it: with no gate, `highlight()` is this function and nothing else.
   */
  const resolve = (
    fence: Fence,
    opts: HighlightOptions,
    callback: HighlightCallback | undefined,
    key: string,
    language: BundledLanguage,
    themes: ThemeNames,
  ): HighlightResult | null => {
    {
      if (fence.result && fence.code === opts.code) {
        const pending = fence.pending;
        if (pending !== null && shedsClosingRun(opts.code, pending.code)) {
          // This may be one fence shedding its closing run or a shorter sibling
          // reusing the same entry. Settle the queued body before serving the
          // shorter exact hit so neither caller loses its final highlighted state.
          const loaded = highlighters.get(key)?.highlighter;
          if (loaded) {
            const exact = fence.result;
            settlePending(fence, loaded, language, themes);
            return exact;
          }
        }
        return fence.result;
      }

      const highlighter = loadHighlighter(
        key,
        language,
        opts.themes,
        (ready) => {
          // Use a stable, oldest-first snapshot because updates can evict fences.
          for (const waiting of [...fences].reverse()) {
            const pending = waiting.pending;
            if (waiting.key !== key || !pending) continue;
            clearTrailing(waiting);
            const resumed = update(
              waiting,
              ready,
              pending.code,
              language,
              themes,
            );
            if (resumed) notifyPending(waiting, pending, resumed);
          }
        },
      );
      if (!highlighter) {
        queuePending(fence, opts.code, callback);
        evict();
        return null;
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
      if (grewLargeFence && elapsed < REFRESH_MS) {
        const result = approximateResult(fence, opts.code);
        queuePending(fence, opts.code, callback);
        fence.trailing = setTimeout(
          () => {
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
            );
            if (refreshed) notifyPending(fence, pending, refreshed);
          },
          Math.max(0, REFRESH_MS - elapsed),
        );
        return result;
      }
      return update(fence, highlighter, opts.code, language, themes);
    }
  };

  // A gate notification names one block. Turn it into that block's new rendering.
  gate?.subscribe((id) => {
    const fence = fencesById.get(id);
    const code = fence?.live;
    if (!fence || code === null || code === undefined) return;
    if (fence.subscribers.size === 0) return;
    if (gate.mode(id) === "plain") {
      // Downgrade only a block that HAS a highlighted result: `plainTokens` reads its line
      // structure off that result, which is what guarantees the two renderings are the same
      // height. Until then the block is highlighted, which is also what the gate reports for a
      // block it has never located.
      const exact = fence.result;
      if (exact === null || fence.code !== code) return;
      publish(fence, { ...exact, tokens: plainTokens(exact.tokens) });
      return;
    }
    const context = fence.context;
    const highlighter = highlighters.get(fence.key)?.highlighter;
    if (!context || !highlighter) return;
    // Back inside the buffer. `update` tokenizes from `committedLength`, so a block that grew
    // while it was plain costs only its uncommitted tail, and the incremental caches that the
    // streaming path depends on are the ones being reused here.
    const refreshed = update(
      fence,
      highlighter,
      code,
      context.language,
      context.themes,
    );
    if (refreshed) publish(fence, refreshed);
  });

  return {
    name: "shiki",
    type: "code-highlighter",
    getSupportedLanguages: () => SUPPORTED_LANGUAGE_LIST,
    getThemes: () => defaultThemes,
    supportsLanguage: (language) =>
      SUPPORTED_LANGUAGES.has(normalizeLanguage(language)),
    highlight: (
      opts: HighlightOptions,
      callback?: (result: HighlightResult) => void,
    ): HighlightResult | null => {
      const language = normalizeLanguage(opts.language);
      const themes: ThemeNames = {
        light: themeName(opts.themes[0]),
        dark: themeName(opts.themes[1]),
      };
      const key = `${language} ${themeKey(opts.themes[0])} ${themeKey(opts.themes[1])}`;
      const fence = findFence(key, opts.code);
      if (gate === undefined)
        return resolve(fence, opts, callback, key, language, themes);

      if (fence.id === "") {
        fence.id = `sf${nextFenceId++}`;
        fencesById.set(fence.id, fence);
      }
      if (callback) {
        const previous = subscriberOwners.get(callback);
        if (previous !== fence) {
          previous?.subscribers.delete(callback);
          subscriberOwners.set(callback, fence);
          fence.subscribers.add(callback);
        }
      }
      fence.context = { language, themes };
      // Growth from code this fence has ALREADY seen is the only streaming signal available here,
      // and it has to be recorded before the mode is read: a block still being streamed into is in
      // view by definition and must not be handed a plain rendering on the frame its next chunk
      // lands. A block arriving complete is not a stream -- treating a first sighting as growth
      // would hold every block in a freshly mounted thread highlighted for the whole grace window,
      // which is exactly the thread this mechanism is for.
      const streamed =
        fence.live !== null && opts.code.length > fence.live.length;
      fence.live = opts.code;
      if (streamed) gate.markStreaming(fence.id);

      if (
        gate.mode(fence.id) === "plain" &&
        fence.result !== null &&
        fence.code === opts.code
      ) {
        return deliver(fence, {
          ...fence.result,
          tokens: plainTokens(fence.result.tokens),
        });
      }
      const result = resolve(fence, opts, callback, key, language, themes);
      if (result === null) return null;
      // The element does not exist until this result has been rendered, so this is the earliest
      // anything can go and measure it. Every block is therefore highlighted once and downgraded
      // afterwards, never plain on arrival -- which is the right trade here, because the cost
      // being removed is the standing presence of the spans and not the work of building them.
      gate.announce(fence.id);
      return deliver(fence, result);
    },
  };
}
