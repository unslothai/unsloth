// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  DEFAULT_TARGET_LANGUAGE,
  findTranslateLanguage,
  suggestedTargetLanguage,
  translatesLanguage,
} from "./languages";
import {
  CHAT_TRANSLATE_MODEL,
  CUSTOM_TRANSLATE_MODEL,
  type CustomTranslateModel,
  DEFAULT_TRANSLATE_MODEL,
  type TranslateModelId,
  type TranslatePromptKind,
  isTranslateModelId,
  parseCustomTranslateModel,
  translateModelKey,
  translatePromptKind,
  translatePromptModel,
} from "./models";
import {
  type TranslateHistoryEntry,
  addHistoryEntry,
  parseHistory,
} from "./translate-history";
import {
  type TranslatePhase,
  TranslateRunError,
  type TranslateRunErrorCode,
  runTranslation,
} from "./translate-runner";
import {
  TRANSLATE_MODES,
  TRANSLATE_TONES,
  type TranslateMode,
  type TranslatePromptOptions,
  type TranslateTone,
  clipText,
} from "./translate-text";

const PREFS_KEY = "unsloth_translate_prefs";
const HISTORY_KEY = "unsloth_translate_history";
const RECENT_LANGUAGES = 3;
export const AUTO_SOURCE = "auto";
export const TRANSLATE_SOURCE_MAX_CHARS = 50_000;

export type TranslatePrefs = {
  from: string;
  target: string;
  recentSources: string[];
  recentTargets: string[];
  modelId: TranslateModelId;
  customModel: CustomTranslateModel | null;
  mode: TranslateMode;
  tone: TranslateTone;
  showPrompt: boolean;
  customPrompts: Partial<Record<TranslatePromptKind, string>>;
  autoTranslate: boolean;
  saveHistory: boolean;
};

export type TranslateError =
  | { code: TranslateRunErrorCode; message?: undefined }
  | { code: "other"; message: string };

type Result = {
  source: string;
  from: string;
  target: string;
  mode: TranslateMode;
  tone: TranslateTone;
  modelKey: string;
  custom: string | null;
  customName: string | null;
};

type TranslateState = {
  prefs: TranslatePrefs;
  history: TranslateHistoryEntry[];
  source: string;
  output: string;
  phase: TranslatePhase | null;
  error: TranslateError | null;
  result: Result | null;
  attempt: Result | null;
  /** Models that served a translation this session: translating as you type never starts a download. */
  readyModels: string[];
  setSource: (source: string) => void;
  setPrefs: (patch: Partial<TranslatePrefs>) => void;
  setFrom: (from: string) => void;
  setTarget: (target: string) => void;
  /** `detected` stands in for an automatic source. */
  swap: (detected?: string | null) => void;
  outputToSource: () => void;
  /** `auto` runs come from typing: each replaces the draft's previous auto entry in the history. */
  translate: (options?: { auto?: boolean }) => Promise<void>;
  setModelReady: (modelKey: string, ready: boolean) => void;
  stop: () => void;
  clear: () => void;
  restore: (entry: TranslateHistoryEntry) => void;
  removeHistory: (id: string) => void;
  clearHistory: () => void;
};

function readJson(key: string): unknown {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): boolean {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function languageTabs(
  current: string,
  recent: readonly string[],
): string[] {
  if (current === AUTO_SOURCE || recent.includes(current)) return [...recent];
  return [current, ...recent].slice(0, RECENT_LANGUAGES);
}

function withLanguage(tabs: string[], code: string): string[] {
  return tabs.includes(code)
    ? tabs
    : [code, ...tabs].slice(0, RECENT_LANGUAGES);
}

export function translatesWithDedicated(
  prefs: Pick<TranslatePrefs, "modelId" | "customModel">,
): boolean {
  return (
    translatePromptKind(
      translatePromptModel(prefs.modelId, prefs.customModel),
    ) === "dedicated"
  );
}

const suggestedTarget = () =>
  typeof navigator === "undefined"
    ? DEFAULT_TARGET_LANGUAGE
    : suggestedTargetLanguage(navigator.languages ?? []);

/** Switching to Hy-MT2 drops a chat-only pick; the recent tabs keep it for the next chat model. */
function fitLanguages(prefs: TranslatePrefs): TranslatePrefs {
  if (!translatesWithDedicated(prefs)) return prefs;
  const fits = (code: string) => translatesLanguage(code, true);
  const from =
    prefs.from === AUTO_SOURCE || fits(prefs.from) ? prefs.from : AUTO_SOURCE;
  const target = fits(prefs.target)
    ? prefs.target
    : (prefs.recentTargets.find(fits) ?? suggestedTarget());
  return from === prefs.from && target === prefs.target
    ? prefs
    : { ...prefs, from, target };
}

function loadPrefs(): TranslatePrefs {
  const saved = (readJson(PREFS_KEY) ?? {}) as Partial<TranslatePrefs>;
  const known = (code: unknown): code is string =>
    typeof code === "string" && findTranslateLanguage(code) !== undefined;
  const customModel = parseCustomTranslateModel(saved.customModel);
  return fitLanguages({
    from: known(saved.from) ? saved.from : AUTO_SOURCE,
    target: known(saved.target) ? saved.target : suggestedTarget(),
    recentSources: Array.isArray(saved.recentSources)
      ? saved.recentSources.filter(known).slice(0, RECENT_LANGUAGES)
      : [],
    recentTargets: Array.isArray(saved.recentTargets)
      ? saved.recentTargets.filter(known).slice(0, RECENT_LANGUAGES)
      : [],
    modelId:
      isTranslateModelId(saved.modelId) &&
      // The menu no longer offers "the model picked in Chat".
      saved.modelId !== CHAT_TRANSLATE_MODEL &&
      (saved.modelId !== CUSTOM_TRANSLATE_MODEL || customModel)
        ? saved.modelId
        : DEFAULT_TRANSLATE_MODEL,
    customModel,
    mode: TRANSLATE_MODES.includes(saved.mode as TranslateMode)
      ? (saved.mode as TranslateMode)
      : "automatic",
    tone: TRANSLATE_TONES.includes(saved.tone as TranslateTone)
      ? (saved.tone as TranslateTone)
      : "default",
    showPrompt: saved.showPrompt === true,
    customPrompts: Object.fromEntries(
      (["dedicated", "chat"] as const)
        .map((kind) => [kind, saved.customPrompts?.[kind]] as const)
        .filter(
          (entry): entry is [TranslatePromptKind, string] =>
            typeof entry[1] === "string" && entry[1].trim() !== "",
        ),
    ),
    autoTranslate: saved.autoTranslate === true,
    saveHistory: saved.saveHistory !== false,
  });
}

function saveHistory(
  history: TranslateHistoryEntry[],
): TranslateHistoryEntry[] {
  let kept = history;
  // A full storage drops the oldest half until the rest fits, but never writes over it with nothing.
  while (!writeJson(HISTORY_KEY, kept)) {
    if (kept.length <= 1) return history;
    kept = kept.slice(0, Math.floor(kept.length / 2));
  }
  return kept;
}

const storedHistory = () => parseHistory(readJson(HISTORY_KEY));

function resultOf(source: string, prefs: TranslatePrefs): Result {
  return {
    source,
    from: prefs.from,
    target: prefs.target,
    mode: prefs.mode,
    tone: prefs.tone,
    modelKey: translateModelKey(prefs.modelId, prefs.customModel),
    custom: customPromptOf(prefs),
    customName:
      prefs.modelId === CUSTOM_TRANSLATE_MODEL
        ? (prefs.customModel?.name ?? null)
        : null,
  };
}

export function customPromptOf(
  prefs: Pick<TranslatePrefs, "modelId" | "customModel" | "customPrompts">,
): string | null {
  const model = translatePromptModel(prefs.modelId, prefs.customModel);
  return prefs.customPrompts[translatePromptKind(model)] ?? null;
}

export function translatePromptOptions(
  prefs: Pick<
    TranslatePrefs,
    | "from"
    | "target"
    | "mode"
    | "tone"
    | "modelId"
    | "customModel"
    | "customPrompts"
  >,
): TranslatePromptOptions | null {
  const target = findTranslateLanguage(prefs.target);
  if (!target) return null;
  return {
    sourceName: findTranslateLanguage(prefs.from)?.name ?? null,
    targetName: target.name,
    mode: prefs.mode,
    tone: prefs.tone,
    custom: customPromptOf(prefs),
  };
}

let controller: AbortController | null = null;
let autoEntryId: string | null = null;
let flushStreamed: (() => void) | null = null;

// One render per frame: a render per streamed token overruns React's nested update limit.
const nextFrame = (callback: () => void): (() => void) => {
  if (typeof requestAnimationFrame === "function") {
    const id = requestAnimationFrame(callback);
    return () => cancelAnimationFrame(id);
  }
  const id = setTimeout(callback, 16);
  return () => clearTimeout(id);
};

function sameDraft(a: string, b: string): boolean {
  let common = 0;
  while (common < a.length && common < b.length && a[common] === b[common])
    common++;
  return common > 0 && common >= Math.min(a.length, b.length) / 2;
}

export const useTranslateStore = create<TranslateState>((set, get) => ({
  prefs: loadPrefs(),
  history: storedHistory(),
  source: "",
  output: "",
  phase: null,
  error: null,
  result: null,
  attempt: null,
  readyModels: [],

  setSource: (source) => {
    const next = clipText(source, TRANSLATE_SOURCE_MAX_CHARS);
    if (next.trim() === "") {
      get().stop();
      set({
        source: next,
        output: "",
        error: null,
        result: null,
        attempt: null,
      });
      return;
    }
    set({ source: next });
  },

  setPrefs: (patch) => {
    const prefs = fitLanguages({ ...get().prefs, ...patch });
    set({ prefs });
    writeJson(PREFS_KEY, prefs);
  },

  setFrom: (from) => {
    const { prefs } = get();
    if (from === prefs.from) return;
    // The shown tabs: Detect must not drop one saved before recentSources.
    const shown = languageTabs(prefs.from, prefs.recentSources);
    get().setPrefs({
      from,
      recentSources: from === AUTO_SOURCE ? shown : withLanguage(shown, from),
    });
  },

  setTarget: (target) => {
    const { prefs } = get();
    if (target === prefs.target) return;
    get().setPrefs({
      target,
      recentTargets: withLanguage(
        languageTabs(prefs.target, prefs.recentTargets),
        target,
      ),
    });
  },

  swap: (detected) => {
    const state = get();
    const { prefs, output, phase, result } = state;
    const from = prefs.from === AUTO_SOURCE ? detected : prefs.from;
    if (!from || !findTranslateLanguage(from) || phase) return;
    // Text already in the target: swapping would leave automatic detection with nothing to do.
    if (prefs.from === AUTO_SOURCE && from === prefs.target) return;
    // Only a finished translation of the current draft replaces it; an edit is never lost.
    const current = output && result !== null && !translationIsStale(state);
    get().setFrom(prefs.target);
    get().setTarget(from);
    if (current) {
      set({
        source: clipText(output, TRANSLATE_SOURCE_MAX_CHARS),
        output: "",
        error: null,
        result: null,
        attempt: null,
      });
    }
  },

  outputToSource: () => {
    const { output, result, attempt, prefs, phase } = get();
    if (!output || phase) return;
    get().setFrom((result ?? attempt)?.target ?? prefs.target);
    set({
      source: clipText(output, TRANSLATE_SOURCE_MAX_CHARS),
      output: "",
      error: null,
      result: null,
      attempt: null,
    });
  },

  setModelReady: (modelKey, ready) => {
    const { readyModels } = get();
    if (readyModels.includes(modelKey) === ready) return;
    set({
      readyModels: ready
        ? [...readyModels, modelKey]
        : readyModels.filter((key) => key !== modelKey),
    });
  },

  translate: async (options) => {
    const auto = options?.auto === true;
    const { source, prefs } = get();
    const prompt = translatePromptOptions(prefs);
    if (!prompt || source.trim() === "") return;
    controller?.abort();
    const run = new AbortController();
    controller = run;
    const current = () => controller === run;
    const request = resultOf(source, prefs);
    set({
      phase: { kind: "preparing" },
      error: null,
      output: "",
      result: null,
      attempt: request,
    });
    let streamed: string | null = null;
    let cancelFrame: (() => void) | null = null;
    const dropFrame = () => {
      cancelFrame?.();
      cancelFrame = null;
      streamed = null;
    };
    const flushOutput = () => {
      cancelFrame?.();
      cancelFrame = null;
      if (streamed !== null && current()) set({ output: streamed });
      streamed = null;
    };
    flushStreamed = flushOutput;
    const release = () => {
      if (flushStreamed === flushOutput) flushStreamed = null;
    };
    try {
      const { output, model } = await runTranslation({
        text: source,
        prompt,
        modelId: prefs.modelId,
        custom: prefs.customModel,
        signal: run.signal,
        onPhase: (phase) => current() && set({ phase }),
        onOutput: (text) => {
          if (!current()) return;
          streamed = text;
          cancelFrame ??= nextFrame(flushOutput);
        },
      });
      dropFrame();
      release();
      if (!current()) return;
      controller = null;
      const state = get();
      // Onto the stored list, not this window's: another window may have added entries since.
      let history = state.history;
      if (state.prefs.saveHistory) {
        const stored = storedHistory();
        const previous = stored[0];
        const replaces =
          auto &&
          previous !== undefined &&
          previous.id === autoEntryId &&
          previous.target === prefs.target &&
          sameDraft(previous.source, source);
        const rest = replaces ? stored.slice(1) : stored;
        const added = addHistoryEntry(rest, {
          source,
          output,
          target: prefs.target,
          from: prefs.from === AUTO_SOURCE ? null : prefs.from,
          model,
        });
        if (added[0] !== rest[0]) {
          autoEntryId = auto ? added[0].id : null;
          history = saveHistory(added);
        } else {
          history = stored;
        }
      }
      set({
        phase: null,
        output,
        result: request,
        history,
        readyModels: state.readyModels.includes(request.modelKey)
          ? state.readyModels
          : [...state.readyModels, request.modelKey],
      });
    } catch (error) {
      flushOutput();
      release();
      if (!current()) return;
      controller = null;
      if (
        run.signal.aborted ||
        (error instanceof DOMException && error.name === "AbortError")
      ) {
        set({ phase: null });
        return;
      }
      set({
        phase: null,
        error:
          error instanceof TranslateRunError
            ? { code: error.code }
            : {
                code: "other",
                message: error instanceof Error ? error.message : String(error),
              },
      });
    }
  },

  stop: () => {
    if (!controller) return;
    flushStreamed?.();
    flushStreamed = null;
    controller.abort();
    controller = null;
    set({ phase: null });
  },

  clear: () => {
    get().stop();
    set({ source: "", output: "", error: null, result: null, attempt: null });
  },

  restore: (entry) => {
    get().stop();
    if (findTranslateLanguage(entry.target)) get().setTarget(entry.target);
    get().setFrom(
      entry.from && findTranslateLanguage(entry.from)
        ? entry.from
        : AUTO_SOURCE,
    );
    const shown = resultOf(entry.source, get().prefs);
    set({
      source: entry.source,
      output: entry.output,
      error: null,
      result: shown,
      attempt: shown,
    });
  },

  removeHistory: (id) => {
    set({
      history: saveHistory(storedHistory().filter((entry) => entry.id !== id)),
    });
  },

  clearHistory: () => {
    set({ history: [] });
    try {
      window.localStorage.removeItem(HISTORY_KEY);
    } catch {
      // Nothing was stored.
    }
  },
}));

export function translationIsStale(
  state: Pick<TranslateState, "result" | "source" | "prefs">,
): boolean {
  const { result, source, prefs } = state;
  return (
    result !== null &&
    (result.source !== source ||
      result.from !== prefs.from ||
      result.target !== prefs.target ||
      result.mode !== prefs.mode ||
      result.tone !== prefs.tone ||
      result.custom !== customPromptOf(prefs) ||
      result.modelKey !== translateModelKey(prefs.modelId, prefs.customModel))
  );
}

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === HISTORY_KEY || event.key === null) {
      useTranslateStore.setState({ history: storedHistory() });
    }
    // Another window's settings, so its next write does not undo this one's.
    if (event.key === PREFS_KEY || event.key === null) {
      useTranslateStore.setState({ prefs: loadPrefs() });
    }
  });
}
