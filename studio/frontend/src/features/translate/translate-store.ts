// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  DEFAULT_TARGET_LANGUAGE,
  findTranslateLanguage,
  suggestedTargetLanguage,
} from "./languages";
import {
  DEFAULT_TRANSLATE_MODEL,
  type TranslateModelId,
  isTranslateModelId,
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
} from "./translate-text";

const PREFS_KEY = "unsloth_translate_prefs";
const HISTORY_KEY = "unsloth_translate_history";
const RECENT_TARGETS = 3;
export const AUTO_SOURCE = "auto";
export const TRANSLATE_SOURCE_MAX_CHARS = 50_000;

export type TranslatePrefs = {
  /** A language code, or "auto" to let the model detect it. */
  from: string;
  target: string;
  recentTargets: string[];
  modelId: TranslateModelId;
  mode: TranslateMode;
  tone: TranslateTone;
  showPrompt: boolean;
  autoTranslate: boolean;
  saveHistory: boolean;
};

export type TranslateError =
  | { code: TranslateRunErrorCode; message?: undefined }
  | { code: "other"; message: string };

/** What the output pane holds was made from; a different draft marks it stale. */
type Result = {
  source: string;
  from: string;
  target: string;
  mode: TranslateMode;
  tone: TranslateTone;
  modelId: TranslateModelId;
};

type TranslateState = {
  prefs: TranslatePrefs;
  history: TranslateHistoryEntry[];
  source: string;
  output: string;
  phase: TranslatePhase | null;
  error: TranslateError | null;
  result: Result | null;
  /** The last request sent, finished or not. */
  attempt: Result | null;
  /** Models that served a translation this session: translating as you type never starts a download. */
  readyModels: TranslateModelId[];
  setSource: (source: string) => void;
  setPrefs: (patch: Partial<TranslatePrefs>) => void;
  setTarget: (target: string) => void;
  /** Flips the pair and moves the translation into the text box; needs a set source language. */
  swap: () => void;
  translate: () => Promise<void>;
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

function loadPrefs(): TranslatePrefs {
  const saved = (readJson(PREFS_KEY) ?? {}) as Partial<TranslatePrefs>;
  const known = (code: unknown): code is string =>
    typeof code === "string" && findTranslateLanguage(code) !== undefined;
  return {
    from: known(saved.from) ? saved.from : AUTO_SOURCE,
    target: known(saved.target)
      ? saved.target
      : typeof navigator === "undefined"
        ? DEFAULT_TARGET_LANGUAGE
        : suggestedTargetLanguage(navigator.languages ?? []),
    recentTargets: Array.isArray(saved.recentTargets)
      ? saved.recentTargets.filter(known).slice(0, RECENT_TARGETS)
      : [],
    modelId: isTranslateModelId(saved.modelId)
      ? saved.modelId
      : DEFAULT_TRANSLATE_MODEL,
    mode: TRANSLATE_MODES.includes(saved.mode as TranslateMode)
      ? (saved.mode as TranslateMode)
      : "automatic",
    tone: TRANSLATE_TONES.includes(saved.tone as TranslateTone)
      ? (saved.tone as TranslateTone)
      : "default",
    showPrompt: saved.showPrompt === true,
    autoTranslate: saved.autoTranslate === true,
    saveHistory: saved.saveHistory !== false,
  };
}

function saveHistory(
  history: TranslateHistoryEntry[],
): TranslateHistoryEntry[] {
  let kept = history;
  // A full storage drops the oldest half until the rest fits.
  while (!writeJson(HISTORY_KEY, kept) && kept.length > 0) {
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
    modelId: prefs.modelId,
  };
}

/** What the prompt needs from the settings; null when the target is not a known language. */
export function translatePromptOptions(
  prefs: Pick<TranslatePrefs, "from" | "target" | "mode" | "tone">,
): TranslatePromptOptions | null {
  const target = findTranslateLanguage(prefs.target);
  if (!target) return null;
  return {
    sourceName: findTranslateLanguage(prefs.from)?.name ?? null,
    targetName: target.name,
    mode: prefs.mode,
    tone: prefs.tone,
  };
}

let controller: AbortController | null = null;

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
    const next = source.slice(0, TRANSLATE_SOURCE_MAX_CHARS);
    if (next.trim() === "") {
      // Emptying the draft discards the translation with it.
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
    const prefs = { ...get().prefs, ...patch };
    set({ prefs });
    writeJson(PREFS_KEY, prefs);
  },

  setTarget: (target) => {
    const { prefs } = get();
    if (target === prefs.target) return;
    get().setPrefs({
      target,
      recentTargets: [
        prefs.target,
        ...prefs.recentTargets.filter((code) => code !== target),
      ]
        .filter((code, index, all) => all.indexOf(code) === index)
        .slice(0, RECENT_TARGETS),
    });
  },

  swap: () => {
    const { prefs, output, phase } = get();
    if (prefs.from === AUTO_SOURCE || phase) return;
    get().setPrefs({ from: prefs.target });
    get().setTarget(prefs.from);
    if (output) {
      set({
        source: output.slice(0, TRANSLATE_SOURCE_MAX_CHARS),
        output: "",
        error: null,
        result: null,
        attempt: null,
      });
    }
  },

  translate: async () => {
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
    try {
      const { output, model } = await runTranslation({
        text: source,
        prompt,
        modelId: prefs.modelId,
        signal: run.signal,
        onPhase: (phase) => current() && set({ phase }),
        onOutput: (text) => current() && set({ output: text }),
      });
      if (!current()) return;
      controller = null;
      const state = get();
      // Onto the stored list, not this window's: another window may have added entries since.
      const history = state.prefs.saveHistory
        ? saveHistory(
            addHistoryEntry(storedHistory(), {
              source,
              output,
              target: prefs.target,
              from: prefs.from === AUTO_SOURCE ? null : prefs.from,
              model,
            }),
          )
        : state.history;
      set({
        phase: null,
        output,
        result: request,
        history,
        readyModels: state.readyModels.includes(prefs.modelId)
          ? state.readyModels
          : [...state.readyModels, prefs.modelId],
      });
    } catch (error) {
      if (!current()) return;
      controller = null;
      if (
        run.signal.aborted ||
        (error instanceof DOMException && error.name === "AbortError")
      ) {
        // Whatever arrived before Stop stays readable, marked as partial by the missing result.
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
    get().setPrefs({
      from:
        entry.from && findTranslateLanguage(entry.from)
          ? entry.from
          : AUTO_SOURCE,
    });
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

/** True when the output pane no longer matches the draft and settings. */
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
      result.modelId !== prefs.modelId)
  );
}

// Another window's translations and deletions show here without a reload.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === HISTORY_KEY || event.key === null) {
      useTranslateStore.setState({ history: storedHistory() });
    }
  });
}
