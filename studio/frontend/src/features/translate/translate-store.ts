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
  dedicatedTranslateModel,
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
import { TRANSLATE_TONES, type TranslateTone } from "./translate-text";

const PREFS_KEY = "unsloth_translate_prefs";
const HISTORY_KEY = "unsloth_translate_history";
const RECENT_TARGETS = 3;
export const TRANSLATE_SOURCE_MAX_CHARS = 50_000;

export type TranslatePrefs = {
  target: string;
  recentTargets: string[];
  modelId: TranslateModelId;
  tone: TranslateTone;
  autoTranslate: boolean;
  saveHistory: boolean;
};

export type TranslateError =
  | { code: TranslateRunErrorCode; message?: undefined }
  | { code: "other"; message: string };

/** What the output pane holds was made from; a different draft marks it stale. */
type Result = {
  source: string;
  target: string;
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
    tone: TRANSLATE_TONES.includes(saved.tone as TranslateTone)
      ? (saved.tone as TranslateTone)
      : "default",
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

let controller: AbortController | null = null;

export const useTranslateStore = create<TranslateState>((set, get) => ({
  prefs: loadPrefs(),
  history: parseHistory(readJson(HISTORY_KEY)),
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

  translate: async () => {
    const { source, prefs } = get();
    const language = findTranslateLanguage(prefs.target);
    if (!language || source.trim() === "") return;
    controller?.abort();
    const run = new AbortController();
    controller = run;
    const current = () => controller === run;
    const request: Result = {
      source,
      target: prefs.target,
      tone: prefs.tone,
      modelId: prefs.modelId,
    };
    set({
      phase: { kind: "preparing" },
      error: null,
      output: "",
      result: null,
      attempt: request,
    });
    try {
      const output = await runTranslation({
        text: source,
        targetName: language.name,
        tone: prefs.tone,
        modelId: prefs.modelId,
        signal: run.signal,
        onPhase: (phase) => current() && set({ phase }),
        onOutput: (text) => current() && set({ output: text }),
      });
      if (!current()) return;
      controller = null;
      const state = get();
      const history = state.prefs.saveHistory
        ? saveHistory(
            addHistoryEntry(state.history, {
              source,
              output,
              target: prefs.target,
              model: dedicatedTranslateModel(prefs.modelId)?.label ?? null,
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
    const { prefs } = get();
    if (findTranslateLanguage(entry.target)) get().setTarget(entry.target);
    const shown: Result = {
      source: entry.source,
      target: entry.target,
      tone: prefs.tone,
      modelId: prefs.modelId,
    };
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
      history: saveHistory(get().history.filter((entry) => entry.id !== id)),
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
      result.target !== prefs.target ||
      result.tone !== prefs.tone ||
      result.modelId !== prefs.modelId)
  );
}
