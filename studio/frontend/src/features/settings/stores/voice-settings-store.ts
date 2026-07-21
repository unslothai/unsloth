// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

// Voice preferences in localStorage. Adapters read them at call time so
// changes apply without reloading the chat runtime.

export interface RecentDictation {
  id: string;
  text: string;
  at: number;
  /** Chat the dictation was spoken into, when one was open at the time. */
  chatId?: string;
}

// Dictation history is kept in full; the list view paginates instead of the
// store discarding entries. QUOTA_TRIM_KEEP is the emergency floor if
// localStorage itself runs out of room (see the persist storage wrapper).
const QUOTA_TRIM_KEEP = 200;
// Cap stored transcript length so a few long dictations cannot bloat the
// persisted blob and trip a synchronous localStorage quota error on save.
const MAX_RECENT_DICTATION_LENGTH = 2000;
const MAX_DICTIONARY_ENTRIES = 100;
const MAX_DICTIONARY_ENTRY_LENGTH = 120;

/** Five curated Whisper choices, mirrored by the backend (stt_sidecar.py). */
export const STT_MODELS = [
  "tiny",
  "base",
  "small",
  "large-v3-turbo",
  "large-v3",
] as const;
export type DefaultSttModel = (typeof STT_MODELS)[number];
/** A curated id or a user-selected Hugging Face `owner/model` repository. */
export type SttModel = string;
/** Whisper repos downloaded through Studio's existing Model Hub manager. */
export const STT_MODEL_REPOS: Record<DefaultSttModel, string> = {
  tiny: "unsloth/whisper-tiny",
  base: "unsloth/whisper-base",
  small: "unsloth/whisper-small",
  "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
  "large-v3": "unsloth/whisper-large-v3",
};
export const DEFAULT_STT_MODEL: DefaultSttModel = "small";
const HF_REPO_ID =
  /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

export function isSttModelId(value: string): boolean {
  const normalized = value.trim();
  return (
    (STT_MODELS as readonly string[]).includes(normalized) ||
    HF_REPO_ID.test(normalized)
  );
}

export function normalizeSttModel(value: unknown): SttModel {
  if (typeof value !== "string") {
    return DEFAULT_STT_MODEL;
  }
  const normalized = value.trim();
  return isSttModelId(normalized) ? normalized : DEFAULT_STT_MODEL;
}

export function getSttModelRepo(model: SttModel): string {
  return STT_MODEL_REPOS[model as DefaultSttModel] ?? normalizeSttModel(model);
}

// All curated models are multilingual. Custom `.en` Whisper checkpoints are
// treated as English-only so a later language change falls back safely.
export const ENGLISH_ONLY_STT_MODELS: ReadonlySet<SttModel> = new Set([]);

/** Whether a model can honor the selected dictation language. */
export function isSttModelLanguageCompatible(
  model: SttModel,
  language: string,
): boolean {
  const isEnglishOnly =
    ENGLISH_ONLY_STT_MODELS.has(model) ||
    getSttModelRepo(model).toLowerCase().endsWith(".en");
  if (!isEnglishOnly) {
    return true;
  }
  const normalized = language.trim().replaceAll("_", "-").toLowerCase();
  // Auto sends no forced language, which English-only checkpoints accept.
  return normalized === "auto" || normalized.split("-", 1)[0] === "en";
}

export type DictationEngine = "browser" | "model";

/**
 * Whether a model id is one of the five curated Whisper choices. Curated
 * models run as GGML checkpoints through whisper.cpp; custom repositories
 * are safetensors checkpoints and run through Transformers.
 */
export function isCuratedSttModel(model: SttModel): boolean {
  return (STT_MODELS as readonly string[]).includes(model.trim());
}

export interface VoiceSettingsState {
  /** Input device for dictation. "default" = system default microphone. */
  micDeviceId: string;
  setMicDeviceId: (value: string) => void;

  /**
   * "browser": Web Speech API. "model": local transcription; the selected
   * model decides the backend (whisper.cpp for curated GGML checkpoints,
   * Transformers for custom safetensors repositories).
   */
  dictationEngine: DictationEngine;
  setDictationEngine: (value: DictationEngine) => void;

  /** STT model to use when dictationEngine is "model". */
  sttModel: SttModel;
  setSttModel: (value: SttModel) => void;

  /** BCP 47 tag for speech recognition, or "auto" for the browser locale. */
  dictationLanguage: string;
  setDictationLanguage: (value: string) => void;

  /** Exact spellings applied to matching transcript words and phrases. */
  dictionary: string[];
  addDictionaryEntry: (value: string) => void;
  updateDictionaryEntry: (index: number, value: string) => void;
  /** Trim the entry; drop it when it was left empty. Call on input blur. */
  commitDictionaryEntry: (index: number) => void;
  removeDictionaryEntry: (index: number) => void;

  /** Final transcripts, newest first, so text can be recovered. */
  recentDictations: RecentDictation[];
  addRecentDictation: (text: string, chatId?: string) => void;
  removeRecentDictation: (id: string) => void;
  clearRecentDictations: () => void;

  /** Show the read-aloud button on assistant responses. */
  ttsEnabled: boolean;
  setTtsEnabled: (value: boolean) => void;

  /** "system": speechSynthesis voices. "studio": the loaded TTS audio model. */
  ttsEngine: "system" | "studio";
  setTtsEngine: (value: "system" | "studio") => void;

  /** speechSynthesis voiceURI, or "default" for the system voice. */
  ttsVoiceURI: string;
  setTtsVoiceURI: (value: string) => void;

  ttsRate: number;
  setTtsRate: (value: number) => void;
  ttsPitch: number;
  setTtsPitch: (value: number) => void;
  ttsVolume: number;
  setTtsVolume: (value: number) => void;
}

/**
 * localStorage wrapper that keeps the full dictation history until the
 * browser's quota is actually hit, then drops the oldest entries instead of
 * throwing away the whole save.
 */
const quotaSafeLocalStorage = {
  getItem: (key: string) => localStorage.getItem(key),
  removeItem: (key: string) => localStorage.removeItem(key),
  setItem: (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
      return;
    } catch {
      // Quota exceeded: trim dictation history, oldest first, and retry.
    }
    for (const keep of [QUOTA_TRIM_KEEP, 20]) {
      try {
        const parsed = JSON.parse(value) as {
          state?: { recentDictations?: RecentDictation[] };
        };
        const state = parsed.state;
        const recents = state?.recentDictations;
        if (!state || !Array.isArray(recents) || recents.length <= keep) {
          continue;
        }
        state.recentDictations = recents.slice(0, keep);
        localStorage.setItem(key, JSON.stringify(parsed));
        return;
      } catch {
        // Fall through to the next, more aggressive trim.
      }
    }
  },
};

export const useVoiceSettingsStore = create<VoiceSettingsState>()(
  persist(
    (set) => ({
      micDeviceId: "default",
      setMicDeviceId: (micDeviceId) => set({ micDeviceId }),

      dictationEngine: "browser",
      setDictationEngine: (dictationEngine) => set({ dictationEngine }),

      sttModel: DEFAULT_STT_MODEL,
      setSttModel: (value) =>
        set((state) => {
          const sttModel = normalizeSttModel(value);
          return {
            sttModel: isSttModelLanguageCompatible(
              sttModel,
              state.dictationLanguage,
            )
              ? sttModel
              : DEFAULT_STT_MODEL,
          };
        }),

      dictationLanguage: "auto",
      setDictationLanguage: (dictationLanguage) =>
        set((state) => ({
          dictationLanguage,
          sttModel: isSttModelLanguageCompatible(
            state.sttModel,
            dictationLanguage,
          )
            ? state.sttModel
            : DEFAULT_STT_MODEL,
        })),

      dictionary: [],
      addDictionaryEntry: (value) =>
        set((state) => {
          const trimmed = value.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
          if (!trimmed) return state;
          if (state.dictionary.length >= MAX_DICTIONARY_ENTRIES) return state;
          if (
            state.dictionary.some(
              (entry) => entry.toLowerCase() === trimmed.toLowerCase(),
            )
          ) {
            return state;
          }
          return { dictionary: [...state.dictionary, trimmed] };
        }),
      // Keep the raw value so the input edits freely; commitDictionaryEntry finalizes on blur.
      updateDictionaryEntry: (index, value) =>
        set((state) => {
          const dictionary = [...state.dictionary];
          if (index < 0 || index >= dictionary.length) return state;
          dictionary[index] = value.slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
          return { dictionary };
        }),
      commitDictionaryEntry: (index) =>
        set((state) => {
          const dictionary = [...state.dictionary];
          if (index < 0 || index >= dictionary.length) return state;
          const trimmed = dictionary[index]?.trim() ?? "";
          if (trimmed) {
            dictionary[index] = trimmed;
          } else {
            dictionary.splice(index, 1);
          }
          return { dictionary };
        }),
      removeDictionaryEntry: (index) =>
        set((state) => ({
          dictionary: state.dictionary.filter((_, i) => i !== index),
        })),

      recentDictations: [],
      addRecentDictation: (text, chatId) =>
        set((state) => {
          const trimmed = text.trim().slice(0, MAX_RECENT_DICTATION_LENGTH);
          if (!trimmed) {
            return state;
          }
          const at = Date.now();
          return {
            recentDictations: [
              {
                id: `${at}-${Math.random().toString(36).slice(2, 10)}`,
                text: trimmed,
                at,
                ...(chatId ? { chatId } : {}),
              },
              ...state.recentDictations,
            ],
          };
        }),
      removeRecentDictation: (id) =>
        set((state) => ({
          recentDictations: state.recentDictations.filter(
            (dictation) => dictation.id !== id,
          ),
        })),
      clearRecentDictations: () => set({ recentDictations: [] }),

      ttsEnabled: true,
      setTtsEnabled: (ttsEnabled) => set({ ttsEnabled }),

      ttsEngine: "system",
      setTtsEngine: (ttsEngine) => set({ ttsEngine }),

      ttsVoiceURI: "default",
      setTtsVoiceURI: (ttsVoiceURI) => set({ ttsVoiceURI }),

      ttsRate: 1,
      setTtsRate: (ttsRate) => set({ ttsRate }),
      ttsPitch: 1,
      setTtsPitch: (ttsPitch) => set({ ttsPitch }),
      ttsVolume: 1,
      setTtsVolume: (ttsVolume) => set({ ttsVolume }),
    }),
    {
      name: "unsloth_voice_settings",
      storage: createJSONStorage(() => quotaSafeLocalStorage),
      merge: (persisted, current) => {
        const saved = persisted as Partial<VoiceSettingsState> | undefined;
        const dictationLanguage = asString(saved?.dictationLanguage, "auto");
        // "gguf" was a short-lived separate engine choice; both local
        // backends now live under "model".
        const savedEngine = saved?.dictationEngine as string | undefined;
        const dictationEngine: DictationEngine =
          savedEngine === "model" || savedEngine === "gguf"
            ? "model"
            : "browser";
        const savedSttModel = normalizeSttModel(saved?.sttModel);
        const sttModel = isSttModelLanguageCompatible(
          savedSttModel,
          dictationLanguage,
        )
          ? savedSttModel
          : DEFAULT_STT_MODEL;
        return {
          ...current,
          micDeviceId: asString(saved?.micDeviceId, "default"),
          dictationEngine,
          sttModel,
          dictationLanguage,
          dictionary: Array.isArray(saved?.dictionary)
            ? saved.dictionary
                .filter((v): v is string => typeof v === "string" && !!v.trim())
                .map((v) => v.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH))
                .slice(0, MAX_DICTIONARY_ENTRIES)
            : [],
          recentDictations: normalizeRecentDictations(saved?.recentDictations),
          ttsEnabled:
            typeof saved?.ttsEnabled === "boolean" ? saved.ttsEnabled : true,
          ttsEngine: saved?.ttsEngine === "studio" ? "studio" : "system",
          ttsVoiceURI: asString(saved?.ttsVoiceURI, "default"),
          ttsRate: clampNumber(saved?.ttsRate, 0.5, 2, 1),
          ttsPitch: clampNumber(saved?.ttsPitch, 0, 2, 1),
          ttsVolume: clampNumber(saved?.ttsVolume, 0, 1, 1),
        };
      },
    },
  ),
);

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function normalizeRecentDictations(value: unknown): RecentDictation[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const normalized: RecentDictation[] = [];
  for (const [index, entry] of value.entries()) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const candidate = entry as Partial<RecentDictation>;
    if (
      typeof candidate.text !== "string" ||
      !candidate.text.trim() ||
      typeof candidate.at !== "number" ||
      !Number.isFinite(candidate.at)
    ) {
      continue;
    }
    normalized.push({
      id:
        typeof candidate.id === "string" && candidate.id
          ? candidate.id
          : `legacy-${candidate.at}-${index}`,
      text: candidate.text.trim().slice(0, MAX_RECENT_DICTATION_LENGTH),
      at: candidate.at,
      ...(typeof candidate.chatId === "string" && candidate.chatId
        ? { chatId: candidate.chatId }
        : {}),
    });
  }
  return normalized;
}

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

/** Resolve the "auto" language setting to a concrete BCP 47 tag. */
export function resolveDictationLanguage(setting?: string): string {
  const value = setting ?? useVoiceSettingsStore.getState().dictationLanguage;
  if (value && value !== "auto") return value;
  return typeof navigator !== "undefined" && navigator.language
    ? navigator.language
    : "en-US";
}

// Whisper's language codes: transformers `LANGUAGES` keys mirrored here the way
// STT_MODELS mirrors the backend, plus the backend's BCP-47 aliases
// (stt_sidecar.py `_WHISPER_LANGUAGE_ALIASES`). Keep in sync with
// `_known_whisper_languages()` in the backend; Auto only resolves to a language
// in this set, so a UI locale Whisper cannot honor stays on auto-detect.
const WHISPER_DICTATION_LANGUAGES = new Set([
  "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
  "ca", "cmn", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa",
  "fi", "fil", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht",
  "hu", "hy", "id", "in", "is", "it", "iw", "ja", "ji", "jw", "ka", "kk",
  "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk",
  "ml", "mn", "mr", "ms", "mt", "my", "nb", "ne", "nl", "nn", "no", "oc",
  "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
  "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
  "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
]);

/**
 * Resolve Auto for the model STT engine (the browser engine already resolves it
 * via `resolveDictationLanguage`). Only the literal "auto" is resolved to a
 * concrete locale; an explicit language (or an empty/malformed setting) passes
 * through unchanged. Resolution is further gated so Auto only becomes a language
 * the model AND Whisper can honor; otherwise it stays auto-detect rather than
 * forcing a locale Whisper cannot handle (e.g. Irish) or 422ing every dictation.
 */
export function resolveModelDictationLanguage(
  model: SttModel,
  requested: string,
): string {
  if (requested !== "auto") return requested;
  const resolved = resolveDictationLanguage(requested);
  const primary = resolved
    .trim()
    .replaceAll("_", "-")
    .toLowerCase()
    .split("-", 1)[0];
  return isSttModelLanguageCompatible(model, resolved) &&
    WHISPER_DICTATION_LANGUAGES.has(primary)
    ? resolved
    : requested;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Rewrite dictionary phrases in a transcript to their exact stored form,
 * matching case-insensitively on word boundaries ("jane doe" -> "Jane Doe").
 */
export function applyDictationDictionary(
  transcript: string,
  dictionary?: string[],
): string {
  const entries = dictionary ?? useVoiceSettingsStore.getState().dictionary;
  if (!transcript || entries.length === 0) return transcript;
  let result = transcript;
  for (const entry of entries) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    // Whitespace-tolerant pattern so "jane   doe" still matches.
    const pattern = trimmed.split(/\s+/).map(escapeRegExp).join("\\s+");
    try {
      // Capture the leading boundary instead of using a lookbehind, which
      // engines that support dictation but not lookbehind (Safari < 16.4)
      // cannot compile; the catch below would otherwise skip every entry.
      const regex = new RegExp(
        `(^|[^\\p{L}\\p{N}])(${pattern})(?![\\p{L}\\p{N}])`,
        "giu",
      );
      // Re-emit the boundary; callback form avoids $-pattern expansion.
      result = result.replace(regex, (_match, prefix) => `${prefix}${trimmed}`);
    } catch {
      // Skip entries that produce an invalid pattern.
    }
  }
  return result;
}

/** Record a finished dictation so it can be recovered from settings. */
export function recordRecentDictation(text: string, chatId?: string): void {
  useVoiceSettingsStore.getState().addRecentDictation(text, chatId);
}
