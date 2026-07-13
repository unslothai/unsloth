// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Voice preferences in localStorage. Adapters read them at call time so
// changes apply without reloading the chat runtime.

export interface RecentDictation {
  text: string;
  at: number;
}

const MAX_RECENT_DICTATIONS = 20;
const MAX_DICTIONARY_ENTRIES = 100;
const MAX_DICTIONARY_ENTRY_LENGTH = 120;

/** STT model ids, mirrored from the backend allowlist (stt_sidecar.py). */
export const STT_MODELS = [
  "tiny",
  "base",
  "small",
  "distil-large-v3",
  "large-v3-turbo",
  "large-v3",
] as const;
export type SttModel = (typeof STT_MODELS)[number];
export const DEFAULT_STT_MODEL: SttModel = "base";

export type DictationEngine = "browser" | "model";

export interface VoiceSettingsState {
  /** Input device for dictation. "default" = system default microphone. */
  micDeviceId: string;
  setMicDeviceId: (value: string) => void;

  /** "browser": Web Speech API. "model": a loaded STT model via the backend. */
  dictationEngine: DictationEngine;
  setDictationEngine: (value: DictationEngine) => void;

  /** STT model to load when dictationEngine is "model". */
  sttModel: SttModel;
  setSttModel: (value: SttModel) => void;

  /** BCP 47 tag for speech recognition, or "auto" for the browser locale. */
  dictationLanguage: string;
  setDictationLanguage: (value: string) => void;

  /** Words or phrases dictation should recognize with this exact casing. */
  dictionary: string[];
  addDictionaryEntry: (value: string) => void;
  updateDictionaryEntry: (index: number, value: string) => void;
  /** Trim the entry; drop it when it was left empty. Call on input blur. */
  commitDictionaryEntry: (index: number) => void;
  removeDictionaryEntry: (index: number) => void;

  /** Final transcripts, newest first, so text can be recovered. */
  recentDictations: RecentDictation[];
  addRecentDictation: (text: string) => void;
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

export const useVoiceSettingsStore = create<VoiceSettingsState>()(
  persist(
    (set) => ({
      micDeviceId: "default",
      setMicDeviceId: (micDeviceId) => set({ micDeviceId }),

      dictationEngine: "browser",
      setDictationEngine: (dictationEngine) => set({ dictationEngine }),

      sttModel: DEFAULT_STT_MODEL,
      setSttModel: (sttModel) => set({ sttModel }),

      dictationLanguage: "auto",
      setDictationLanguage: (dictationLanguage) => set({ dictationLanguage }),

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
      // Keeps the raw value (including spaces and empties) so the input can
      // be edited freely; commitDictionaryEntry finalizes on blur.
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
      addRecentDictation: (text) =>
        set((state) => {
          const trimmed = text.trim();
          if (!trimmed) return state;
          return {
            recentDictations: [
              { text: trimmed, at: Date.now() },
              ...state.recentDictations,
            ].slice(0, MAX_RECENT_DICTATIONS),
          };
        }),
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
      merge: (persisted, current) => {
        const saved = persisted as Partial<VoiceSettingsState> | undefined;
        return {
          ...current,
          micDeviceId: asString(saved?.micDeviceId, "default"),
          dictationEngine:
            saved?.dictationEngine === "model" ? "model" : "browser",
          sttModel: (STT_MODELS as readonly string[]).includes(
            saved?.sttModel as string,
          )
            ? (saved?.sttModel as SttModel)
            : DEFAULT_STT_MODEL,
          dictationLanguage: asString(saved?.dictationLanguage, "auto"),
          dictionary: Array.isArray(saved?.dictionary)
            ? saved.dictionary
                .filter((v): v is string => typeof v === "string" && !!v.trim())
                .map((v) => v.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH))
                .slice(0, MAX_DICTIONARY_ENTRIES)
            : [],
          recentDictations: Array.isArray(saved?.recentDictations)
            ? saved.recentDictations.filter(
                (v): v is RecentDictation =>
                  typeof v?.text === "string" && typeof v?.at === "number",
              )
            : [],
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
      const regex = new RegExp(
        `(?<![\\p{L}\\p{N}])${pattern}(?![\\p{L}\\p{N}])`,
        "giu",
      );
      // Callback form: a plain string would expand $-patterns ($&, $$).
      result = result.replace(regex, () => trimmed);
    } catch {
      // Skip entries that produce an invalid pattern.
    }
  }
  return result;
}

/** Record a finished dictation so it can be recovered from settings. */
export function recordRecentDictation(text: string): void {
  useVoiceSettingsStore.getState().addRecentDictation(text);
}
