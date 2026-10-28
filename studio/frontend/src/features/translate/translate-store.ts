// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type TranslateLanguageCode,
  type TranslateMode,
  type TranslateSourceLanguage,
  type TranslateTone,
  type TranslationEntry,
  isTranslateLanguage,
  isTranslateMode,
  isTranslateSource,
  isTranslateTone,
} from "./translate-prompt";

export const TRANSLATE_STORAGE_KEY = "unsloth_translate";
/** Keeps the persisted record well under localStorage quotas; the oldest entries go first. */
export const MAX_TRANSLATE_HISTORY = 200;

export interface TranslateState {
  source: TranslateSourceLanguage;
  target: TranslateLanguageCode;
  mode: TranslateMode;
  tone: TranslateTone;
  showSystemPrompt: boolean;
  /** Newest first. */
  history: TranslationEntry[];
  setSource: (source: TranslateSourceLanguage) => void;
  setTarget: (target: TranslateLanguageCode) => void;
  setMode: (mode: TranslateMode) => void;
  setTone: (tone: TranslateTone) => void;
  setShowSystemPrompt: (show: boolean) => void;
  addEntry: (entry: TranslationEntry) => void;
  removeEntry: (id: string) => void;
  clearHistory: () => void;
}

function isEntry(value: unknown): value is TranslationEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.id === "string" &&
    typeof entry.createdAt === "number" &&
    typeof entry.input === "string" &&
    typeof entry.output === "string" &&
    typeof entry.model === "string" &&
    isTranslateSource(entry.source) &&
    isTranslateLanguage(entry.target) &&
    isTranslateMode(entry.mode) &&
    isTranslateTone(entry.tone)
  );
}

// Whitelist: localStorage is untyped and a bad record would crash the page.
function sanitize(persisted: unknown, current: TranslateState): TranslateState {
  const raw = (
    persisted && typeof persisted === "object" ? persisted : {}
  ) as Record<string, unknown>;
  return {
    ...current,
    source: isTranslateSource(raw.source) ? raw.source : current.source,
    target: isTranslateLanguage(raw.target) ? raw.target : current.target,
    mode: isTranslateMode(raw.mode) ? raw.mode : current.mode,
    tone: isTranslateTone(raw.tone) ? raw.tone : current.tone,
    showSystemPrompt:
      typeof raw.showSystemPrompt === "boolean"
        ? raw.showSystemPrompt
        : current.showSystemPrompt,
    history: Array.isArray(raw.history)
      ? raw.history.filter(isEntry).slice(0, MAX_TRANSLATE_HISTORY)
      : current.history,
  };
}

export const useTranslateStore = create<TranslateState>()(
  persist(
    (set) => ({
      source: "auto",
      target: "en",
      mode: "automatic",
      tone: "balanced",
      showSystemPrompt: false,
      history: [],
      setSource: (source) => set({ source }),
      setTarget: (target) => set({ target }),
      setMode: (mode) => set({ mode }),
      setTone: (tone) => set({ tone }),
      setShowSystemPrompt: (showSystemPrompt) => set({ showSystemPrompt }),
      addEntry: (entry) =>
        set((state) => ({
          history: [entry, ...state.history].slice(0, MAX_TRANSLATE_HISTORY),
        })),
      removeEntry: (id) =>
        set((state) => ({
          history: state.history.filter((entry) => entry.id !== id),
        })),
      clearHistory: () => set({ history: [] }),
    }),
    {
      name: TRANSLATE_STORAGE_KEY,
      // pinned so later shape changes can migrate from this version.
      version: 1,
      // drop newer versions that may reuse these field names.
      migrate: (persisted, version) =>
        (version <= 1 ? persisted : {}) as TranslateState,
      merge: (persisted, current) => sanitize(persisted, current),
    },
  ),
);
