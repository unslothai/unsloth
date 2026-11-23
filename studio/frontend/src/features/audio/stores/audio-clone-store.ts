// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type AudioSourceSelection,
  REFERENCE_MAX_SECONDS,
} from "../audio-run-request";

const AUDIO_CLONE_STORAGE_KEY = "unsloth_audio_clone_v1";

// clone sends the reference's first 30 s, so a longer clip's full text would not match it.
export function referenceTranscript(next: AudioSourceSelection | null): string {
  if (!next || (next.durationS ?? 0) > REFERENCE_MAX_SECONDS) return "";
  return next.transcript || "";
}

/** also holds every page's tool values, not just clone's. */
interface AudioCloneState {
  reference: AudioSourceSelection | null;
  referenceText: string;
  text: string;
  language: string;
  toolValues: Record<string, unknown>;
  setReference: (reference: AudioSourceSelection | null) => void;
  adoptReference: (next: AudioSourceSelection | null) => void;
  setReferenceText: (referenceText: string) => void;
  applyTranscript: (source: AudioSourceSelection, text: string) => void;
  setText: (text: string) => void;
  setLanguage: (language: string) => void;
  setToolValue: (key: string, value: unknown) => void;
}

const MAX_TOOL_VALUES = 200;

export const useAudioCloneStore = create<AudioCloneState>()(
  persist(
    (set) => ({
      reference: null,
      referenceText: "",
      text: "",
      language: "",
      toolValues: {},
      setReference: (reference) => set({ reference }),
      // Typed text survives a new pick; text that came from the old clip goes with it.
      adoptReference: (next) =>
        set((state) => {
          const typed =
            state.referenceText.trim() !== "" &&
            state.referenceText !== (state.reference?.transcript ?? "");
          return {
            reference: next,
            referenceText: typed ? state.referenceText : referenceTranscript(next),
            language: (!state.language && next?.language) || state.language,
          };
        }),
      setReferenceText: (referenceText) => set({ referenceText }),
      // Kept on the source so a new pick replaces it; results for unpicked clips are dropped.
      applyTranscript: (source, text) =>
        set((state) =>
          state.reference?.kind === source.kind &&
          state.reference.id === source.id
            ? {
                reference: { ...state.reference, transcript: text },
                referenceText: text,
              }
            : {},
        ),
      setText: (text) => set({ text }),
      setLanguage: (language) => set({ language }),
      setToolValue: (key, value) =>
        set((state) => {
          const entries = Object.entries(state.toolValues).filter(
            ([existing]) => existing !== key,
          );
          const kept =
            entries.length >= MAX_TOOL_VALUES
              ? entries.slice(entries.length - MAX_TOOL_VALUES + 1)
              : entries;
          return { toolValues: { ...Object.fromEntries(kept), [key]: value } };
        }),
    }),
    {
      name: AUDIO_CLONE_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        reference: state.reference,
        referenceText: state.referenceText,
        text: state.text,
        language: state.language,
        toolValues: state.toolValues,
      }),
    },
  ),
);
