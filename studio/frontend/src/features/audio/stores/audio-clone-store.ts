// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";

const AUDIO_CLONE_STORAGE_KEY = "unsloth_audio_clone_v1";

/** Also holds every page's tool values, not just Clone's. */
interface AudioCloneState {
  reference: AudioSourceSelection | null;
  referenceText: string;
  text: string;
  language: string;
  toolValues: Record<string, unknown>;
  setReference: (reference: AudioSourceSelection | null) => void;
  setReferenceText: (referenceText: string) => void;
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
      setReferenceText: (referenceText) => set({ referenceText }),
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
