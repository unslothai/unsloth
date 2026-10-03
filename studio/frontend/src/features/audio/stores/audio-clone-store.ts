// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";

export const AUDIO_CLONE_STORAGE_KEY = "unsloth_audio_clone_v1";

/** Clone's draft, and every page's model-tool values, kept across page switches and reloads. */
interface AudioCloneState {
  reference: AudioSourceSelection | null;
  referenceText: string;
  text: string;
  /** Empty means Auto. */
  language: string;
  /** Tool panel values by `${model}:${workflow}:${panelId}` (see toolValueKey). */
  toolValues: Record<string, unknown>;
  setReference: (reference: AudioSourceSelection | null) => void;
  setReferenceText: (referenceText: string) => void;
  setText: (text: string) => void;
  setLanguage: (language: string) => void;
  setToolValue: (key: string, value: unknown) => void;
}

/** Tool values kept at most; the oldest go first so a long history of models cannot grow storage forever. */
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
          const { [key]: _previous, ...rest } = state.toolValues;
          const entries = Object.entries(rest);
          const kept =
            entries.length >= MAX_TOOL_VALUES
              ? entries.slice(entries.length - MAX_TOOL_VALUES + 1)
              : entries;
          // Re-inserted last, so insertion order is recency.
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
