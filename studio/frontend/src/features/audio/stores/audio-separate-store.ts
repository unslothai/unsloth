// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";

export const AUDIO_SEPARATE_STORAGE_KEY = "unsloth_audio_separate_v1";

/** Separate's draft, kept across page switches and reloads. */
interface AudioSeparateState {
  source: AudioSourceSelection | null;
  /** The overlap each model last ran with, so a change is announced as a reload. */
  lastOverlapByModel: Record<string, boolean>;
  setSource: (source: AudioSourceSelection | null) => void;
  setLastOverlap: (model: string, overlap: boolean) => void;
}

/** Models remembered at most; the oldest go first. */
const MAX_MODELS = 20;

export const useAudioSeparateStore = create<AudioSeparateState>()(
  persist(
    (set) => ({
      source: null,
      lastOverlapByModel: {},
      setSource: (source) => set({ source }),
      setLastOverlap: (model, overlap) =>
        set((state) => {
          const { [model]: _previous, ...rest } = state.lastOverlapByModel;
          const entries = Object.entries(rest);
          const kept = entries.slice(
            Math.max(0, entries.length - MAX_MODELS + 1),
          );
          return {
            lastOverlapByModel: {
              ...Object.fromEntries(kept),
              [model]: overlap,
            },
          };
        }),
    }),
    {
      name: AUDIO_SEPARATE_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        source: state.source,
        lastOverlapByModel: state.lastOverlapByModel,
      }),
    },
  ),
);
