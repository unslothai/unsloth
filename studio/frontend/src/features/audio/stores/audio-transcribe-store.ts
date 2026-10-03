// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";

export const AUDIO_TRANSCRIBE_STORAGE_KEY = "unsloth_audio_transcribe_v1";

export type TranscriptView = "text" | "segments";

interface AudioTranscribeState {
  source: AudioSourceSelection | null;
  /** Empty means Auto. */
  language: string;
  timestamps: boolean;
  speakers: boolean;
  view: TranscriptView;
  setSource: (source: AudioSourceSelection | null) => void;
  setLanguage: (language: string) => void;
  setTimestamps: (timestamps: boolean) => void;
  setSpeakers: (speakers: boolean) => void;
  setView: (view: TranscriptView) => void;
}

export const useAudioTranscribeStore = create<AudioTranscribeState>()(
  persist(
    (set) => ({
      source: null,
      language: "",
      timestamps: false,
      speakers: true,
      view: "text",
      setSource: (source) => set({ source }),
      setLanguage: (language) => set({ language }),
      setTimestamps: (timestamps) => set({ timestamps }),
      setSpeakers: (speakers) => set({ speakers }),
      setView: (view) => set({ view }),
    }),
    {
      name: AUDIO_TRANSCRIBE_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        source: state.source,
        language: state.language,
        timestamps: state.timestamps,
        speakers: state.speakers,
        view: state.view,
      }),
    },
  ),
);
