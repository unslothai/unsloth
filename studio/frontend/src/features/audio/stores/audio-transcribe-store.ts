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
}

export const useAudioTranscribeStore = create<AudioTranscribeState>()(
  persist(
    (): AudioTranscribeState => ({
      source: null,
      language: "",
      timestamps: false,
      speakers: true,
      view: "text",
    }),
    { name: AUDIO_TRANSCRIBE_STORAGE_KEY, version: 1 },
  ),
);
