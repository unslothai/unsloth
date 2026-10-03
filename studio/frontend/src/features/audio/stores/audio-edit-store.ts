// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";

export const AUDIO_EDIT_STORAGE_KEY = "unsloth_audio_edit_v1";

export type AudioEditMode = "words" | "delivery";

export interface AudioEditDelivery {
  /** Playback speed, 1 = unchanged. */
  speed: number;
  /** Steps to raise the pitch, 0 = unchanged. */
  pitchSteps: number;
}

export const DEFAULT_EDIT_DELIVERY: AudioEditDelivery = {
  speed: 1,
  pitchSteps: 0,
};

/** Edit's draft, kept across page switches and reloads. Tool values live in the Clone store with every page's. */
interface AudioEditState {
  source: AudioSourceSelection | null;
  /** ① What the recording says. */
  transcript: string;
  /** The source id the transcript was taken from, so a new recording gets transcribed again. */
  transcriptFor: string | null;
  /** ② The transcript with the user's changes. */
  edited: string;
  /** Whether the user typed in ②; until then ② follows ①. */
  editedTouched: boolean;
  mode: AudioEditMode;
  delivery: AudioEditDelivery;
  setSource: (source: AudioSourceSelection | null) => void;
  setTranscript: (transcript: string, transcriptFor?: string | null) => void;
  setEdited: (edited: string) => void;
  resetEdited: () => void;
  setMode: (mode: AudioEditMode) => void;
  setDelivery: (delivery: Partial<AudioEditDelivery>) => void;
}

export const useAudioEditStore = create<AudioEditState>()(
  persist(
    (set) => ({
      source: null,
      transcript: "",
      transcriptFor: null,
      edited: "",
      editedTouched: false,
      mode: "words",
      delivery: DEFAULT_EDIT_DELIVERY,
      setSource: (source) => set({ source }),
      setTranscript: (transcript, transcriptFor) =>
        set((state) => ({
          transcript,
          ...(transcriptFor !== undefined ? { transcriptFor } : {}),
          ...(state.editedTouched ? {} : { edited: transcript }),
        })),
      setEdited: (edited) => set({ edited, editedTouched: true }),
      resetEdited: () =>
        set((state) => ({ edited: state.transcript, editedTouched: false })),
      setMode: (mode) => set({ mode }),
      setDelivery: (delivery) =>
        set((state) => ({ delivery: { ...state.delivery, ...delivery } })),
    }),
    {
      name: AUDIO_EDIT_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        source: state.source,
        transcript: state.transcript,
        transcriptFor: state.transcriptFor,
        edited: state.edited,
        editedTouched: state.editedTouched,
        mode: state.mode,
        delivery: state.delivery,
      }),
    },
  ),
);
