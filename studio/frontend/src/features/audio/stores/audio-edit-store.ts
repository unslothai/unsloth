// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";
import type { EditDelivery, EditMode } from "../edit-adapters";

export const AUDIO_EDIT_STORAGE_KEY = "unsloth_audio_edit_v1";

interface AudioEditState {
  source: AudioSourceSelection | null;
  transcript: string;
  transcriptFor: string | null;
  edited: string;
  editedTouched: boolean;
  mode: EditMode;
  delivery: EditDelivery;
  setSource: (source: AudioSourceSelection | null) => void;
  setTranscript: (transcript: string, transcriptFor?: string | null) => void;
  setEdited: (edited: string) => void;
  resetEdited: () => void;
  setMode: (mode: EditMode) => void;
  setDelivery: (delivery: Partial<EditDelivery>) => void;
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
      delivery: { speed: 1, pitchSteps: 0 },
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
