// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection, ConvertMode } from "../audio-run-request";

export const AUDIO_CONVERT_STORAGE_KEY = "unsloth_audio_convert_v1";

type ConvertCompareSide = "source" | "converted";

// Tool panel values live in useAudioCloneStore.toolValues, keyed by toolValueKey(model, "convert", id).
interface AudioConvertState {
  source: AudioSourceSelection | null;
  target: AudioSourceSelection | null;
  builtinVoice: string;
  mode: ConvertMode;
  pitchAuto: boolean;
  pitch: number;
  sourceText: string;
  compareSide: ConvertCompareSide;
  setSource: (source: AudioSourceSelection | null) => void;
  setTarget: (target: AudioSourceSelection | null) => void;
  setBuiltinVoice: (builtinVoice: string) => void;
  setMode: (mode: ConvertMode) => void;
  setPitchAuto: (pitchAuto: boolean) => void;
  setPitch: (pitch: number) => void;
  setSourceText: (sourceText: string) => void;
  setCompareSide: (compareSide: ConvertCompareSide) => void;
}

const clampPitch = (pitch: number) =>
  Number.isFinite(pitch) ? Math.min(24, Math.max(-24, Math.round(pitch))) : 0;

export const useAudioConvertStore = create<AudioConvertState>()(
  persist(
    (set) => ({
      source: null,
      target: null,
      builtinVoice: "default",
      mode: "speech",
      pitchAuto: true,
      pitch: 0,
      sourceText: "",
      compareSide: "converted",
      setSource: (source) =>
        set((state) =>
          state.source?.kind === source?.kind && state.source?.id === source?.id
            ? { source }
            : { source, sourceText: source?.transcript?.trim() ?? "" },
        ),
      setTarget: (target) => set({ target }),
      setBuiltinVoice: (builtinVoice) => set({ builtinVoice }),
      setMode: (mode) => set({ mode }),
      setPitchAuto: (pitchAuto) => set({ pitchAuto }),
      setPitch: (pitch) => set({ pitch: clampPitch(pitch) }),
      setSourceText: (sourceText) => set({ sourceText }),
      setCompareSide: (compareSide) => set({ compareSide }),
    }),
    {
      name: AUDIO_CONVERT_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        source: state.source,
        target: state.target,
        builtinVoice: state.builtinVoice,
        mode: state.mode,
        pitchAuto: state.pitchAuto,
        pitch: state.pitch,
        sourceText: state.sourceText,
        compareSide: state.compareSide,
      }),
    },
  ),
);
