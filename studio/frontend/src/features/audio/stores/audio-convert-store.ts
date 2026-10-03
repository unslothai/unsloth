// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection, ConvertMode } from "../audio-run-request";

export const AUDIO_CONVERT_STORAGE_KEY = "unsloth_audio_convert_v1";

type ConvertCompareSide = "source" | "converted";

/** Convert's draft, kept across page switches and reloads. Tool panel values live in
 *  useAudioCloneStore.toolValues under toolValueKey(model, "convert", panelId). */
interface AudioConvertState {
  /** The recording to convert. */
  source: AudioSourceSelection | null;
  /** The voice to convert it to, for models that take a recording. */
  target: AudioSourceSelection | null;
  /** The packaged voice, for models that convert to built-in voices (RVC). */
  builtinVoice: string;
  mode: ConvertMode;
  pitchAuto: boolean;
  /** Semitones, -12..12. */
  pitch: number;
  /** What's said in the recording, for Take target style. */
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
  Number.isFinite(pitch) ? Math.min(12, Math.max(-12, Math.round(pitch))) : 0;

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
      setSource: (source) => set({ source }),
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
