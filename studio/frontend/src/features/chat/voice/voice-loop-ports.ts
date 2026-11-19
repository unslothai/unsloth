// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SttModel, VoiceTurnEnding } from "../../settings/stores/stt-model-catalog.ts";
import type { MicTarget } from "../../settings/stores/voice-settings-store.ts";
import type { VoiceOrbState } from "./orb-state.ts";

export interface VoiceLoopRuntime {
  selectedSttModelId: string;
  selectedVadModelId: string;
  sttSlotDownload: "needed" | "downloading" | null;
  voiceOrbState: VoiceOrbState | null;
  models: readonly { id: string; hasAudioInput?: boolean }[];
  params: { checkpoint: string };
  selectedVoiceModelId: string;
  modelLoading: boolean;
}

export interface VoiceLoopSettings {
  micDeviceId: string;
  micTarget: MicTarget;
  sttModel: SttModel;
  dictationLanguage: string;
  turnEnding: VoiceTurnEnding;
  turnJudgeBar: number;
}

export interface VoiceLoopPorts {
  runtime(): VoiceLoopRuntime;
  settings(): VoiceLoopSettings;
  onMicDeviceChange(listener: (micDeviceId: string) => void): () => void;
  setHearing(hearing: boolean): void;
  setTranscribing(transcribing: boolean): void;
  setHeldForEdit(held: boolean): void;
  setSttSlotLoaded(loaded: boolean): void;
  setSttSlotDownload(state: "needed" | "downloading" | null): void;
  setVadSlotLoaded(loaded: boolean): void;
  // Not persisted: an install that gains the detector later gets the pick back.
  useVadFallback(id: string): void;
}
