// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export type VoiceOrbState =
  | "waiting"
  | "listening"
  | "transcribing"
  | "thinking"
  | "generating"
  | "synthesizing"
  | "speaking"
  | "hearing"
  | "loading"
  | "paused";

export type VoiceOrbInputs = {
  voiceMode: "off" | "configuring" | "active";
  hasChatModel: boolean;
  voiceSlotLoading: boolean;
  voiceHearing: boolean;
  isPlaying: boolean;
  voiceTranscribing: boolean;
  isThreadRunning: boolean;
  isReasoning: boolean;
  isSpeaking: boolean;
  voicePaused?: boolean;
};

export type VoiceOrbDecision =
  | { kind: "set"; state: VoiceOrbState | null }
  | { kind: "hold" };

const set = (state: VoiceOrbState | null): VoiceOrbDecision => ({
  kind: "set",
  state,
});

export function deriveOrbState(inputs: VoiceOrbInputs): VoiceOrbDecision {
  if (inputs.voiceMode !== "active") return set(null);
  if (!inputs.hasChatModel) return set("waiting");
  if (inputs.voicePaused) return set("paused");
  if (inputs.voiceSlotLoading) return set("loading");
  if (inputs.voiceHearing) return set("hearing");
  if (inputs.isPlaying) return set("speaking");
  if (inputs.voiceTranscribing) return set("transcribing");
  if (inputs.isThreadRunning && inputs.isReasoning) return set("thinking");
  if (inputs.isThreadRunning) return set("generating");
  if (inputs.isSpeaking) return { kind: "hold" };
  return set("listening");
}
