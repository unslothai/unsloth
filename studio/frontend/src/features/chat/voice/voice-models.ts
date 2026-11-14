// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  audioCppModelFor,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog.ts";
import type { MicTarget } from "../../settings/stores/voice-settings-store.ts";
import { micSendsAudioToModel } from "../utils/mic-target.ts";

export const VOICE_DEFAULT_TTS_MODEL = `${AUDIO_CPP_REPO}/Kokoro-82M-GGUF`;
export const VOICE_DEFAULT_STT_MODEL = "audiocpp-parakeet-tdt-0.6b-v3";

export const VOICE_TTS_MODELS: readonly string[] = [
  VOICE_DEFAULT_TTS_MODEL,
  ...AUDIO_CPP_MODELS.filter(
    (model) =>
      model.task === "tts" &&
      model.id !== VOICE_DEFAULT_TTS_MODEL &&
      audioCppWorkflowsFor(model).includes("speak"),
  ).map((model) => model.id),
];

/** Opt-in pick: a `speaks` chat model hears each turn and answers in its own voice. */
export const VOICE_CHAT_MODEL_VOICE = "chat-model-voice";
export const CHAT_MODEL_VOICE_SPEAKERS = ["female", "male"] as const;

export function isChatModelVoice(id: string | null | undefined): boolean {
  return id === VOICE_CHAT_MODEL_VOICE;
}

export function speakerCarriesOver(from: string | null | undefined, to: string): boolean {
  return isChatModelVoice(from) === isChatModelVoice(to);
}

export function chatModelSpeaks(state: {
  models: readonly { id: string; speaks?: boolean }[];
  params: { checkpoint: string };
}): boolean {
  return Boolean(state.models.find((m) => m.id === state.params.checkpoint)?.speaks);
}

/** Voice turns carry their speech only to an audio-input model, opted in by mic target or the own-voice pick. */
export function voiceHearsAudio(
  state: {
    models: readonly { id: string; hasAudioInput?: boolean }[];
    params: { checkpoint: string };
    selectedVoiceModelId: string;
    modelLoading?: boolean;
  },
  micTarget: MicTarget,
): boolean {
  const model = state.modelLoading
    ? undefined
    : state.models.find((m) => m.id === state.params.checkpoint);
  const target = isChatModelVoice(state.selectedVoiceModelId) ? "model" : micTarget;
  return micSendsAudioToModel(target, model?.hasAudioInput === true);
}

/** In voice mode only the own-voice pick asks for speech: an audio.cpp voice would talk over it. */
export function wantsSpokenReply(input: {
  speaks: boolean;
  turnHasAudio: boolean;
  voiceModeOn: boolean;
  selectedVoiceModelId: string;
}): boolean {
  if (!input.speaks || !input.turnHasAudio) return false;
  return !input.voiceModeOn || isChatModelVoice(input.selectedVoiceModelId);
}

/** Whether the voice can speak in a saved Audio voice (the request turns into a clone). */
export function voiceTtsCanClone(id: string | null | undefined): boolean {
  const model = audioCppModelFor(id);
  return model !== null && audioCppWorkflowsFor(model).includes("clone");
}

export function migrateVoiceTtsModel(saved: string | null | undefined): string {
  const id = saved?.trim() ?? "";
  if (isChatModelVoice(id)) return id;
  if (!isAudioCppFolderId(id)) return VOICE_DEFAULT_TTS_MODEL;
  return VOICE_TTS_MODELS.some((m) => m.toLowerCase() === id.toLowerCase())
    ? id
    : VOICE_DEFAULT_TTS_MODEL;
}

export function voiceIdServes(serving: string, id: string): boolean {
  const a = serving.trim().replace(/[\\/]+$/, "").toLowerCase();
  const b = id.trim().replace(/[\\/]+$/, "").toLowerCase();
  if (!a || !b) return false;
  if (a === b) return true;
  const folder = b.split("/").pop() ?? b;
  return a.endsWith(`/${folder}`) || a.endsWith(`\\${folder}`);
}

export function servedVoiceTtsModel(serving: readonly string[]): string | null {
  return VOICE_TTS_MODELS.find((id) => serving.some((s) => voiceIdServes(s, id))) ?? null;
}

// The GPU fit estimate counts the loaded chat model, so a small card refuses the voice; audio.cpp TTS runs on CPU.
export function voiceLoadNeedsCpu(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /enough (free )?memory|out of memory/i.test(message);
}
