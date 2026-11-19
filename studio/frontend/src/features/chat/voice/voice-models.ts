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

// Voices needing an input voice mode never sends (Maya1: model_specs/maya1.json `instruct` required).
const VOICE_NEEDS_INPUT = new Set([`${AUDIO_CPP_REPO}/Maya1-GGUF`.toLowerCase()]);

export const VOICE_TTS_MODELS: readonly string[] = [
  VOICE_DEFAULT_TTS_MODEL,
  ...AUDIO_CPP_MODELS.filter(
    (model) =>
      model.task === "tts" &&
      model.id !== VOICE_DEFAULT_TTS_MODEL &&
      audioCppWorkflowsFor(model).includes("speak") &&
      !VOICE_NEEDS_INPUT.has(model.id.toLowerCase()),
  ).map((model) => model.id),
];

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

/** Voice mode reasons only when asked: reasoning tokens would stall the silent wait. */
export function reasoningForTurn(input: {
  enabled: boolean;
  voiceModeOn: boolean;
  voiceThinking: boolean;
}): boolean {
  return input.enabled && (!input.voiceModeOn || input.voiceThinking);
}

/** Only chat-model voices abort mid-sentence: an audio.cpp cancel kills its process, and reloading takes 35 to 88 s. */
export function cutAbortsSpeechRequest(voiceModelId: string | null | undefined): boolean {
  return isChatModelVoice(voiceModelId);
}

// These models change speaker every sentence, even with a fixed seed; a clone keeps one.
const VOICE_VARIES_PER_SENTENCE = new Set(
  [
    "OmniVoice-GGUF",
    "Breeze-TTS-2-GGUF",
    "MOSS-TTS-Nano-100M-GGUF",
    "DotTTS-Edit-GGUF",
    "VoxCPM1-GGUF",
    "Irodori-TTS-v4-Small-GGUF",
    "Irodori-TTS-500M-v3-GGUF",
    "Irodori-TTS-600M-v3-VoiceDesign-GGUF",
  ].map((folder) => `${AUDIO_CPP_REPO}/${folder}`.toLowerCase()),
);

export function voiceVariesPerSentence(id: string | null | undefined): boolean {
  return Boolean(id) && VOICE_VARIES_PER_SENTENCE.has(id!.toLowerCase());
}

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
