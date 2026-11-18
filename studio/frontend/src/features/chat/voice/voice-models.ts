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

/**
 * Whether the model reasons before this turn. A turn sent while voice mode is running waits in
 * silence for every reasoning token, so it thinks only when the user asked voice mode to.
 */
export function reasoningForTurn(input: {
  enabled: boolean;
  voiceModeOn: boolean;
  voiceThinking: boolean;
}): boolean {
  return input.enabled && (!input.voiceModeOn || input.voiceThinking);
}

/**
 * Whether cutting a reply also aborts the sentence being synthesized. The server can only cancel an
 * audio.cpp voice by stopping its process, and the next reply then waits for the voice to load again
 * (35 to 88 seconds on a busy host). A sentence takes a fraction of a second, so it is left to
 * finish and its clip is dropped.
 */
export function cutAbortsSpeechRequest(voiceModelId: string | null | undefined): boolean {
  return isChatModelVoice(voiceModelId);
}

// From the 31-voice listening sweep (an audio model counting speakers over seven sentences): with no saved
// voice these switch speaker every sentence, even with a fixed seed. A clone keeps a single speaker.
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
