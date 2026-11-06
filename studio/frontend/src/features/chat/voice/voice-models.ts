// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which audio.cpp models voice mode listens and speaks with. Free of app imports
// so the node test runner can load it directly.

import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  audioCppModelFor,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog.ts";
import type { MicTarget } from "../../settings/stores/voice-settings-store.ts";
import { micSendsAudioToModel } from "../utils/mic-target.ts";

/** Kokoro: 82M, built-in voices, fast enough on a CPU to keep up sentence by sentence. */
export const VOICE_DEFAULT_TTS_MODEL = `${AUDIO_CPP_REPO}/Kokoro-82M-GGUF`;
/** Parakeet TDT on audio.cpp: a short utterance transcribes in well under a second. */
export const VOICE_DEFAULT_STT_MODEL = "audiocpp-parakeet-tdt-0.6b-v3";

/** audio.cpp voices that can speak plain text, in catalog order, default first. */
export const VOICE_TTS_MODELS: readonly string[] = [
  VOICE_DEFAULT_TTS_MODEL,
  ...AUDIO_CPP_MODELS.filter(
    (model) =>
      model.task === "tts" &&
      model.id !== VOICE_DEFAULT_TTS_MODEL &&
      audioCppWorkflowsFor(model).includes("speak"),
  ).map((model) => model.id),
];

/**
 * Opt-in pick for a chat model that reports `speaks`: it hears each turn and answers in its
 * own voice, so no audio.cpp voice loads. Never picked for the user.
 */
export const VOICE_CHAT_MODEL_VOICE = "chat-model-voice";
/** The chat model's TTS mode speakers (audio_options.voice on /audio/generate). */
export const CHAT_MODEL_VOICE_SPEAKERS = ["female", "male"] as const;

export function isChatModelVoice(id: string | null | undefined): boolean {
  return id === VOICE_CHAT_MODEL_VOICE;
}

/** The chat model's female / male and an audio.cpp voice's speakers are separate lists. */
export function speakerCarriesOver(from: string | null | undefined, to: string): boolean {
  return isChatModelVoice(from) === isChatModelVoice(to);
}

/** Whether the loaded chat model answers in speech (the inference status's `speaks`). */
export function chatModelSpeaks(state: {
  models: readonly { id: string; speaks?: boolean }[];
  params: { checkpoint: string };
}): boolean {
  return Boolean(state.models.find((m) => m.id === state.params.checkpoint)?.speaks);
}

/**
 * Whether voice mode sends each turn's speech to the chat model as audio. Any model with audio
 * input hears it once the mic is set to send audio (the same setting as the composer mic), or
 * the chat model's own voice is picked; otherwise only the transcript goes.
 */
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

/**
 * Whether a turn asks the chat model to answer in speech. Outside voice mode a speaking model
 * answers an audio turn aloud, as before; in voice mode only when its own voice is the pick,
 * since an audio.cpp voice reads the text and the model's speech would go unheard.
 */
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

/**
 * The remembered voice, or the default when it is not an audio.cpp voice. Voice mode
 * used to speak through a llama-server slot (Orpheus and other SNAC GGUFs), and a
 * pick left over from then would load into nothing.
 */
export function migrateVoiceTtsModel(saved: string | null | undefined): string {
  const id = saved?.trim() ?? "";
  if (isChatModelVoice(id)) return id;
  if (!isAudioCppFolderId(id)) return VOICE_DEFAULT_TTS_MODEL;
  return VOICE_TTS_MODELS.some((m) => m.toLowerCase() === id.toLowerCase())
    ? id
    : VOICE_DEFAULT_TTS_MODEL;
}

/** Whether a status `serving` entry is this voice: ids come back as the load path or the repo id. */
export function voiceIdServes(serving: string, id: string): boolean {
  const a = serving.trim().replace(/[\\/]+$/, "").toLowerCase();
  const b = id.trim().replace(/[\\/]+$/, "").toLowerCase();
  if (!a || !b) return false;
  if (a === b) return true;
  const folder = b.split("/").pop() ?? b;
  return a.endsWith(`/${folder}`) || a.endsWith(`\\${folder}`);
}

/** The offered voice a status `serving` list already holds, so voice mode adopts it rather than loading a second. */
export function servedVoiceTtsModel(serving: readonly string[]): string | null {
  return VOICE_TTS_MODELS.find((id) => serving.some((s) => voiceIdServes(s, id))) ?? null;
}

// The GPU fit estimate counts what is already loaded (the chat model), so a small
// card refuses the voice there; audio.cpp's TTS voices still run on the CPU.
export function voiceLoadNeedsCpu(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /enough (free )?memory|out of memory/i.test(message);
}
