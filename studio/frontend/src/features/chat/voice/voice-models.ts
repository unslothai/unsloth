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

// The GPU fit estimate counts what is already loaded (the chat model), so a small
// card refuses the voice there; audio.cpp's TTS voices still run on the CPU.
export function voiceLoadNeedsCpu(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /enough (free )?memory|out of memory/i.test(message);
}
