// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  audioCppModelFor,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog.ts";

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

export function voiceTtsCanClone(id: string | null | undefined): boolean {
  const model = audioCppModelFor(id);
  return model !== null && audioCppWorkflowsFor(model).includes("clone");
}

export function migrateVoiceTtsModel(saved: string | null | undefined): string {
  const id = saved?.trim() ?? "";
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
