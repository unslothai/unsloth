// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Task lookups over AUDIO_CATALOG plus the mapping from curated STT repo ids to
// the dictation sidecar model keys the /audio/stt/* endpoints understand.

import {
  AUDIO_CATALOG,
  catalogToModelOptions,
  groupForRepoId,
} from "@/features/model-picker/components/model-selector/model-catalog";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";

export type AudioTask = "tts" | "stt";

/** The catalog task for a repo id; null for repos outside AUDIO_CATALOG. */
export function audioTaskFor(repoId: string): AudioTask | null {
  return groupForRepoId(repoId, AUDIO_CATALOG)?.task ?? null;
}

/** Every curated audio artifact, in catalog order (TTS groups, then STT). */
export const AUDIO_MODEL_OPTIONS: ModelOption[] =
  catalogToModelOptions(AUDIO_CATALOG);

/** Picker rows for a mode, that mode's own task first.
 *
 *  Recommended seeds curated rows in the order given, and the list scrolls, so a
 *  fixed order left every STT row (both Qwen3-ASR included) below the fold on
 *  Transcribe. The other task still follows, so switching stays one click away. */
export function audioModelsForTask(task: AudioTask): ModelOption[] {
  const matches = (option: ModelOption) => audioTaskFor(option.id) === task;
  return [
    ...AUDIO_MODEL_OPTIONS.filter(matches),
    ...AUDIO_MODEL_OPTIONS.filter((option) => !matches(option)),
  ];
}

/** Curated STT repo id -> the sidecar model key /audio/stt/* and /audio/transcribe
 *  expect. Mirrors STT_MODEL_REPOS in stt-model-catalog.ts, inverted. */
const STT_SIDECAR_KEY_BY_REPO: Record<string, string> = {
  "unsloth/whisper-tiny": "tiny",
  "unsloth/whisper-base": "base",
  "unsloth/whisper-small": "small",
  "unsloth/whisper-large-v3-turbo": "large-v3-turbo",
  "unsloth/whisper-large-v3": "large-v3",
  "unslothai/qwen3-asr-0.6b-gguf": "qwen3-asr-0.6b",
  "unslothai/qwen3-asr-1.7b-gguf": "qwen3-asr-1.7b",
};

/** The sidecar key for a curated STT repo, falling back to the id itself so a
 *  custom Whisper repo still reaches the transformers sidecar unchanged. */
export function sttSidecarKeyFor(repoId: string): string {
  return STT_SIDECAR_KEY_BY_REPO[repoId.trim().toLowerCase()] ?? repoId;
}

/** Short capability line for the rail header ("Text-to-speech · snac codec"). */
export function audioCapabilityLine(
  task: AudioTask,
  detail?: string | null,
): string {
  const base = task === "tts" ? "Text-to-speech" : "Speech-to-text";
  return detail ? `${base} · ${detail}` : base;
}
