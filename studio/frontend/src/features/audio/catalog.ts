// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_CATALOG,
  catalogToModelOptions,
  groupForRepoId,
} from "@/features/model-picker/components/model-selector/model-catalog";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import {
  AUDIO_CPP_AUDIO_TYPES,
  AUDIO_CPP_MODELS,
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
} from "./audio-cpp-catalog";
import {
  type AudioSttEngine,
  isKnownSttArtifactRepoId,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "./stt-artifacts";

export {
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
  type AudioSttEngine,
};

export type AudioTask = "tts" | "stt";

// Recommended GGUF speech and music load into the main slot like the native PyTorch models, but
// through the GGUF audio runtime, so they need no remote code and run on every backend it builds for.
const AUDIO_CPP_GENERATION_IDS = AUDIO_CPP_MODELS.filter(
  (model) => model.task !== "asr",
).map((model) => model.id.toLowerCase());
const AUDIO_CPP_MUSIC_IDS = AUDIO_CPP_MODELS.filter(
  (model) => model.task === "music",
).map((model) => model.id.toLowerCase());

const NATIVE_TTS_REPOS = new Set([
  "bosonai/higgs-tts-2-3b-base",
  "openmoss-team/moss-tts-local-transformer-v1.5",
  "openmoss-team/moss-tts-nano-100m",
  "multimodalart/higgs-audio-v3-tts-4b-transformers",
  "minimaxai/minimax-music3",
  ...AUDIO_CPP_GENERATION_IDS,
]);

const REMOTE_CODE_TTS_REPOS = new Set([
  "openmoss-team/moss-tts-local-transformer-v1.5",
  "openmoss-team/moss-tts-nano-100m",
  "multimodalart/higgs-audio-v3-tts-4b-transformers",
]);

/** Music models that only run on NVIDIA CUDA. */
const CUDA_MUSIC_GENERATION_REPOS = new Set(["minimaxai/minimax-music3"]);
const MUSIC_GENERATION_REPOS = new Set([
  ...CUDA_MUSIC_GENERATION_REPOS,
  ...AUDIO_CPP_MUSIC_IDS,
]);
const NATIVE_TTS_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
  ...AUDIO_CPP_AUDIO_TYPES,
]);
const REMOTE_CODE_TTS_AUDIO_TYPES = new Set([
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
]);

const normalizedRepoId = (repoId: string): string =>
  repoId.trim().toLowerCase();

export function usesNativeAudioRuntime(
  repoId: string,
  audioType?: string | null,
): boolean {
  return (
    NATIVE_TTS_REPOS.has(normalizedRepoId(repoId)) ||
    Boolean(audioType && NATIVE_TTS_AUDIO_TYPES.has(audioType))
  );
}

export function audioModelRequiresRemoteCode(
  repoId: string,
  audioType?: string | null,
): boolean {
  return (
    REMOTE_CODE_TTS_REPOS.has(normalizedRepoId(repoId)) ||
    Boolean(audioType && REMOTE_CODE_TTS_AUDIO_TYPES.has(audioType))
  );
}

export function isMusicGenerationModel(
  repoId?: string | null,
  audioType?: string | null,
): boolean {
  return Boolean(
    audioType === "minimax_music3" ||
      audioType === AUDIO_CPP_MUSIC_AUDIO_TYPE ||
      (repoId && MUSIC_GENERATION_REPOS.has(normalizedRepoId(repoId))),
  );
}

/** The MiniMax Music 3 pipeline needs a CUDA GPU; GGUF music runs wherever the audio runtime does. */
export function musicGenerationRequiresCuda(
  repoId?: string | null,
  audioType?: string | null,
): boolean {
  return Boolean(
    audioType === "minimax_music3" ||
      (repoId && CUDA_MUSIC_GENERATION_REPOS.has(normalizedRepoId(repoId))),
  );
}

export function audioTaskFor(repoId: string): AudioTask | null {
  if (isKnownSttArtifactRepoId(repoId)) return "stt";
  return groupForRepoId(repoId, AUDIO_CATALOG)?.task ?? null;
}

export function ggufSiblingFor(repoId: string): string | null {
  const group = groupForRepoId(repoId, AUDIO_CATALOG);
  if (!group || group.task !== "tts") return null;
  const gguf = group.artifacts.find((a) => a.format === "gguf");
  return gguf && gguf.repoId !== repoId ? gguf.repoId : null;
}

export function macTtsCatalogChoiceIsRunnable(repoId: string): boolean {
  const group = groupForRepoId(repoId, AUDIO_CATALOG);
  if (!group || group.task !== "tts") return false;
  return (
    group.artifacts.some((artifact) => artifact.format === "gguf") ||
    (usesNativeAudioRuntime(repoId) && !musicGenerationRequiresCuda(repoId))
  );
}

export const AUDIO_MODEL_OPTIONS: ModelOption[] =
  catalogToModelOptions(AUDIO_CATALOG);

export function audioModelsForTask(task: AudioTask): ModelOption[] {
  const matches = (option: ModelOption) => audioTaskFor(option.id) === task;
  return AUDIO_MODEL_OPTIONS.filter(matches);
}

const CAPABILITY_LABEL: Record<AudioTask | "music" | "clone" | "edit", string> = {
  music: "Music generation",
  clone: "Voice cloning",
  edit: "Speech editing",
  tts: "Text-to-speech",
  stt: "Speech-to-text",
};

export function audioCapabilityLine(
  task: AudioTask | "music" | "clone" | "edit",
  detail?: string | null,
): string {
  const base = CAPABILITY_LABEL[task];
  // GGUF audio models report one runtime-wide type; the internal name means nothing to a user.
  const shown = detail && AUDIO_CPP_AUDIO_TYPES.has(detail) ? "GGUF" : detail;
  return shown ? `${base} · ${shown}` : base;
}
