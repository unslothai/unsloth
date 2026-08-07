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

/** The catalog task for a repo id; null for repos outside AUDIO_CATALOG. */
export function audioTaskFor(repoId: string): AudioTask | null {
  if (isKnownSttArtifactRepoId(repoId)) return "stt";
  return groupForRepoId(repoId, AUDIO_CATALOG)?.task ?? null;
}

/** The GGUF artifact in this repo's TTS group, if it publishes one. llama.cpp
 *  alone carries the snac/bicodec/dac decoders, so where safetensors loads
 *  through MLX (no TTS branch) this is the only build that can generate. */
export function ggufSiblingFor(repoId: string): string | null {
  const group = groupForRepoId(repoId, AUDIO_CATALOG);
  if (!group || group.task !== "tts") return null;
  const gguf = group.artifacts.find((a) => a.format === "gguf");
  return gguf && gguf.repoId !== repoId ? gguf.repoId : null;
}

/** Curated TTS rows that can actually generate on Apple Silicon. A safetensors
 * row is still useful when its group publishes a GGUF sibling, because the
 * Audio page resolves and stages that exact sibling before loading. */
export function macTtsCatalogChoiceIsRunnable(repoId: string): boolean {
  const group = groupForRepoId(repoId, AUDIO_CATALOG);
  if (!group || group.task !== "tts") return false;
  return group.artifacts.some((artifact) => artifact.format === "gguf");
}

/** Every curated audio artifact, in catalog order (TTS groups, then STT). */
export const AUDIO_MODEL_OPTIONS: ModelOption[] =
  catalogToModelOptions(AUDIO_CATALOG);

/** Curated picker rows for the active mode only. Advertising the other task in
 *  Recommended lets Transcribe load a TTS checkpoint (and vice versa). */
export function audioModelsForTask(task: AudioTask): ModelOption[] {
  const matches = (option: ModelOption) => audioTaskFor(option.id) === task;
  return AUDIO_MODEL_OPTIONS.filter(matches);
}

/** Short capability line for the rail header ("Text-to-speech · snac codec"). */
export function audioCapabilityLine(
  task: AudioTask,
  detail?: string | null,
): string {
  const base = task === "tts" ? "Text-to-speech" : "Speech-to-text";
  return detail ? `${base} · ${detail}` : base;
}
