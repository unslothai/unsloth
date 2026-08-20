// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  return group.artifacts.some((artifact) => artifact.format === "gguf");
}

export const AUDIO_MODEL_OPTIONS: ModelOption[] =
  catalogToModelOptions(AUDIO_CATALOG);

export function audioModelsForTask(task: AudioTask): ModelOption[] {
  const matches = (option: ModelOption) => audioTaskFor(option.id) === task;
  return AUDIO_MODEL_OPTIONS.filter(matches);
}

export function audioCapabilityLine(
  task: AudioTask,
  detail?: string | null,
): string {
  const base = task === "tts" ? "Text-to-speech" : "Speech-to-text";
  return detail ? `${base} · ${detail}` : base;
}
