// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type AudioSttEngine = "transformers" | "gguf" | "mtmd";

const TRANSFORMERS_REPO_BY_KEY: Record<string, string> = {
  tiny: "unsloth/whisper-tiny",
  base: "unsloth/whisper-base",
  small: "unsloth/whisper-small",
  "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
  "large-v3": "unsloth/whisper-large-v3",
};

const MTMD_REPO_BY_KEY: Record<string, string> = {
  "qwen3-asr-0.6b": "unslothai/Qwen3-ASR-0.6B-GGUF",
  "qwen3-asr-1.7b": "unslothai/Qwen3-ASR-1.7B-GGUF",
};

const GGUF_REPO_BY_KEY: Record<string, string> = {
  tiny: "unslothai/whisper-tiny-GGUF",
  base: "unslothai/whisper-base-GGUF",
  small: "unslothai/whisper-small-GGUF",
  "large-v3-turbo": "unslothai/whisper-large-v3-turbo-GGUF",
  "large-v3": "unslothai/whisper-large-v3-GGUF",
};

const KEY_BY_REPO = new Map<string, string>(
  [
    ...Object.entries(TRANSFORMERS_REPO_BY_KEY),
    ...Object.entries(GGUF_REPO_BY_KEY),
    ...Object.entries(MTMD_REPO_BY_KEY),
  ].map(([key, repoId]) => [repoId.toLowerCase(), key]),
);

const ENGINE_BY_REPO = new Map<string, AudioSttEngine>([
  ...Object.values(TRANSFORMERS_REPO_BY_KEY).map(
    (repoId) => [repoId.toLowerCase(), "transformers"] as const,
  ),
  ...Object.values(GGUF_REPO_BY_KEY).map(
    (repoId) => [repoId.toLowerCase(), "gguf"] as const,
  ),
  ...Object.values(MTMD_REPO_BY_KEY).map(
    (repoId) => [repoId.toLowerCase(), "mtmd"] as const,
  ),
]);

export function isKnownSttArtifactRepoId(repoId: string): boolean {
  return KEY_BY_REPO.has(repoId.trim().toLowerCase());
}

export function sttSidecarKeyFor(repoId: string): string {
  return KEY_BY_REPO.get(repoId.trim().toLowerCase()) ?? repoId;
}

export function sttRepoIdForSidecarKey(
  sidecarKey: string,
  engine: AudioSttEngine = "transformers",
): string {
  const normalized = sidecarKey.trim().toLowerCase();
  if (engine === "gguf") return GGUF_REPO_BY_KEY[normalized] ?? sidecarKey;
  if (engine === "mtmd") return MTMD_REPO_BY_KEY[normalized] ?? sidecarKey;
  return TRANSFORMERS_REPO_BY_KEY[normalized] ?? sidecarKey;
}

/** Resolve from the picker artifact, not the shared short sidecar key. */
export function sttEngineForRepoId(repoId: string): AudioSttEngine {
  const normalized = repoId.trim().toLowerCase();
  return ENGINE_BY_REPO.get(normalized) ?? "transformers";
}
