// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_CPP_DICTATION_MODELS,
  isAudioCppFolderId,
} from "./audio-cpp-catalog.ts";

export type AudioSttEngine = "transformers" | "gguf" | "mtmd" | "audiocpp";

export function isAudioSttEngine(value: unknown): value is AudioSttEngine {
  return (
    value === "transformers" ||
    value === "gguf" ||
    value === "mtmd" ||
    value === "audiocpp"
  );
}

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

// whisper.cpp's GGML repos, retired: a saved one resolves to its Transformers twin.
const GGUF_REPO_BY_KEY: Record<string, string> = {
  tiny: "unslothai/whisper-tiny-GGUF",
  base: "unslothai/whisper-base-GGUF",
  small: "unslothai/whisper-small-GGUF",
  "large-v3-turbo": "unslothai/whisper-large-v3-turbo-GGUF",
  "large-v3": "unslothai/whisper-large-v3-GGUF",
};

// The audiocpp engine takes any GGUF ASR repo or package folder id as its key. Dictation keys
// saved before that are distinct from every other engine's and resolve to their folder id.
const AUDIOCPP_REPO_BY_KEY: Record<string, string> = Object.fromEntries(
  AUDIO_CPP_DICTATION_MODELS.map((model) => [model.key, model.id]),
);

const GGUF_REPO_SUFFIX = /-GGUF$/i;

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
    (repoId) => [repoId.toLowerCase(), "transformers"] as const,
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
  // A saved audiocpp dictation key names one folder whatever engine the caller assumed.
  if (engine === "audiocpp" || Object.hasOwn(AUDIOCPP_REPO_BY_KEY, normalized))
    return AUDIOCPP_REPO_BY_KEY[normalized] ?? sidecarKey;
  if (engine === "mtmd") return MTMD_REPO_BY_KEY[normalized] ?? sidecarKey;
  return TRANSFORMERS_REPO_BY_KEY[normalized] ?? sidecarKey;
}

/** Resolve from the picker artifact, not the shared short sidecar key. The curated Qwen3-ASR
 *  GGUFs run on llama.cpp and the retired Whisper GGUFs on Transformers; every other GGUF ASR
 *  repo is the audio runtime's, and a safetensors repo runs on Transformers. */
export function sttEngineForRepoId(
  repoId: string,
  isGguf?: boolean | null,
): AudioSttEngine {
  const normalized = repoId.trim().toLowerCase().replace(/\/+$/, "");
  const known = ENGINE_BY_REPO.get(normalized);
  if (known) return known;
  if (Object.hasOwn(AUDIOCPP_REPO_BY_KEY, normalized)) return "audiocpp";
  const leaf = normalized.split("/").pop() ?? "";
  return isGguf || isAudioCppFolderId(normalized) || GGUF_REPO_SUFFIX.test(leaf)
    ? "audiocpp"
    : "transformers";
}

const TRANSFORMERS_TWIN_BY_GGUF_REPO = new Map<string, string>(
  Object.entries(GGUF_REPO_BY_KEY).map(([key, repoId]) => [
    repoId.toLowerCase(),
    TRANSFORMERS_REPO_BY_KEY[key],
  ]),
);

/** Whether a saved Transcribe pick was a whisper.cpp one: its GGML repo or its engine. */
export function isRetiredWhisperChoice(repo: string, engine: string): boolean {
  return (
    engine === "gguf" ||
    TRANSFORMERS_TWIN_BY_GGUF_REPO.has(repo.trim().toLowerCase())
  );
}

type SttHostAvailability = {
  transformers?: { available: boolean };
  mtmd?: { available: boolean };
  audiocpp?: { available: boolean };
};

/** Where a retired whisper.cpp pick goes on this host: null keeps its Transformers twin (PyTorch
 *  is here), otherwise llama.cpp Qwen3-ASR 0.6B, else audio.cpp's, as the backend serves the
 *  retired engine. undefined while the host cannot tell yet or has neither. */
export function retiredWhisperFallback(
  host: SttHostAvailability,
): { repo: string; engine: AudioSttEngine } | null | undefined {
  if (host.transformers?.available === true) return null;
  if (host.transformers?.available !== false) return undefined;
  if (host.mtmd?.available) {
    return { repo: MTMD_REPO_BY_KEY["qwen3-asr-0.6b"], engine: "mtmd" };
  }
  if (host.audiocpp?.available) {
    return {
      repo: AUDIOCPP_REPO_BY_KEY["audiocpp-qwen3-asr-0.6b"],
      engine: "audiocpp",
    };
  }
  return undefined;
}

/** A saved Transcribe pick in its current form: a retired whisper.cpp repo or engine becomes
 *  its Transformers twin. Other picks pass through unchanged. */
export function canonicalSttChoice(
  repo: string,
  engine: string,
): { repo: string; engine: string } {
  return {
    repo: TRANSFORMERS_TWIN_BY_GGUF_REPO.get(repo.trim().toLowerCase()) ?? repo,
    engine: engine === "gguf" ? "transformers" : engine,
  };
}
