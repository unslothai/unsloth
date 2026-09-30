// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The curated dictation model list and its persisted-settings migration. Split
// out of voice-settings-store so it stays free of app imports and can be tested
// directly by the node runner.

import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_STT_KEYS,
  type AudioCppModel,
  type AudioCppSttKey,
  audioCppSizeLabel,
} from "../../audio/audio-cpp-catalog.ts";

const AUDIO_CPP_STT = AUDIO_CPP_MODELS.filter((model) => model.task === "asr");

/** Curated dictation models, mirrored by the backend sidecars. Listed in the
 * order the picker shows them: the recommended models lead. */
export const STT_MODELS = [
  "qwen3-asr-0.6b",
  "qwen3-asr-1.7b",
  "tiny",
  "base",
  "small",
  "large-v3-turbo",
  "large-v3",
  ...AUDIO_CPP_STT_KEYS,
] as const;

/** Models the picker marks as recommended. Qwen3-ASR is more accurate than
 * Whisper at a comparable size and covers more languages. */
export const RECOMMENDED_STT_MODELS: ReadonlySet<SttModel> = new Set([
  "qwen3-asr-0.6b",
  "qwen3-asr-1.7b",
]);

/** Models served by llama.cpp mtmd rather than whisper.cpp, which is
 * Whisper-only. Each is a GGUF plus an audio mmproj. */
export const MTMD_STT_MODELS: ReadonlySet<SttModel> = new Set([
  "qwen3-asr-0.6b",
  "qwen3-asr-1.7b",
]);
/** Models served by the audio.cpp sidecar. Each key maps to one GGUF package in
 * the shared audio.cpp repo. */
export const AUDIO_CPP_STT_MODELS: ReadonlySet<SttModel> = new Set(
  AUDIO_CPP_STT_KEYS,
);
/** Curated models that transcribe only these primary languages; every other
 * curated model is multilingual. Moonshine and Nemotron are English-only, and
 * Canary covers en/de/es/fr. */
export const STT_MODEL_LANGUAGES: ReadonlyMap<SttModel, readonly string[]> =
  new Map(
    (AUDIO_CPP_STT as readonly AudioCppModel[]).flatMap((model) =>
      model.languages ? [[model.key, model.languages] as const] : [],
    ),
  );
/** Curated models that only transcribe English. */
export const ENGLISH_ONLY_STT_MODELS: ReadonlySet<SttModel> = new Set(
  [...STT_MODEL_LANGUAGES]
    .filter(([, languages]) => languages.length === 1 && languages[0] === "en")
    .map(([model]) => model),
);
export type DefaultSttModel = (typeof STT_MODELS)[number];
/** A curated id or a user-selected Hugging Face `owner/model` repository. */
export type SttModel = string;
/** Whisper repos downloaded through Unsloth's existing Model Hub manager. */
export const STT_MODEL_REPOS: Record<DefaultSttModel, string> = {
  tiny: "unsloth/whisper-tiny",
  base: "unsloth/whisper-base",
  small: "unsloth/whisper-small",
  "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
  "large-v3": "unsloth/whisper-large-v3",
  "qwen3-asr-0.6b": "unslothai/Qwen3-ASR-0.6B-GGUF",
  "qwen3-asr-1.7b": "unslothai/Qwen3-ASR-1.7B-GGUF",
  ...(Object.fromEntries(
    AUDIO_CPP_STT.map((model) => [model.key, model.id]),
  ) as Record<AudioCppSttKey, string>),
};
export const DEFAULT_STT_MODEL: DefaultSttModel = "qwen3-asr-0.6b";
/** The default before Qwen3-ASR, used only by the v1 migration. */
const LEGACY_DEFAULT_STT_MODEL = "small";

/**
 * v0's default was Whisper Small, so a stored "small" is far more often a
 * default nobody touched than a deliberate pick. Move those to the recommended
 * model; choosing Small again persists at v1 and sticks. Any other saved model
 * was chosen on purpose and is left alone.
 */
export function migrateVoiceSettings(
  persisted: unknown,
  fromVersion: number,
): Record<string, unknown> | undefined {
  const saved = persisted as Record<string, unknown> | undefined;
  if (fromVersion < 1 && saved?.sttModel === LEGACY_DEFAULT_STT_MODEL) {
    return { ...saved, sttModel: DEFAULT_STT_MODEL };
  }
  return saved;
}

// Speech-recognition models, not voices. Name and size are separate so lists can
// right-align the size; the download confirmation reads both.
export const STT_MODEL_NAMES: Record<DefaultSttModel, string> = {
  tiny: "Whisper Tiny",
  base: "Whisper Base",
  small: "Whisper Small",
  "large-v3-turbo": "Whisper Large v3 Turbo",
  "large-v3": "Whisper Large v3",
  "qwen3-asr-0.6b": "Qwen3-ASR 0.6B",
  "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
  ...(Object.fromEntries(
    AUDIO_CPP_STT.map((model) => [model.key, `${model.displayName} (audio.cpp)`]),
  ) as Record<AudioCppSttKey, string>),
};
// Whisper sizes are f16 GGML for whisper.cpp; the mtmd entries cover the model
// plus its mmproj, which is why they are larger than the weights alone.
export const STT_MODEL_SIZES: Record<DefaultSttModel, string> = {
  tiny: "78 MB",
  base: "148 MB",
  small: "488 MB",
  "large-v3-turbo": "1.6 GB",
  "large-v3": "3.1 GB",
  "qwen3-asr-0.6b": "1.0 GB",
  "qwen3-asr-1.7b": "2.5 GB",
  ...(Object.fromEntries(
    AUDIO_CPP_STT.map((model) => [model.key, audioCppSizeLabel(model.sizeBytes)]),
  ) as Record<AudioCppSttKey, string>),
};

export function sttModelName(model: SttModel): string {
  return STT_MODEL_NAMES[model as DefaultSttModel] ?? model;
}

export function sttModelSize(model: SttModel): string {
  return STT_MODEL_SIZES[model as DefaultSttModel] ?? "";
}
