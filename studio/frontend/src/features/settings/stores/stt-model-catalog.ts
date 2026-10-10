// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node runner can test it directly.

import {
  AUDIO_CPP_DICTATION_MODELS,
  AUDIO_CPP_MODELS,
  AUDIO_CPP_STT_KEYS,
  type AudioCppSttKey,
  audioCppDisplayName,
  audioCppModelFor,
  audioCppSizeLabel,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog.ts";

/** Mirrored by the backend sidecars; picker order, recommended first. */
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

export const RECOMMENDED_STT_MODELS: ReadonlySet<SttModel> = new Set([
  "qwen3-asr-0.6b",
  "qwen3-asr-1.7b",
]);

/** Served by llama.cpp mtmd (GGUF plus audio mmproj); whisper.cpp is Whisper-only. */
export const MTMD_STT_MODELS: ReadonlySet<SttModel> = new Set([
  "qwen3-asr-0.6b",
  "qwen3-asr-1.7b",
]);
/** Each key maps to one package folder of the shared GGUF repo. */
export const AUDIO_CPP_STT_MODELS: ReadonlySet<SttModel> = new Set(
  AUDIO_CPP_STT_KEYS,
);
const KEYED_FOLDERS = new Set(
  AUDIO_CPP_DICTATION_MODELS.map((model) => model.id.toLowerCase()),
);
/** The Transcribe page's audio.cpp ASR folders no key above names; dictation
 * lists them by folder id, in catalog order (the largest come last). */
export const AUDIO_CPP_STT_FOLDER_IDS: readonly string[] =
  AUDIO_CPP_MODELS.filter(
    (model) =>
      audioCppWorkflowsFor(model).includes("transcribe") &&
      !KEYED_FOLDERS.has(model.id.toLowerCase()),
  ).map((model) => model.id);
/** What the Voice picker lists: the curated models, then every other ASR model
 * Transcribe offers. */
export const STT_PICKER_MODELS: readonly SttModel[] = [
  ...STT_MODELS,
  ...AUDIO_CPP_STT_FOLDER_IDS,
];
/** Models that transcribe only these primary languages; every other model is
 * multilingual. Moonshine and Nemotron are English-only, Canary covers
 * en/de/es/fr, and Hviske is Danish. */
export const STT_MODEL_LANGUAGES: ReadonlyMap<SttModel, readonly string[]> =
  new Map(
    [
      ...AUDIO_CPP_DICTATION_MODELS.map((model) => [model.key, model.id]),
      ...AUDIO_CPP_STT_FOLDER_IDS.map((id) => [id, id]),
    ].flatMap(([model, id]) => {
      const languages = audioCppModelFor(id)?.languages;
      return languages ? [[model, languages] as const] : [];
    }),
  );
/** Models that only transcribe English. */
export const ENGLISH_ONLY_STT_MODELS: ReadonlySet<SttModel> = new Set(
  [...STT_MODEL_LANGUAGES]
    .filter(([, languages]) => languages.length === 1 && languages[0] === "en")
    .map(([model]) => model),
);
export type DefaultSttModel = (typeof STT_MODELS)[number];
export type SttModel = string;
export const STT_MODEL_REPOS: Record<DefaultSttModel, string> = {
  tiny: "unsloth/whisper-tiny",
  base: "unsloth/whisper-base",
  small: "unsloth/whisper-small",
  "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
  "large-v3": "unsloth/whisper-large-v3",
  "qwen3-asr-0.6b": "unslothai/Qwen3-ASR-0.6B-GGUF",
  "qwen3-asr-1.7b": "unslothai/Qwen3-ASR-1.7B-GGUF",
  ...(Object.fromEntries(
    AUDIO_CPP_DICTATION_MODELS.map((model) => [model.key, model.id]),
  ) as Record<AudioCppSttKey, string>),
};
export const DEFAULT_STT_MODEL: DefaultSttModel = "qwen3-asr-0.6b";
const LEGACY_DEFAULT_STT_MODEL = "small";

/** v0 defaulted to Whisper Small, so a stored "small" is usually untouched; move it to the
 * recommended model. Re-picking Small persists at v1 and sticks. */
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

// Separate name and size so lists can right-align the size.
export const STT_MODEL_NAMES: Record<DefaultSttModel, string> = {
  tiny: "Whisper Tiny",
  base: "Whisper Base",
  small: "Whisper Small",
  "large-v3-turbo": "Whisper Large v3 Turbo",
  "large-v3": "Whisper Large v3",
  "qwen3-asr-0.6b": "Qwen3-ASR 0.6B",
  "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
  ...(Object.fromEntries(
    AUDIO_CPP_DICTATION_MODELS.map((model) => [
      model.key,
      "variant" in model
        ? `${audioCppDisplayName(model.id)} (${model.variant})`
        : audioCppDisplayName(model.id),
    ]),
  ) as Record<AudioCppSttKey, string>),
};
// Whisper sizes are f16 GGML; mtmd sizes include the mmproj.
export const STT_MODEL_SIZES: Record<DefaultSttModel, string> = {
  tiny: "78 MB",
  base: "148 MB",
  small: "488 MB",
  "large-v3-turbo": "1.6 GB",
  "large-v3": "3.1 GB",
  "qwen3-asr-0.6b": "1.0 GB",
  "qwen3-asr-1.7b": "2.5 GB",
  ...(Object.fromEntries(
    AUDIO_CPP_DICTATION_MODELS.map((model) => [
      model.key,
      audioCppSizeLabel(model.sizeBytes),
    ]),
  ) as Record<AudioCppSttKey, string>),
};

export function sttModelName(model: SttModel): string {
  return (
    STT_MODEL_NAMES[model as DefaultSttModel] ??
    (isAudioCppFolderId(model) ? audioCppDisplayName(model) : model)
  );
}

/** package folder quant; "" uses the resident or default, while saved keys encode their own. */
export function sttModelVariant(
  model: SttModel,
  variant: string,
): string | null {
  return variant && isAudioCppFolderId(model) ? variant : null;
}

/** folds a quant into the audio runtime model id as `row:variant`. */
export function withSttVariant(model: SttModel, variant: string): string {
  const quant = sttModelVariant(model, variant);
  return quant ? `${model}:${quant}` : model;
}

/** picks the pinned, loaded, cached, or default quant; resolves loose cache keys like `Q8_0`. */
export function sttShownVariant(
  pinned: string | null,
  loaded: string | null,
  listing: {
    default_variant: string | null;
    variants: readonly { quant: string; downloaded?: boolean }[];
  } | null,
): string | null {
  if (pinned) return pinned;
  if (!listing) return loaded;
  if (loaded) {
    if (listing.variants.some((variant) => variant.quant === loaded)) {
      return loaded;
    }
    const scoped = listing.variants.filter(
      (variant) => variant.downloaded && variant.quant.endsWith(`/${loaded}`),
    );
    if (scoped.length === 1) return scoped[0].quant;
  }
  return (
    listing.variants.find((variant) => variant.downloaded)?.quant ??
    listing.default_variant
  );
}

/** Whether a listing leaves the pinned quant possibly on disk. Only the row with that exact key can
 *  say no: a cache-only (offline) listing keys cached files by what tells them apart ("ctc/F16" for
 *  "v3-ctc/F16"), so a key it leaves out may still be cached; the backend matches it on load. */
export function sttListedQuantDownloaded(
  listing: { variants: readonly { quant: string; downloaded?: boolean }[] },
  pinned: string,
): boolean {
  return (
    listing.variants.find((variant) => variant.quant === pinned)?.downloaded !==
    false
  );
}

export function sttModelSize(model: SttModel): string {
  return STT_MODEL_SIZES[model as DefaultSttModel] ?? "";
}
