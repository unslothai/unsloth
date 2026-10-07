// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The curated dictation model list and its persisted-settings migration. Split
// out of voice-settings-store so it stays free of app imports and can be tested
// directly by the node runner.

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
/** Models served by the GGUF audio runtime's sidecar. Each key maps to one
 * package folder of the shared GGUF repo. */
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
    AUDIO_CPP_DICTATION_MODELS.map((model) => [model.key, model.id]),
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
  // The name the Hub shows for the package, plus the size where one folder ships two.
  ...(Object.fromEntries(
    AUDIO_CPP_DICTATION_MODELS.map((model) => [
      model.key,
      "variant" in model
        ? `${audioCppDisplayName(model.id)} (${model.variant})`
        : audioCppDisplayName(model.id),
    ]),
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

/** The quant a dictation pick runs. Only a package folder row takes one: a saved key already
 *  names its own, and "" leaves the row's resident or default quant. */
export function sttModelVariant(
  model: SttModel,
  variant: string,
): string | null {
  return variant && isAudioCppFolderId(model) ? variant : null;
}

/** `model` as one id with its quant folded in (`row:variant`), as the audio runtime reads it. */
export function withSttVariant(model: SttModel, variant: string): string {
  const quant = sttModelVariant(model, variant);
  return quant ? `${model}:${quant}` : model;
}

/** The quant a dictation pick runs, as the listing names it: the pinned one, else the resident one
 *  (a cache-only load can report the loose key, "Q8_0" for "small/Q8_0"), else what a bare load
 *  picks offline, the first cached quant in listing order, else the default. */
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
