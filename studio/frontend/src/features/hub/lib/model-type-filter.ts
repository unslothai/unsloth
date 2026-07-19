// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Model-type classification for the On Device list's type filter. Rows are
// bucketed by pipeline tag first (authoritative when present), then hub tags,
// then well-known name patterns; anything left is a text model.

import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "@/features/hub/inventory/types";
import { EMBEDDING_TAGS } from "./hf-model-meta";

export type ModelTypeFilter =
  | "all"
  | "text"
  | "vision"
  | "embedding"
  | "stt"
  | "tts"
  | "diffusion";

export const MODEL_TYPE_FILTER_OPTIONS: ReadonlyArray<{
  value: ModelTypeFilter;
  label: string;
}> = [
  { value: "all", label: "All types" },
  { value: "text", label: "Text" },
  { value: "vision", label: "Vision" },
  { value: "embedding", label: "Embedding" },
  { value: "stt", label: "Speech to text" },
  { value: "tts", label: "Text to speech" },
  { value: "diffusion", label: "Diffusion" },
];

const STT_PIPELINES = new Set([
  "automatic-speech-recognition",
  "audio-classification",
]);
const STT_TAGS = new Set(["asr", "speech-recognition", "speech-to-text"]);
const STT_NAME_RE = /(whisper|parakeet|canary|moonshine|wav2vec)/i;

const TTS_PIPELINES = new Set(["text-to-speech", "text-to-audio"]);
const TTS_TAGS = new Set(["tts", "speech-synthesis"]);
const TTS_NAME_RE =
  /(-tts|tts-|kokoro|orpheus|csm-|chatterbox|bark|\bdia\b|xtts|piper)/i;

const EMBEDDING_NAME_RE =
  /(embed|-bge-|^bge-|\bgte\b|-gte-|\be5\b|-e5-|minilm|reranker|nomic-embed|snowflake-arctic-embed)/i;

const DIFFUSION_PIPELINES = new Set([
  "text-to-image",
  "image-to-image",
  "text-to-video",
  "image-to-video",
  "unconditional-image-generation",
]);
const DIFFUSION_TAGS = new Set([
  "diffusers",
  "diffusion",
  "stable-diffusion",
  "latent-diffusion",
  "flux",
]);
const DIFFUSION_NAME_RE =
  /(stable-diffusion|\bsdxl\b|\bflux\b|qwen-image|hunyuan-?(video|image)|wan2|latent-consistency|-lcm\b|dreamshaper)/i;

const VISION_PIPELINES = new Set([
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "image-classification",
  "image-segmentation",
  "object-detection",
  "zero-shot-image-classification",
  "any-to-any",
]);
const VISION_TAGS = new Set([
  "vision",
  "vision-language",
  "multimodal",
  "vlm",
  "llava",
]);
const VISION_NAME_RE = /(-vl\b|-vl-|vision|llava|moondream|paligemma)/i;

export type ModelTypeKey = Exclude<ModelTypeFilter, "all">;

export function classifyInventoryModelType(
  row: CachedInventoryRow | LocalInventoryRow,
): ModelTypeKey {
  const pipeline = row.pipelineTag?.toLowerCase() ?? "";
  const tags = new Set((row.tags ?? []).map((t) => t.toLowerCase()));
  const name =
    row.kind === "local"
      ? `${row.id} ${row.repoId ?? ""} ${row.title} ${row.modelId ?? ""}`
      : `${row.id} ${row.repoId}`;
  const hasTag = (wanted: ReadonlySet<string>) => {
    for (const tag of wanted) {
      if (tags.has(tag)) return true;
    }
    return false;
  };

  if (
    STT_PIPELINES.has(pipeline) ||
    hasTag(STT_TAGS) ||
    STT_NAME_RE.test(name)
  ) {
    return "stt";
  }
  if (
    TTS_PIPELINES.has(pipeline) ||
    hasTag(TTS_TAGS) ||
    TTS_NAME_RE.test(name)
  ) {
    return "tts";
  }
  if (
    EMBEDDING_TAGS.has(pipeline) ||
    hasTag(EMBEDDING_TAGS) ||
    EMBEDDING_NAME_RE.test(name)
  ) {
    return "embedding";
  }
  if (
    DIFFUSION_PIPELINES.has(pipeline) ||
    hasTag(DIFFUSION_TAGS) ||
    row.libraryName?.toLowerCase() === "diffusers" ||
    DIFFUSION_NAME_RE.test(name)
  ) {
    return "diffusion";
  }
  if (
    VISION_PIPELINES.has(pipeline) ||
    hasTag(VISION_TAGS) ||
    row.capabilities.supportsVision ||
    VISION_NAME_RE.test(name)
  ) {
    return "vision";
  }
  return "text";
}

export function matchesModelType(
  row: CachedInventoryRow | LocalInventoryRow,
  filter: ModelTypeFilter,
): boolean {
  return filter === "all" || classifyInventoryModelType(row) === filter;
}
