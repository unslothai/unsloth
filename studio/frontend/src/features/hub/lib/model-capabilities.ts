// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { EMBEDDING_TAGS } from "./hf-model-meta";

export type CapabilityKey =
  | "vision"
  | "audio"
  | "tools"
  | "reasoning"
  | "code"
  | "embedding"
  | "diffusion"
  | "multilingual"
  | "conversational";

export interface Capability {
  key: CapabilityKey;
  label: string;
}

/** What the Hub's filter picks from. Image and video generation share the "diffusion" capability
 *  everywhere else (loading, the Images and Video pages) but are different models to look for. */
export type HubCapabilityFilter =
  | "text"
  | "reasoning"
  | "vision"
  | "audio"
  | "embedding"
  | "image-gen"
  | "video-gen";

export const HUB_CAPABILITY_FILTERS: readonly HubCapabilityFilter[] = [
  "text",
  "reasoning",
  "vision",
  "audio",
  "embedding",
  "image-gen",
  "video-gen",
];

const VIDEO_GEN_TAGS = new Set([
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
]);
const VIDEO_GEN_ID_RE =
  /wan2|hunyuan[-_]?video|\bltx(?:[-_]?video|v|[-_]?\d)|minimax[-_]?h\d|cogvideo|mochi|skyreels|[-_]t2v\b|[-_]i2v\b/;

const NEITHER_GEN_TASKS = new Set([
  "video-to-video",
  "text-to-audio",
  "audio-to-audio",
  "text-to-speech",
]);

/** A video family in the repo name (not the owner), e.g. LTX-2.3 or Wan2.2. */
export function isVideoFamilyName(modelId: string): boolean {
  return VIDEO_GEN_ID_RE.test(modelId.toLowerCase().split("/").pop() ?? "");
}

/** Which generation an image/video model does, from its tags and name. A video signal wins: several
 *  video repos carry an image tag for their first-frame path, and the picker reads them the same
 *  way. Anything else is taken for an image model, as most are. */
export function generationKinds(
  tags: string[] | undefined,
  pipelineTag: string | undefined,
  modelId: string,
): { image: boolean; video: boolean } {
  const tagSet = lower(tags);
  const pipeline = pipelineTag?.toLowerCase();
  // The repo name only: "hunyuanvideo-community/HunyuanImage-2.1" is an image model.
  const repo = modelId.toLowerCase().split("/").pop() ?? "";
  // Neither: the Video page takes a prompt and reference images, never a video, and the audio
  // tasks belong to the Audio pages.
  if (
    pipeline !== undefined
      ? NEITHER_GEN_TASKS.has(pipeline)
      : [...NEITHER_GEN_TASKS].some((task) => tagSet.has(task))
  ) {
    return { image: false, video: false };
  }
  const video =
    [...VIDEO_GEN_TAGS].some((tag) => tagSet.has(tag)) ||
    (pipeline !== undefined && VIDEO_GEN_TAGS.has(pipeline)) ||
    VIDEO_GEN_ID_RE.test(repo);
  return { image: !video, video };
}

const VISION_TAGS = new Set([
  "vision",
  "vision-language",
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "image-classification",
  "image-segmentation",
  "object-detection",
  "any-to-any",
  "multimodal",
  "vlm",
  "llava",
]);

const AUDIO_TAGS = new Set([
  "audio",
  "audio-text-to-text",
  "automatic-speech-recognition",
  "text-to-speech",
  "audio-classification",
  "voice-activity-detection",
  "asr",
  "tts",
]);

const TOOL_TAGS = new Set([
  "tool-use",
  "tools",
  "tool-calling",
  "function-calling",
  "function-call",
  "agent",
  "agentic",
]);

const REASONING_TAGS = new Set([
  "reasoning",
  "thinking",
  "chain-of-thought",
  "math-reasoning",
  "step-by-step",
]);

// Image and video generation (one "diffusion" capability; the Hub filter splits it, generationKinds).
const DIFFUSION_TAGS = new Set([
  "diffusers",
  "diffusion",
  "stable-diffusion",
  "latent-diffusion",
  "flux",
  "text-to-image",
  "image-to-image",
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
  "unconditional-image-generation",
]);

// Text-diffusion language models (LLaDA, Dream, SDAR, DiffusionGemma, ...) are text generators that
// can carry a bare "diffusion" tag; only an image / video tag or family name makes them image gen.
const TEXT_DIFFUSION_LM_TAGS = new Set([
  "llada",
  "llada2_moe",
  "dream",
  "sdar",
  "nemotron_labs_diffusion",
  "diffusion_gemma",
  "diffusion_gemma4",
]);
const TEXT_DIFFUSION_LM_ID_RE =
  /(?:^|[-_/])(?:llada|dream[-_]v\d|diffucoder|sdar[-_]\d|diffusiongemma|nemotron[-_]labs[-_]diffusion)/;

const CODE_TAGS = new Set([
  "code",
  "code-generation",
  "code-completion",
  "code-llm",
  "programming",
]);

const CONVERSATIONAL_TAGS = new Set([
  "conversational",
  "chat",
  "chatbot",
  "text-generation-inference",
]);

const CONVERSATIONAL_ID_RE = /(?:^|[-_/])conversational(?:[-_/]|$)/;

const COMMON_LANGUAGE_TAGS = new Set([
  "en",
  "fr",
  "de",
  "es",
  "it",
  "pt",
  "nl",
  "ru",
  "pl",
  "uk",
  "tr",
  "cs",
  "sv",
  "no",
  "da",
  "fi",
  "el",
  "ro",
  "bg",
  "hu",
  "hr",
  "sr",
  "sl",
  "sk",
  "zh",
  "ja",
  "ko",
  "ar",
  "he",
  "fa",
  "hi",
  "bn",
  "ta",
  "te",
  "th",
  "vi",
  "id",
  "ms",
  "tl",
  "sw",
  "ur",
  "ca",
  "eu",
  "is",
  "lv",
  "lt",
  "et",
]);

function lower(values: string[] | undefined): Set<string> {
  if (!values) return new Set();
  return new Set(values.map((v) => v.toLowerCase()));
}

export function detectCapabilities(
  tags: string[] | undefined,
  pipelineTag: string | undefined,
  modelId: string,
): Capability[] {
  const tagSet = lower(tags);
  const pipeline = pipelineTag?.toLowerCase();
  const lowerId = modelId.toLowerCase();
  const out: Capability[] = [];

  const hasAny = (set: ReadonlySet<string>) => {
    for (const t of set) {
      if (tagSet.has(t)) return true;
    }
    return pipeline ? set.has(pipeline) : false;
  };

  if (
    hasAny(VISION_TAGS) ||
    /[-_/]vl[-_]|vision|multimodal|llava|kosmos|qwen2[-_]?vl|cogvlm/.test(
      lowerId,
    )
  ) {
    out.push({ key: "vision", label: "Vision" });
  }
  if (
    hasAny(AUDIO_TAGS) ||
    /audio|whisper|speech|tts|stt|asr|parler|sesame|bark|wav2vec|hubert|musicgen|speecht5|vocos|encodec|clap|ultravox|kyutai|moshi/.test(
      lowerId,
    )
  ) {
    out.push({ key: "audio", label: "Audio" });
  }
  if (hasAny(TOOL_TAGS) || /tool[-_]use|function[-_]call/.test(lowerId)) {
    out.push({ key: "tools", label: "Tool use" });
  }
  if (
    hasAny(REASONING_TAGS) ||
    /think|reason|qwq|deepseek[-_]?r|gpt[-_]?oss|(?:^|[-_/])o1(?:[-_]|$)|nemotron[-_]reasoning/.test(
      lowerId,
    )
  ) {
    out.push({ key: "reasoning", label: "Reasoning" });
  }
  if (
    hasAny(CODE_TAGS) ||
    /(?:^|[-_/])coder(?:[-_/]|$)|(?:^|[-_/])code(?:[-_]|$)|starcoder|codellama|deepseek[-_]coder/.test(
      lowerId,
    )
  ) {
    out.push({ key: "code", label: "Code" });
  }
  if (hasAny(CONVERSATIONAL_TAGS) || CONVERSATIONAL_ID_RE.test(lowerId)) {
    out.push({ key: "conversational", label: "Conversational" });
  }
  if (
    hasAny(EMBEDDING_TAGS) ||
    /embed|embedding|retriever|reranker|bge[-_]|e5[-_]|gte[-_]|colbert|sentence[-_]?transformer|sentence[-_]?similarity|jina[-_]?embeddings?|nomic[-_]?embed|arctic[-_]?embed|qwen3[-_]?embedding|qwen3[-_]?reranker|text[-_]?embedding/.test(
      lowerId,
    )
  ) {
    out.push({ key: "embedding", label: "Embeddings" });
  }
  const imageGenName =
    /stable[-_]?diffusion|\bsdxl\b|\bflux\b|qwen[-_]?image|hunyuan[-_]?(?:video|image)|wan2|latent[-_]?consistency|[-_]lcm\b|dreamshaper/.test(
      lowerId,
    );
  const textDiffusionLm =
    !imageGenName &&
    (hasAny(TEXT_DIFFUSION_LM_TAGS) || TEXT_DIFFUSION_LM_ID_RE.test(lowerId));
  const imageGenTagged = textDiffusionLm
    ? [...DIFFUSION_TAGS].some(
        (tag) => tag !== "diffusion" && (tagSet.has(tag) || pipeline === tag),
      )
    : hasAny(DIFFUSION_TAGS);
  const kinds = generationKinds(tags, pipelineTag, modelId);
  // A Diffusers audio or video-to-video pipeline makes neither, so it gets no generation pill.
  if ((imageGenTagged || imageGenName) && (kinds.image || kinds.video)) {
    out.push({
      key: "diffusion",
      label: kinds.video ? "Video gen" : "Image gen",
    });
  }
  const languageCodes = new Set<string>();
  for (const tag of tags ?? []) {
    const lower = tag.toLowerCase();
    const code = lower.startsWith("language:")
      ? lower.slice("language:".length)
      : lower;
    if (COMMON_LANGUAGE_TAGS.has(code)) languageCodes.add(code);
  }
  if (tagSet.has("multilingual") || languageCodes.size >= 3) {
    out.push({ key: "multilingual", label: "Multilingual" });
  }
  return out;
}

export function detectLicense(tags: string[] | undefined): string | null {
  if (!tags) return null;
  const license = tags.find((t) => t.startsWith("license:"));
  return license ? license.slice("license:".length) : null;
}

export function detectBaseModel(tags: string[] | undefined): string | null {
  if (!tags) {
    return null;
  }
  let quantizedBaseModel: string | null = null;

  for (const tag of tags) {
    if (!tag.startsWith("base_model:")) {
      continue;
    }
    const value = tag.slice("base_model:".length).trim();
    if (!value) {
      continue;
    }
    if (value.startsWith("quantized:")) {
      quantizedBaseModel ??= value.slice("quantized:".length).trim() || null;
      continue;
    }
    return value;
  }

  return quantizedBaseModel;
}
