// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers that infer what a model can do (vision / reasoning / audio / media generation)
// from its HF tags and pipeline tag, falling back to repo-name keywords. No React/DOM deps.

export interface ModelCapabilities {
  vision: boolean;
  reasoning: boolean;
  audio: boolean;
  /** Generates images. `vision` is the opposite direction: reading them. */
  imageGen: boolean;
  /** Generates video. */
  videoGen: boolean;
}

// Authoritative HF pipeline tags / tags for each capability.
const VISION_TAGS = new Set([
  "image-text-to-text",
  "image-to-text",
  "visual-question-answering",
  "video-text-to-text",
  "any-to-any",
  "multimodal",
  "vision",
]);
const AUDIO_TAGS = new Set([
  "automatic-speech-recognition",
  "audio-text-to-text",
  "text-to-speech",
  "text-to-audio",
  "audio-to-audio",
  "audio-classification",
]);
const REASONING_TAGS = new Set(["reasoning"]);
const IMAGE_GEN_TAGS = new Set([
  "text-to-image",
  "image-to-image",
  // The Images picker's own IMAGE_GEN_TASKS accepts this, so a row it lists has to draw the glyph.
  "image-text-to-image",
  "inpainting",
]);
// Every tag here must be one the Video page can run, or a row draws a glyph promising something
// no page delivers. video-to-video is deliberately absent: the backend takes a prompt and
// reference IMAGES (routes/video.py), never a source video.
const VIDEO_GEN_TAGS = new Set([
  "text-to-video",
  "image-to-video",
  // What MiniMax-H3 is tagged with on the Hub: a frame plus a prompt in, video out.
  "image-text-to-video",
]);

// Repo-name fallbacks, bounded so we never read a token out of a longer word.
const SEP = "(?:^|[-_/. ])";
const END = "(?=$|[-_/. ])";
const VISION_NAME_RE = new RegExp(
  `${SEP}(?:vl|llava|pixtral|moondream|smolvlm|internvl|cogvlm|idefics|paligemma|vision)${END}`,
  "i",
);
const REASONING_NAME_RE = new RegExp(
  `${SEP}(?:r1|qwq|thinking|reason(?:ing|er)?|magistral|o1|marco)${END}`,
  "i",
);
const AUDIO_NAME_RE = new RegExp(
  `${SEP}(?:whisper|asr|tts|parakeet|parler|musicgen|bark|orpheus|csm|voice|speech|audio)${END}`,
  "i",
);
// Families, not the word "image": a local GGUF carries no tags, so "Z-Image-Turbo-GGUF" has to
// read as an image generator from its name alone. Video is matched FIRST below, since
// "HunyuanVideo" and "hunyuanimage" share a stem and the video families are narrower. These
// end on a letter boundary rather than END, since a family stem runs into its version
// ("flux1", "sd3"); without it "fluxion-7b" and "SVDQuant" read as generators.
const FAMILY_END = "(?![a-z])";
const IMAGE_GEN_NAME_RE = new RegExp(
  `${SEP}(?:flux|sdxl|sd3|stable[-_]?diffusion|z[-_]?image|qwen[-_]?image|hidream|ideogram|lumina|hunyuanimage|krea|kolors|playground|pixart)${FAMILY_END}`,
  "i",
);
const VIDEO_GEN_NAME_RE = new RegExp(
  // cogvideox keeps its trailing letter: it is part of the family name, not another word.
  `${SEP}(?:wan\\d|ltx|hunyuanvideo|minimax[-_]?h\\d|cogvideox?|mochi|animatediff|svd|zeroscope)${FAMILY_END}`,
  "i",
);

function hasAny(tagSet: Set<string>, wanted: Set<string>): boolean {
  for (const tag of wanted) if (tagSet.has(tag)) return true;
  return false;
}

/** Infer capabilities from HF tags + pipeline tag, then repo-name keywords. */
export function detectCapabilities(opts: {
  id: string;
  tags?: readonly string[];
  pipelineTag?: string;
}): ModelCapabilities {
  const { id, tags, pipelineTag } = opts;
  const tagSet = new Set((tags ?? []).map((t) => t.toLowerCase()));
  if (pipelineTag) tagSet.add(pipelineTag.toLowerCase());
  // Name part only. These two match model FAMILIES, and a family word turns up in owners too:
  // "hunyuanvideo-community/HunyuanImage-2.1" is an image model published by a video org.
  const name = id.split("/").pop() ?? id;
  const videoGen = hasAny(tagSet, VIDEO_GEN_TAGS) || VIDEO_GEN_NAME_RE.test(name);
  return {
    vision: hasAny(tagSet, VISION_TAGS) || VISION_NAME_RE.test(id),
    reasoning: hasAny(tagSet, REASONING_TAGS) || REASONING_NAME_RE.test(id),
    audio: hasAny(tagSet, AUDIO_TAGS) || AUDIO_NAME_RE.test(id),
    // A video model is not also an image model. Several ship a text-to-image tag for their
    // first-frame path, and both badges on one row says less than the video one alone.
    imageGen:
      !videoGen && (hasAny(tagSet, IMAGE_GEN_TAGS) || IMAGE_GEN_NAME_RE.test(name)),
    videoGen,
  };
}
