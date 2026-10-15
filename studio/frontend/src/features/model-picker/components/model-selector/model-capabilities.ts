// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ModelCapabilities {
  vision: boolean;
  reasoning: boolean;
  audio: boolean;
  /** Generates images; `vision` is reading them. */
  imageGen: boolean;
  videoGen: boolean;
}

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
  // The Images picker accepts this tag, so its rows must draw the glyph.
  "image-text-to-image",
  "inpainting",
]);
// Only tags the Video page can run; video-to-video is absent (backend takes reference images).
const VIDEO_GEN_TAGS = new Set([
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
]);

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
// Local GGUFs have no tags, so match family names. Video matches first (shared stems); a letter
// boundary, not END, since stems run into versions ("flux1") but not "fluxion".
const FAMILY_END = "(?![a-z])";
const IMAGE_GEN_NAME_RE = new RegExp(
  `${SEP}(?:flux|sdxl|sd3|stable[-_]?diffusion|z[-_]?image|qwen[-_]?image|hidream|ideogram|lumina|hunyuanimage|krea|kolors|playground|pixart)${FAMILY_END}`,
  "i",
);
const VIDEO_GEN_NAME_RE = new RegExp(
  `${SEP}(?:wan\\d|ltx|hunyuanvideo|minimax[-_]?h\\d|cogvideox?|mochi|animatediff|svd|zeroscope)${FAMILY_END}`,
  "i",
);

function hasAny(tagSet: Set<string>, wanted: Set<string>): boolean {
  for (const tag of wanted) if (tagSet.has(tag)) return true;
  return false;
}

export function detectCapabilities(opts: {
  id: string;
  tags?: readonly string[];
  pipelineTag?: string;
}): ModelCapabilities {
  const { id, tags, pipelineTag } = opts;
  const tagSet = new Set((tags ?? []).map((t) => t.toLowerCase()));
  if (pipelineTag) tagSet.add(pipelineTag.toLowerCase());
  // Name part only: family words appear in owners too (hunyuanvideo-community).
  const name = id.split("/").pop() ?? id;
  const videoGen = hasAny(tagSet, VIDEO_GEN_TAGS) || VIDEO_GEN_NAME_RE.test(name);
  return {
    vision: hasAny(tagSet, VISION_TAGS) || VISION_NAME_RE.test(id),
    reasoning: hasAny(tagSet, REASONING_TAGS) || REASONING_NAME_RE.test(id),
    audio: hasAny(tagSet, AUDIO_TAGS) || AUDIO_NAME_RE.test(id),
    // A video model is not also an image model, despite first-frame text-to-image tags.
    imageGen:
      !videoGen && (hasAny(tagSet, IMAGE_GEN_TAGS) || IMAGE_GEN_NAME_RE.test(name)),
    videoGen,
  };
}
