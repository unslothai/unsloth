// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers that infer what a model can do (vision / reasoning / audio) from
// its HF tags + pipeline tag, falling back to repo-name keywords. No React/DOM
// deps so they stay easy to test.

export interface ModelCapabilities {
  vision: boolean;
  reasoning: boolean;
  audio: boolean;
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
  `${SEP}(?:whisper|tts|parakeet|parler|musicgen|bark|orpheus|csm|voice|speech|audio)${END}`,
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
  return {
    vision: hasAny(tagSet, VISION_TAGS) || VISION_NAME_RE.test(id),
    reasoning: hasAny(tagSet, REASONING_TAGS) || REASONING_NAME_RE.test(id),
    audio: hasAny(tagSet, AUDIO_TAGS) || AUDIO_NAME_RE.test(id),
  };
}

/** True when at least one capability is present (worth rendering a badge). */
export function hasAnyCapability(caps: ModelCapabilities): boolean {
  return caps.vision || caps.reasoning || caps.audio;
}
