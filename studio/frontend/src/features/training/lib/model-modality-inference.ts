


import { pathDisplayName } from "../../../components/resource-picker/path-display-name.ts";

export interface TrainingModelModalityMetadata {
  tags?: readonly string[] | null;
  pipelineTag?: string | null;
  identifiers?: readonly (string | null | undefined)[];
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
const SEP = "(?:^|[-_/. ])";
const END = "(?=$|[-_/. ])";
const VISION_NAME_RE = new RegExp(
  `${SEP}(?:vl|llava|pixtral|moondream|smolvlm|internvl|cogvlm|idefics|paligemma|vision)${END}`,
  "i",
);
const AUDIO_NAME_RE = new RegExp(
  `${SEP}(?:whisper|tts|parakeet|parler|musicgen|bark|orpheus|csm|voice|speech|audio)${END}`,
  "i",
);

function hasAny(tagSet: Set<string>, wanted: Set<string>): boolean {
  for (const tag of wanted) {
    if (tagSet.has(tag)) {
      return true;
    }
  }
  return false;
}

export function inferTrainingModelModalityFlags({
  tags,
  pipelineTag,
  identifiers = [],
}: TrainingModelModalityMetadata): {
  isAudio: boolean;
  isVision: boolean;
} {
  const tagSet = new Set((tags ?? []).map((tag) => tag.toLowerCase()));
  if (pipelineTag) {
    tagSet.add(pipelineTag.toLowerCase());
  }
  const names = identifiers
    .map((identifier) => identifier?.trim())
    .filter((identifier): identifier is string => Boolean(identifier))
    .map(pathDisplayName)
    .join(" ");
  return {
    isAudio: hasAny(tagSet, AUDIO_TAGS) || AUDIO_NAME_RE.test(names),
    isVision: hasAny(tagSet, VISION_TAGS) || VISION_NAME_RE.test(names),
  };
}
