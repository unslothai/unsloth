// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type HubModelTaskMetadata = {
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
};

// These generation tasks belong to the Audio page and cannot serve coding-agent
// chat completions. Keep this narrower than generic audio tags so audio-capable
// chat and vision-language models remain available.
const SPEECH_ONLY_TAGS: ReadonlySet<string> = new Set([
  "text-to-speech",
  "automatic-speech-recognition",
]);

export function isSpeechOnlyHubModel(model: HubModelTaskMetadata): boolean {
  return [model.pipelineTag, ...(model.tags ?? [])].some((tag) => {
    if (tag == null) {
      return false;
    }
    return SPEECH_ONLY_TAGS.has(tag.toLowerCase().trim());
  });
}
