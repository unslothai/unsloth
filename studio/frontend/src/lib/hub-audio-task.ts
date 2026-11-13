// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PipelineType } from "@huggingface/hub";

// Audio pipeline tasks a picker's Search Hub link can narrow the Audio filter to.
const HUB_AUDIO_TASKS: readonly PipelineType[] = [
  "text-to-speech",
  "text-to-audio",
  "automatic-speech-recognition",
  "audio-to-audio",
];

/** The `audioTask` search param as the Hub keeps it: known audio tasks, only with the Audio filter. */
export function hubAudioTaskSearch(
  raw: unknown,
  capability: string | undefined,
): PipelineType[] | undefined {
  if (capability !== "audio") return undefined;
  const tasks = (Array.isArray(raw) ? raw : [raw]).filter(
    (task): task is PipelineType =>
      HUB_AUDIO_TASKS.includes(task as PipelineType),
  );
  return tasks.length > 0 ? tasks : undefined;
}

/** Whether a Hub row fits the linked audio tasks. A typed search drops the task server-side, so the
 *  rows are checked here; a row without a pipeline tag stays, as the task listing keeps it too. */
export function matchesHubAudioTask(
  pipelineTag: string | null | undefined,
  tasks: readonly PipelineType[] | null | undefined,
): boolean {
  return (
    !tasks ||
    tasks.length === 0 ||
    !pipelineTag ||
    tasks.includes(pipelineTag as PipelineType)
  );
}
