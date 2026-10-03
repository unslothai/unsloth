// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { audioCppModelFor, audioCppWorkflowsFor } from "./audio-cpp-catalog";
import {
  type AudioWorkflowId,
  audioWorkflowForAudioType,
  audioWorkflowForTask,
} from "./workflows";

/** The fields of a downloaded picker row the Audio page splits its list on. */
export interface AudioPickerRow {
  id?: string | null;
  task?: string | null;
  audioType?: string | null;
  audioWorkflows?: readonly string[] | null;
}

/** Whether a downloaded row belongs in this workflow's picker. The backend's `audio_workflows`
 *  answers when present; older rows fall back to the task, then the audio type (MiniMax Music 3
 *  is tagged text-to-speech) or the audio.cpp catalog entry. A row with nothing to go on stays. */
export function audioRowMatchesWorkflow(
  row: AudioPickerRow,
  workflow: AudioWorkflowId,
): boolean {
  if (row.audioWorkflows && row.audioWorkflows.length > 0) {
    return row.audioWorkflows.includes(workflow);
  }
  const byTask = audioWorkflowForTask(row.task);
  if (byTask === "transcribe" || byTask === "music") {
    return byTask === workflow;
  }
  const catalogModel = audioCppModelFor(row.id);
  if (catalogModel?.workflows) {
    return audioCppWorkflowsFor(catalogModel).includes(workflow);
  }
  const catalogTask = catalogModel?.task;
  if (catalogTask === "music") {
    return workflow === "music";
  }
  if (catalogTask === "asr") {
    return workflow === "transcribe";
  }
  if (byTask === "speak" || catalogTask === "tts" || row.audioType) {
    // Speech models clone only when the backend says so; an older row stays on Speak.
    return audioWorkflowForAudioType(row.audioType) === workflow;
  }
  return true;
}
