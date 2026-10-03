// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { audioCppModelFor } from "./audio-cpp-catalog";
import { isMusicGenerationModel } from "./catalog";
import {
  type AudioWorkflowId,
  audioWorkflowForAudioType,
  audioWorkflowForTask,
} from "./workflows";

export interface AudioPickerRow {
  id?: string | null;
  task?: string | null;
  audioType?: string | null;
  audioWorkflows?: readonly string[] | null;
}

export function audioRowMatchesWorkflow(
  row: AudioPickerRow,
  workflow: AudioWorkflowId,
): boolean {
  if (row.audioWorkflows && row.audioWorkflows.length > 0) {
    return row.audioWorkflows.includes(workflow);
  }
  // MiniMax Music 3 is tagged text-to-speech on the Hub.
  if (isMusicGenerationModel(row.id, row.audioType)) return workflow === "music";
  const byTask = audioWorkflowForTask(row.task);
  if (byTask === "transcribe" || byTask === "music") {
    return byTask === workflow;
  }
  const catalogTask = audioCppModelFor(row.id)?.task;
  if (catalogTask === "music") {
    return workflow === "music";
  }
  if (catalogTask === "asr") {
    return workflow === "transcribe";
  }
  if (byTask === "speak" || catalogTask === "tts" || row.audioType) {
    return audioWorkflowForAudioType(row.audioType) === workflow;
  }
  return true;
}
