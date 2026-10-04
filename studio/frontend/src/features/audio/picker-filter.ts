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

// Clone-only audio.cpp families outside AUDIO_CPP_MODELS (the backend's clone-only list).
const CLONE_ONLY_FAMILY_HINT =
  /miotts|vevo-?2|fireredtts-?3|firered-?audio|indextts-?2[._-]?5|confucius-?4/i;

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
  const catalogModel = audioCppModelFor(row.id);
  if (catalogModel?.workflows) {
    return catalogModel.workflows.includes(workflow);
  }
  // Hub search rows carry no backend workflows before download: name the clone-only families.
  if (CLONE_ONLY_FAMILY_HINT.test(row.id ?? "")) {
    return workflow === "clone";
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
