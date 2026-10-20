// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { audioCppModelFor } from "./audio-cpp-catalog";
import { rowOnPage } from "./page-models";
import type { AudioWorkflowId } from "./workflows";

export interface AudioPickerRow {
  id?: string | null;
  task?: string | null;
  audioType?: string | null;
  audioWorkflows?: readonly string[] | null;
}

/** Voice mode's speech check is a picker workflow with no Audio page of its own. */
export type AudioPickerWorkflow = AudioWorkflowId | "detect";

const VAD_HUB_TASK = "voice-activity-detection";

export function audioRowMatchesWorkflow(
  row: AudioPickerRow,
  workflow: AudioPickerWorkflow,
): boolean {
  if (row.task === VAD_HUB_TASK || audioCppModelFor(row.id)?.task === "vad") {
    return workflow === "detect";
  }
  if (workflow === "detect") return false;
  return rowOnPage(workflow, { ...row, id: row.id ?? "" }, { isMac: false });
}
