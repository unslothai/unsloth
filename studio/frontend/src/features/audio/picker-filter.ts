// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { rowOnPage } from "./page-models";
import type { AudioWorkflowId } from "./workflows";

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
  return rowOnPage(workflow, { ...row, id: row.id ?? "" }, { isMac: false });
}
