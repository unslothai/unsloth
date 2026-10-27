// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  GpuRenderingSetting,
  GpuRenderingStatus,
} from "./api/gpu-rendering";

export type GpuRenderingNote =
  | { kind: "macNote" }
  | { kind: "envOverride"; name: string }
  | { kind: "fallbackNote" }
  | { kind: "oldWebkitWarning"; version: string }
  | { kind: "restartPending" };

/** Notes shown under the GPU rendering row, in display order. */
export function gpuRenderingNotes(
  status: GpuRenderingStatus,
  selected: GpuRenderingSetting,
): GpuRenderingNote[] {
  if (!status.configurable) return [{ kind: "macNote" }];
  const notes: GpuRenderingNote[] = [];
  if (status.envOverride) {
    notes.push({ kind: "envOverride", name: status.envOverride });
  }
  if (status.fallbackActive) notes.push({ kind: "fallbackNote" });
  if (selected === "on" && status.oldWebkitWarning) {
    notes.push({ kind: "oldWebkitWarning", version: status.oldWebkitWarning });
  }
  if (status.restartRequired) notes.push({ kind: "restartPending" });
  return notes;
}
