// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Labels for model load/unload/download rows, shared by the overlay and the page. Their own
// module because the overlay mounts from __root.tsx: importing them from the page would pull it
// into the eager bundle and undo the route's lazyRouteComponent.

import type { ApiMonitorEntry } from "@/features/chat/types/api";

// A lifecycle row carries an event and reason instead of a prompt, so nothing expands.
export function isLifecycleEntry(entry: ApiMonitorEntry): boolean {
  return entry.kind === "lifecycle";
}

export function lifecycleLabel(entry: ApiMonitorEntry): string {
  if (entry.event === "unload") {
    return entry.reason === "idle" ? "Model unloaded (idle)" : "Model unloaded";
  }
  if (entry.event === "download") {
    if (entry.status === "running") {
      const pct = entry.progress;
      return typeof pct === "number"
        ? `Downloading model (${Math.round(pct)}%)`
        : "Downloading model";
    }
    if (entry.status === "completed") {
      return "Model downloaded";
    }
    // A cancel is deliberate, so calling it a failure misreads the user's action.
    return entry.status === "cancelled"
      ? "Model download cancelled"
      : "Model download failed";
  }
  if (entry.status === "running") {
    return "Loading model";
  }
  if (entry.status === "completed") {
    return "Model loaded";
  }
  return "Model load failed";
}
