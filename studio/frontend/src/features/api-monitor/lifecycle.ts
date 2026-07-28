// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Labels for model load/unload/download rows, shared by the overlay and the
// full page.
//
// They live here rather than on the page because the overlay is mounted from
// __root.tsx, so importing them from the page would pull the whole page and its
// dependency graph into the eagerly loaded bundle and undo the route's
// lazyRouteComponent: every route would pay for the monitor page even when it
// is never opened.

import type { ApiMonitorEntry } from "@/features/chat/types/api";

// A lifecycle row is a model load/unload/download, not an HTTP call: it carries
// an event and reason instead of a prompt, so there is no payload to expand.
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
    // A cancel is deliberate, so saying it failed misreads the user's own action.
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
