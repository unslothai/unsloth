// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ApiMonitorEntry } from "../types/api";

export type LiveChatTpsSample = {
  running: boolean;
  tps: number | null;
};

export function readLiveChatTpsSample(
  entry: ApiMonitorEntry,
  monitorId: string,
): LiveChatTpsSample {
  if (entry.id !== monitorId || entry.status !== "running") {
    return { running: false, tps: null };
  }
  const tps = entry.tok_per_sec;
  return {
    running: true,
    tps:
      typeof tps === "number" && Number.isFinite(tps) && tps >= 0
        ? tps
        : null,
  };
}

export function formatLiveChatTps(tps: number | null): string {
  return tps === null ? "—" : tps.toFixed(1);
}

export function visibleLiveChatTps(
  phase: "running" | "terminal" | undefined,
  lastRunningTps: number | null,
): number | null {
  return phase === "running" ? lastRunningTps : null;
}
