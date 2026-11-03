// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkLogEntry } from "../api/benchmark-api";

export const MAX_LOG_LINES = 4000;

export interface BenchmarkProgress {
  pct: number;
  current: number;
  total: number;
  /** ``tqdm.format_interval`` text; null before a rate is known. */
  elapsed: string | null;
  eta: string | null;
}

export interface LogState {
  logLines: BenchmarkLogEntry[];
  lastSeq: number | null;
  stage: string | null;
  progressPercent: number;
  progressDetail: BenchmarkProgress | null;
}

function parseProgress(line: string): { pct: number; detail: BenchmarkProgress | null } {
  try {
    const parsed = JSON.parse(line);
    return { pct: parsed.pct, detail: parsed };
  } catch {
    const pct = Number(line);
    return { pct: Number.isFinite(pct) ? pct : 0, detail: null };
  }
}

/** The state after one stream event, or ``state`` itself when nothing changes. Progress
 * events only move the bar: they arrive several times a second and would flush the log. */
export function applyLogEntry<S extends LogState>(
  state: S,
  entry: BenchmarkLogEntry,
  seq?: number,
): S | Partial<LogState> {
  if (typeof seq === "number" && state.lastSeq !== null && seq <= state.lastSeq) {
    return state;
  }
  if (entry.stream === "progress") {
    const parsed = parseProgress(entry.line);
    return { progressPercent: parsed.pct, progressDetail: parsed.detail };
  }
  const next =
    state.logLines.length >= MAX_LOG_LINES
      ? state.logLines.slice(state.logLines.length - MAX_LOG_LINES + 1)
      : state.logLines.slice();
  next.push(entry);
  return {
    logLines: next,
    lastSeq: typeof seq === "number" ? seq : state.lastSeq,
    stage: entry.stream === "status" ? entry.line : state.stage,
  };
}
