// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeStatus } from "../types";

export type BenchmarkState =
  | "needs_rows"
  | "ready"
  | "scoring"
  | "scored"
  | "stopped"
  | "failed";

export const BENCHMARK_STATE_LABEL: Record<BenchmarkState, string> = {
  needs_rows: "Needs rows",
  ready: "Ready",
  scoring: "Scoring",
  scored: "Scored",
  stopped: "Stopped",
  failed: "Failed",
};

export const RECIPE_TASK = "Recipe held-out";

export interface BenchmarkResult {
  score: number | null;
  passed: number;
  scored: number;
  baseScore: number | null;
  metric: string;
  runId: string | null;
}

export interface BenchmarkView {
  state: BenchmarkState;
  detail: string | null;
  result: BenchmarkResult | null;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function benchmarkResult(outputs: Record<string, unknown> | null): BenchmarkResult | null {
  const r = outputs?.results as Record<string, unknown> | undefined;
  if (!r || typeof r !== "object") return null;
  return {
    score: num(outputs?.score) ?? num(r.score),
    passed: num(r.passed) ?? 0,
    scored: num(r.scored) ?? 0,
    baseScore: num(r.base_score),
    metric: typeof r.metric === "string" ? r.metric : "",
    runId: typeof r.run_id === "string" ? r.run_id : null,
  };
}

export function percent(score: number | null): string {
  return score === null ? "-" : `${Math.round(score * 100)}%`;
}

export function benchmarkView(input: {
  status: NodeStatus;
  params: Record<string, unknown>;
  hasRows: boolean;
  outputs: Record<string, unknown> | null;
  phase: string | null;
  error: string | null;
}): BenchmarkView {
  const { status } = input;
  if (["queued", "waiting", "loading_model", "running"].includes(status))
    return {
      state: "scoring",
      detail: status === "loading_model" ? "Loading the model" : input.phase,
      result: null,
    };
  if (status === "failed") return { state: "failed", detail: input.error, result: null };
  if (status === "cancelled") return { state: "stopped", detail: null, result: null };
  const result = benchmarkResult(input.outputs);
  if ((status === "done" || status === "cached") && result)
    return { state: "scored", detail: null, result };
  const task = String(input.params.task ?? RECIPE_TASK);
  if (task === RECIPE_TASK && !input.hasRows)
    return { state: "needs_rows", detail: "Connect held-out rows or a recipe", result: null };
  return { state: "ready", detail: null, result: null };
}
