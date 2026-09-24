// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Runs live in studio.db beside chats and training runs, through /api/benchmarks.

import { authFetch } from "@/features/auth";
import type { BenchRun } from "../lib/bench-math";

/** A run as the list returns it: everything but the measurements. */
export type BenchRunSummary = Omit<BenchRun, "results"> & {
  resultCount: number;
};

async function parse<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `${what} failed (${res.status})${text ? `: ${text.slice(0, 200)}` : ""}`,
    );
  }
  return (await res.json()) as T;
}

export async function listBenchRuns(
  signal?: AbortSignal,
): Promise<BenchRunSummary[]> {
  const res = await authFetch("/api/benchmarks/runs", { signal });
  return (
    await parse<{ runs: BenchRunSummary[] }>(res, "Listing benchmark runs")
  ).runs;
}

export async function getBenchRun(
  id: string,
  signal?: AbortSignal,
): Promise<BenchRun> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(id)}`,
    { signal },
  );
  return parse<BenchRun>(res, "Reading the benchmark run");
}

export async function saveBenchRun(run: BenchRun): Promise<BenchRun> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(run.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "sweep", sweep: run.config.sweep, ...run }),
    },
  );
  return parse<BenchRun>(res, "Saving the benchmark run");
}

export async function deleteBenchRun(id: string): Promise<void> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404)
    throw new Error(`Deleting the benchmark run failed (${res.status})`);
}
