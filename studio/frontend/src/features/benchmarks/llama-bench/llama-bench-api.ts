// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// llama-bench runs server-side under /api/benchmarks/llama-bench; finished runs are saved
// with the config sweeps as kind "llama-bench".

import { authFetch } from "@/features/auth";

export interface LlamaBenchConfig {
  prompt_tokens: number[];
  gen_tokens: number[];
  depths: number[];
  repetitions: number;
  flash_attn: "auto" | "on" | "off";
  n_gpu_layers?: number | null;
}

export interface LlamaBenchRow {
  test: string;
  n_prompt: number;
  n_gen: number;
  n_depth: number;
  avg_ts: number;
  stddev_ts: number;
  samples_ts: number[];
  n_gpu_layers?: number | null;
  flash_attn?: number | boolean | string | null;
}

export interface LlamaBenchMeta {
  build_commit?: string | null;
  build_number?: number | null;
  gpu_info?: string | null;
  backends?: string | null;
  model_type?: string | null;
  model_size?: number | null;
  model_n_params?: number | null;
}

export type LlamaBenchStatus = "running" | "done" | "error" | "cancelled";

export interface LlamaBenchJob {
  id: string;
  status: LlamaBenchStatus;
  stage: string;
  error: string | null;
  model: string;
  ggufVariant: string | null;
  config: LlamaBenchConfig;
  rows: LlamaBenchRow[];
  meta: LlamaBenchMeta;
  done: number;
  total: number;
  log: string[];
  createdAt: number;
  finishedAt: number | null;
}

/** A saved run as /runs lists it; the rows ride in `outcomes`. */
export interface SavedLlamaBenchRun {
  id: string;
  model: string;
  ggufVariant: string | null;
  config: LlamaBenchConfig;
  meta: LlamaBenchMeta;
  outcomes: LlamaBenchRow[];
  createdAt: number;
  finishedAt: number | null;
}

async function parse<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : typeof detail?.message === "string"
          ? detail.message
          : `${what} failed (${res.status})`;
    throw new Error(message);
  }
  return (await res.json()) as T;
}

const BASE = "/api/benchmarks/llama-bench";

export async function getLlamaBenchStatus(signal?: AbortSignal): Promise<{
  available: boolean;
  model: string | null;
  ggufVariant: string | null;
  job: LlamaBenchJob | null;
}> {
  return parse(await authFetch(`${BASE}/status`, { signal }), "Reading llama-bench");
}

export async function startLlamaBench(
  config: LlamaBenchConfig,
): Promise<LlamaBenchJob> {
  const res = await authFetch(`${BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return parse(res, "Starting llama-bench");
}

export async function getLlamaBenchJob(
  signal?: AbortSignal,
): Promise<LlamaBenchJob | null> {
  const res = await authFetch(`${BASE}/run`, { signal });
  return (await parse<{ job: LlamaBenchJob | null }>(res, "Reading llama-bench"))
    .job;
}

export async function cancelLlamaBench(): Promise<void> {
  await authFetch(`${BASE}/run`, { method: "DELETE" });
}

export async function listLlamaBenchRuns(
  signal?: AbortSignal,
): Promise<SavedLlamaBenchRun[]> {
  const res = await authFetch("/api/benchmarks/runs?kind=llama-bench", {
    signal,
  });
  return (await parse<{ runs: SavedLlamaBenchRun[] }>(res, "Listing runs"))
    .runs;
}

export async function deleteLlamaBenchRun(id: string): Promise<void> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404)
    throw new Error(`Deleting the run failed (${res.status})`);
}
