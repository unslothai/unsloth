// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { isAbortError, parseSseMessage } from "@/lib/sse-parser";

const readError = (r: Response): Promise<string> => readFastApiError(r);

export class BenchmarkRequestError extends Error {
  status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "BenchmarkRequestError";
    this.status = status;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new BenchmarkRequestError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

export function isRecoverableTransportError(err: unknown): boolean {
  if (err instanceof BenchmarkRequestError) {
    return typeof err.status === "number" ? err.status >= 502 : true;
  }
  // authFetch tags a dropped connection this way; the run may still finish server-side.
  return (
    (err as { unslothTransportFailure?: unknown } | null)
      ?.unslothTransportFailure === true
  );
}

export interface CheckpointInfo {
  display_name: string;
  path: string;
  loss?: number | null;
}

export interface ModelCheckpoints {
  name: string;
  checkpoints: CheckpointInfo[];
  base_model?: string | null;
  peft_type?: string | null;
  lora_rank?: number | null;
  is_quantized?: boolean;
}

export interface CheckpointListResponse {
  outputs_dir: string;
  models: ModelCheckpoints[];
}

export interface BenchmarkOperationResponse {
  success: boolean;
  message: string;
  details?: { output_path?: string | null } & Record<string, unknown>;
}

export interface BenchmarkStatus {
  is_benchmark_active: boolean;
  last_op_seq?: number;
  last_op_status?: "success" | "error" | "cancelled" | null;
  last_op_error?: string | null;
  /** Present when the status was asked with a run token; absent from older servers. */
  run_token_matches?: boolean | null;
}

export interface BenchmarkTaskInfo {
  id: string;
  name: string;
  description?: string;
  task_type?: string;
  runner?: "lm_eval" | "recipe";
}

export interface BenchmarkTasksResponse {
  tasks: BenchmarkTaskInfo[];
}

export async function fetchBenchmarkTasks(): Promise<BenchmarkTasksResponse> {
  const response = await authFetch("/api/benchmarks/evals/tasks");
  const data = await parseJson<BenchmarkTasksResponse>(response);
  return { ...data, tasks: data.tasks.filter((t) => t.runner !== "recipe") };
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const response = await authFetch("/api/models/checkpoints");
  return parseJson<CheckpointListResponse>(response);
}

export interface BenchmarkTaskConfig {
  task_id: string;
  num_fewshot: number | null;
}

export async function fetchBenchmarkTaskConfig(taskId: string): Promise<BenchmarkTaskConfig> {
  const response = await authFetch(`/api/benchmarks/evals/task/${encodeURIComponent(taskId)}/config`);
  return parseJson<BenchmarkTaskConfig>(response);
}

export interface BenchmarkRunParams {
  checkpoint_path: string;
  model_source: "checkpoint" | "hf" | "local";
  hf_token?: string | null;
  task?: string;
  batch_size?: string;
  num_fewshot?: number | null;
  max_tokens?: number | null;
  expected_model?: string | null;
  run_token?: string;
}

export async function runBenchmark(params: BenchmarkRunParams): Promise<BenchmarkOperationResponse> {
  const response = await authFetch("/api/benchmarks/evals/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<BenchmarkOperationResponse>(response);
}

export async function cancelBenchmark(): Promise<BenchmarkOperationResponse> {
  const response = await authFetch("/api/benchmarks/evals/cancel", { method: "POST" });
  return parseJson<BenchmarkOperationResponse>(response);
}

export async function getBenchmarkStatus(runToken?: string): Promise<BenchmarkStatus> {
  const query = runToken ? `?run_token=${encodeURIComponent(runToken)}` : "";
  const response = await authFetch(`/api/benchmarks/evals/status${query}`);
  return parseJson<BenchmarkStatus>(response);
}

export type BenchmarkLogStream = "stdout" | "stderr" | "status" | "progress";

export interface BenchmarkLogEntry {
  stream: BenchmarkLogStream;
  line: string;
  ts: number | null;
}


export interface BenchmarkRunMetric {
  name: string;
  score: number;
  stderr?: string | null;
}

export interface BenchmarkRunSummary {
  id: string;
  task: string;
  model: string;
  metrics: BenchmarkRunMetric[];
  default_metric: string;
  n_samples: number;
  num_fewshot?: number | null;
  created_at: string;
  output_path: string;
}

export interface BenchmarkRunListResponse {
  runs: BenchmarkRunSummary[];
}

export interface BenchmarkSampleResult {
  doc_id: number;
  question: string;
  target: string;
  response?: string | null;
  raw_response?: string | null;
  correct: boolean;
}

export interface BenchmarkRunDetail extends BenchmarkRunSummary {
  samples: BenchmarkSampleResult[];
  correct_count: number;
  total_count: number;
}

export async function listBenchmarkRuns(): Promise<BenchmarkRunListResponse> {
  const response = await authFetch("/api/benchmarks/evals/runs");
  return parseJson<BenchmarkRunListResponse>(response);
}

export async function getBenchmarkRunDetail(runId: string): Promise<BenchmarkRunDetail> {
  const response = await authFetch(`/api/benchmarks/evals/runs/${encodeURIComponent(runId)}`);
  return parseJson<BenchmarkRunDetail>(response);
}

export async function deleteBenchmarkRun(runId: string): Promise<BenchmarkOperationResponse> {
  const response = await authFetch(`/api/benchmarks/evals/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  return parseJson<BenchmarkOperationResponse>(response);
}


export async function exportBenchmarkRuns(runIds: string[]): Promise<void> {
  const response = await authFetch("/api/benchmarks/evals/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds }),
  });
  if (!response.ok) {
    throw new BenchmarkRequestError(await readError(response), response.status);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "benchmark-export.json";
  a.click();
  URL.revokeObjectURL(url);
}


export interface BenchmarkGraphParams {
  runIds: string[];
  chartType: "bar" | "line" | "grouped_bar" | "radar";
  metric: string;
  width: number;
  height: number;
  theme: string;
}

export async function generateBenchmarkGraph(params: BenchmarkGraphParams): Promise<Blob> {
  const response = await authFetch("/api/benchmarks/evals/graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_ids: params.runIds,
      chart_type: params.chartType,
      metric: params.metric,
      width: params.width,
      height: params.height,
      theme: params.theme,
    }),
  });
  if (!response.ok) {
    throw new BenchmarkRequestError(await readError(response), response.status);
  }
  return response.blob();
}


export type BenchmarkLogEventName = "log" | "heartbeat" | "complete" | "error";

export interface BenchmarkLogEvent {
  event: BenchmarkLogEventName;
  id: number | null;
  entry?: BenchmarkLogEntry;
  error?: string;
  active?: boolean;
}



export async function streamBenchmarkLogs(options: {
  signal: AbortSignal;
  since?: number | null;
  onOpen?: () => void;
  onEvent: (event: BenchmarkLogEvent) => void;
}): Promise<void> {
  const headers = new Headers();
  if (typeof options.since === "number") {
    headers.set("Last-Event-ID", String(options.since));
  }

  const url =
    typeof options.since === "number"
      ? `/api/benchmarks/evals/logs/stream?since=${options.since}`
      : "/api/benchmarks/evals/logs/stream";

  const response = await authFetch(url, {
    method: "GET",
    headers,
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }
  if (!response.body) {
    throw new Error("Benchmark log stream unavailable");
  }

  options.onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;

      buffer += decoder.decode(value, { stream: true });

      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
        buffer = buffer.slice(separatorIndex + separatorLength);

        if (rawEvent.startsWith("retry:") || rawEvent.startsWith(":")) {
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }

        const parsed = parseSseMessage(rawEvent);
        if (!parsed) {
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }

        try {
          if (parsed.event === "log") {
            const payload = JSON.parse(parsed.data) as {
              stream?: BenchmarkLogStream;
              line?: string;
              ts?: number | null;
            };
            options.onEvent({
              event: "log",
              id: parsed.id,
              entry: {
                stream: payload.stream ?? "stdout",
                line: payload.line ?? "",
                ts: payload.ts ?? null,
              },
            });
          } else if (parsed.event === "heartbeat") {
            let active: boolean | undefined;
            try {
              const payload = JSON.parse(parsed.data) as { active?: boolean };
              if (typeof payload.active === "boolean") active = payload.active;
            } catch {
            }
            options.onEvent({ event: "heartbeat", id: parsed.id, active });
          } else if (parsed.event === "progress") {
            options.onEvent({
              event: "log",
              id: null,
              entry: {
                stream: "progress",
                line: parsed.data,
                ts: null,
              },
            });
          } else if (parsed.event === "complete") {
            options.onEvent({ event: "complete", id: parsed.id });
          } else if (parsed.event === "error") {
            let errorMessage = "Benchmark log stream error";
            try {
              const payload = JSON.parse(parsed.data) as { error?: string };
              if (payload.error) errorMessage = payload.error;
            } catch {
            }
            options.onEvent({
              event: "error",
              id: parsed.id,
              error: errorMessage,
            });
          }
        } catch (err) {
          if (isAbortError(err)) return;
        }

        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
  } catch (err) {
    if (isAbortError(err)) return;
    throw err;
  } finally {
    try {
      await reader.cancel();
    } catch {
    }
  }
}
