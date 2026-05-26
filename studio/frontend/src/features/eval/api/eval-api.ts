// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

// ── Types (mirror studio/backend/models/eval.py) ──────────────────────

export type EvalStatus =
  | "running"
  | "completed"
  | "cancelled"
  | "error"
  | "interrupted";

export interface EvalDatasetRef {
  is_local: boolean;
  path?: string | null;
  name?: string | null;
  split: string;
  subset?: string | null;
}

export interface EvalStartRequest {
  model_identifier: string;
  dataset: EvalDatasetRef;
  input_column: string;
  reference_column: string;
  metric_name: string;
  metric_config: Record<string, unknown>;
  system_prompt: string;
  template?: string | null;
  limit: number | null; // null = all rows
  max_new_tokens: number;
  temperature: number;
}

export interface EvalLastResult {
  idx: number;
  score: number;
  error?: string | null;
  input?: string | null;
  prediction?: string | null;
}

export interface EvalProgress {
  run_id: string;
  status: EvalStatus;
  done: number;
  total: number;
  avg_score: number;
  eta_sec?: number | null;
  last_result?: EvalLastResult | null;
}

export interface EvalRunSummary {
  id: string;
  status: EvalStatus;
  model_identifier: string;
  dataset_ref: string;
  metric_name: string;
  started_at: string;
  ended_at?: string | null;
  num_examples?: number | null;
  avg_score?: number | null;
  display_name?: string | null;
}

// ScoreNode breakdown produced by the json_document metric.
export interface ScoreNode {
  score: number;
  n_leaves: number;
  note?: string;
  matched_option?: number;
  children?: Record<string, ScoreNode> | ScoreNode[];
}

export interface EvalResultRow {
  idx: number;
  input_text?: string | null;
  prediction_text?: string | null;
  reference_text?: string | null;
  score?: number | null;
  breakdown?: ScoreNode | null;
  error?: string | null;
}

export interface EvalRunDetail {
  run: EvalRunSummary;
  results: EvalResultRow[];
  total_results: number;
}

export interface MetricConfigField {
  name: string;
  type: "bool" | "float" | "json" | "string" | string;
  default: unknown;
  label: string;
}

export interface MetricInfo {
  name: string;
  label: string;
  reference_kind: string;
  config_fields: MetricConfigField[];
}

// ── Helpers ───────────────────────────────────────────────────────────

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

// ── Calls ─────────────────────────────────────────────────────────────

export async function listMetrics(): Promise<MetricInfo[]> {
  const res = await authFetch("/api/eval/metrics");
  const data = await parseJson<{ metrics: MetricInfo[] }>(res);
  return data.metrics;
}

export interface SchemaComparatorField {
  path: string;
  comparator: string;
}

/** Preview which comparator each field resolves to for a schema (our
 *  field→comparator mapping OR a standard JSON Schema). */
export async function previewSchemaComparators(
  schema: unknown,
): Promise<SchemaComparatorField[]> {
  const res = await authFetch("/api/eval/schema-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ schema }),
  });
  const data = await parseJson<{ fields: SchemaComparatorField[] }>(res);
  return data.fields;
}

export async function startEval(
  payload: EvalStartRequest,
): Promise<{ run_id: string }> {
  const res = await authFetch("/api/eval/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<{ run_id: string }>(res);
}

export async function cancelEval(runId: string): Promise<void> {
  const res = await authFetch(`/api/eval/cancel/${encodeURIComponent(runId)}`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
}

export async function listEvalRuns(
  limit = 50,
  offset = 0,
): Promise<{ runs: EvalRunSummary[]; total: number }> {
  const res = await authFetch(`/api/eval/runs?limit=${limit}&offset=${offset}`);
  return parseJson<{ runs: EvalRunSummary[]; total: number }>(res);
}

export async function getEvalRun(
  runId: string,
  limit = 200,
  offset = 0,
): Promise<EvalRunDetail> {
  const res = await authFetch(
    `/api/eval/runs/${encodeURIComponent(runId)}?limit=${limit}&offset=${offset}`,
  );
  return parseJson<EvalRunDetail>(res);
}

// ── SSE progress stream (mirrors streamTrainingProgress) ───────────────

export interface EvalLogEntry {
  seq: number;
  ts: string;
  level: string;
  message: string;
}

type EvalEventName = "progress" | "complete" | "log";

type ParsedEvalEvent =
  | { event: "progress" | "complete"; payload: EvalProgress }
  | { event: "log"; logs: EvalLogEntry[] };

function parseEvalSseEvent(rawEvent: string): ParsedEvalEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let eventName: EvalEventName = "progress";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (!line) continue;
    if (line.startsWith("event:")) {
      const value = line.slice(6).trim();
      if (value === "progress" || value === "complete" || value === "log")
        eventName = value;
      continue;
    }
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  if (eventName === "log") {
    const parsed = JSON.parse(dataLines.join("\n")) as {
      entries?: EvalLogEntry[];
    };
    return { event: "log", logs: parsed.entries ?? [] };
  }
  const parsed = JSON.parse(dataLines.join("\n")) as EvalProgress;
  return { event: eventName, payload: parsed };
}

export async function streamEvalProgress(options: {
  runId: string;
  signal: AbortSignal;
  onOpen?: () => void;
  onEvent: (event: ParsedEvalEvent) => void;
}): Promise<void> {
  const res = await authFetch(
    `/api/eval/progress/${encodeURIComponent(options.runId)}`,
    { method: "GET", signal: options.signal },
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  if (!res.body) throw new Error("Progress stream unavailable");

  options.onOpen?.();

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep = buffer.search(/\r?\n\r?\n/);
    while (sep >= 0) {
      const rawEvent = buffer.slice(0, sep);
      const sepLen = buffer[sep] === "\r" ? 4 : 2;
      buffer = buffer.slice(sep + sepLen);

      if (!rawEvent.startsWith("retry:")) {
        try {
          const evt = parseEvalSseEvent(rawEvent);
          if (evt) options.onEvent(evt);
        } catch (error) {
          if (!isAbortError(error)) throw error;
        }
      }
      sep = buffer.search(/\r?\n\r?\n/);
    }
  }
}

export { isAbortError };
