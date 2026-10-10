// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";

const readError = (r: Response): Promise<string> => readFastApiError(r);

/** Keeps the HTTP status so callers can tell a 4xx rejection from a tunnel timeout. */
export class ExportRequestError extends Error {
  status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ExportRequestError";
    this.status = status;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new ExportRequestError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

/** Gateway/tunnel timeouts (502+, 52x) and network drops are recoverable; real 4xx are not. */
export function isRecoverableTransportError(err: unknown): boolean {
  if (err instanceof ExportRequestError) {
    return typeof err.status === "number" ? err.status >= 502 : true;
  }
  // No status means a network drop: indeterminate, so recover by polling status.
  return err instanceof Error;
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
  adapter_features?: {
    dora?: boolean | null;
    full_state?: boolean | null;
    moe_target_parameters?: boolean | null;
    non_uniform?: boolean | null;
  } | null;
}

export interface CheckpointListResponse {
  outputs_dir: string;
  models: ModelCheckpoints[];
}

export interface ExportSizeEstimate {
  fp16_bytes: number | null;
  total_params: number | null;
  source: string;
}

export interface ExportOperationResponse {
  success: boolean;
  message: string;
  /** Local saves set `details.output_path`; hub-only pushes leave it undefined. */
  details?: { output_path?: string | null } & Record<string, unknown>;
}

/** GGUF export options for a decision model (Clef / Laya) checkpoint folder. */
export interface DecisionExportInfo {
  is_decision: boolean;
  layout: "clef" | "laya";
  /** Clef folder with LoRA adapters only; merged at export time. */
  adapter_only: boolean;
  /** false: llama.cpp cannot serve it (see reason); null: only the export can tell. */
  eligible: boolean | null;
  reason: string | null;
  /** Allowed lowercase quantizations, default first. */
  quantizations: string[];
  default_quantization: string;
  /** Where the GGUF files land: <run folder>/gguf. */
  output_dir: string;
  existing_export: ({ quantizations?: string[] } & Record<string, unknown>) | null;
}

/** Decision export info for a local checkpoint folder; null for any other model. */
export async function fetchDecisionExportInfo(
  checkpointPath: string,
  signal?: AbortSignal,
): Promise<DecisionExportInfo | null> {
  const response = await authFetch(
    `/api/export/decision-info?checkpoint_path=${encodeURIComponent(checkpointPath)}`,
    { signal },
  );
  const body = await parseJson<{ decision: DecisionExportInfo | null }>(
    response,
  );
  return body.decision ?? null;
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const response = await authFetch("/api/models/checkpoints");
  return parseJson<CheckpointListResponse>(response);
}

export async function fetchExportSize(
  modelId: string,
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<ExportSizeEstimate> {
  // Token in a header, not the query string, so it never lands in URLs or logs.
  const headers: Record<string, string> = {};
  if (hfToken) {
    headers["X-HF-Token"] = hfToken;
  }
  const response = await authFetch(
    `/api/models/export-size?model=${encodeURIComponent(modelId)}`,
    { signal, headers },
  );
  return parseJson<ExportSizeEstimate>(response);
}

export async function loadCheckpoint(params: {
  checkpoint_path: string;
  max_seq_length?: number;
  load_in_4bit?: boolean;
  trust_remote_code?: boolean;
  approved_remote_code_fingerprint?: string | null;
  hf_token?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/load-checkpoint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportMerged(params: {
  save_directory: string;
  format_type?: string;
  /** Compressed-tensors scheme alias (e.g. "fp8", "w4a16", "mxfp4"); overrides format_type. */
  compressed_method?: string | null;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
  install_missing_dependencies?: boolean;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/merged", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportBase(params: {
  save_directory: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
  base_model_id?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/base", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportGGUF(params: {
  save_directory: string;
  quantization_method: string | string[];
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  imatrix?: boolean;
  imatrix_path?: string | null;
  private?: boolean;
  /** Also convert a Q4_0/Q4_1/Q4_K_M GGUF to Q4NX for the AMD NPU, into <save_directory>/npu-q4nx. */
  npu_q4nx?: boolean;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/gguf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function convertGgufToQ4nx(params: {
  save_directory: string;
  gguf_path?: string | null;
  repo_id?: string | null;
  filename?: string | null;
  /** Original (non-GGUF) repo or local folder that supplies config.json and the tokenizer files. */
  base_model: string;
  hf_token?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/convert/q4nx", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportLoRA(params: {
  save_directory: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
  gguf?: boolean;
  gguf_outtype?: string;
  adapter_format?: "mlx" | "peft";
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/lora", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function cleanupExport(): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/cleanup", { method: "POST" });
  return parseJson<ExportOperationResponse>(response);
}

/** Terminates the export worker only; always resolves so callers need no guard. */
export async function cancelExport(): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/cancel", { method: "POST" });
  return parseJson<ExportOperationResponse>(response);
}

export interface ExportStatus {
  current_checkpoint: string | null;
  is_vision: boolean;
  is_peft: boolean;
  is_export_active: boolean;
  active_op_kind?: string | null;
  /** Monotonic counter of finished ops; baseline to detect "my op finished". */
  last_op_seq?: number;
  last_op_kind?: string | null;
  last_op_status?: "success" | "error" | "cancelled" | null;
  last_op_output_path?: string | null;
  last_op_error?: string | null;
  /** {layout, adapter_only} when the loaded checkpoint is a decision model. */
  decision?: { layout: "clef" | "laya"; adapter_only: boolean } | null;
}

/** Used on mount to show an export started in another tab or before a reload. */
export async function getExportStatus(): Promise<ExportStatus> {
  const response = await authFetch("/api/export/status");
  return parseJson<ExportStatus>(response);
}

export type ExportLogStream = "stdout" | "stderr" | "status";

export interface ExportLogEntry {
  stream: ExportLogStream;
  line: string;
  ts: number | null;
}

export interface ExportLogPollEntry extends ExportLogEntry {
  seq: number;
}

export interface ExportLogsResponse {
  entries: ExportLogPollEntry[];
  /** Pass back as `since` on the next poll. */
  cursor: number;
  active: boolean;
}

/** Polling fallback for when a proxy drops or stalls the SSE stream; dedupe by seq. */
export async function fetchExportLogs(
  since: number | null,
): Promise<ExportLogsResponse> {
  const url =
    typeof since === "number"
      ? `/api/export/logs?since=${since}`
      : "/api/export/logs";
  const response = await authFetch(url);
  return parseJson<ExportLogsResponse>(response);
}

export type ExportLogEventName = "log" | "heartbeat" | "complete" | "error";

export interface ExportLogEvent {
  event: ExportLogEventName;
  id: number | null;
  entry?: ExportLogEntry;
  error?: string;
}

interface ParsedSseMessage {
  event: string;
  id: number | null;
  data: string;
}

function parseSseMessage(raw: string): ParsedSseMessage | null {
  const lines = raw.split(/\r?\n/);
  let event = "message";
  let id: number | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("id:")) {
      const value = Number(line.slice(3).trim());
      id = Number.isFinite(value) ? value : null;
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
      continue;
    }
  }

  if (dataLines.length === 0) return null;
  return { event, id, data: dataLines.join("\n") };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export async function streamExportLogs(options: {
  signal: AbortSignal;
  since?: number | null;
  onOpen?: () => void;
  onEvent: (event: ExportLogEvent) => void;
}): Promise<void> {
  const headers = new Headers();
  if (typeof options.since === "number") {
    headers.set("Last-Event-ID", String(options.since));
  }

  const url =
    typeof options.since === "number"
      ? `/api/export/logs/stream?since=${options.since}`
      : "/api/export/logs/stream";

  const response = await openStreamResponse(authFetch, url, {
    headers,
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }
  if (!response.body) {
    throw new Error("Export log stream unavailable");
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
              stream?: ExportLogStream;
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
            options.onEvent({ event: "heartbeat", id: parsed.id });
          } else if (parsed.event === "complete") {
            options.onEvent({ event: "complete", id: parsed.id });
            return;
          } else if (parsed.event === "error") {
            let errorMessage = "Export log stream error";
            try {
              const payload = JSON.parse(parsed.data) as { error?: string };
              if (payload.error) errorMessage = payload.error;
            } catch {
              // fall through with default message
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
    // Release the stream lock now instead of leaking the reader until GC.
    try {
      await reader.cancel();
    } catch {
      // already closed
    }
  }
}
