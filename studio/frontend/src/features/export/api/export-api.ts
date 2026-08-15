// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const readError = (r: Response): Promise<string> => readFastApiError(r);

/**
 * Error from an export request that preserves the HTTP status, so callers can
 * tell an authoritative backend rejection (4xx) from a transport timeout that
 * the long export may survive (e.g. Cloudflare's 524 over a quick tunnel).
 */
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

/**
 * Whether an error from an export POST is a recoverable transport failure (the
 * backend op may still be running and may still succeed) rather than an
 * authoritative rejection. A Cloudflare quick tunnel returns 524 (and friends
 * 520/522/523) when a request runs past ~100s; a dropped connection surfaces as
 * a status-less network error from authFetch. Real 4xx (400/404/409/422) are
 * the backend saying no and must fail immediately.
 */
export function isRecoverableTransportError(err: unknown): boolean {
  if (err instanceof ExportRequestError) {
    // 502/503/504 gateway + 520/522/523/524 Cloudflare timeouts are recoverable.
    return typeof err.status === "number" ? err.status >= 502 : true;
  }
  // No HTTP status at all (authFetch threw on a fetch TypeError = network drop /
  // backend unreachable): indeterminate, so recover by polling status.
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
}

export interface CheckpointListResponse {
  outputs_dir: string;
  models: ModelCheckpoints[];
}

export interface ExportSizeEstimate {
  /** Estimated FP16/BF16-equivalent on-disk size, or null when unknown. */
  fp16_bytes: number | null;
  total_params: number | null;
  source: string;
}

export interface ExportOperationResponse {
  success: boolean;
  message: string;
  /**
   * Optional backend extras. Local saves set `details.output_path` to the
   * saved model's on-disk directory; hub-only pushes leave it undefined.
   */
  details?: { output_path?: string | null } & Record<string, unknown>;
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const response = await authFetch("/api/models/checkpoints");
  return parseJson<CheckpointListResponse>(response);
}

/** Estimate a model's fp16-equivalent size to scale the GGUF quant labels; nulls (not error) when unknown. */
export async function fetchExportSize(
  modelId: string,
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<ExportSizeEstimate> {
  // Token in a header (not the query string) so it never lands in URLs/logs.
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
  /** Allow loading models with custom code. Only enable for checkpoints you trust. */
  trust_remote_code?: boolean;
  /** sha256 fingerprint pinning user approval of this exact custom-code version. */
  approved_remote_code_fingerprint?: string | null;
  /** HF token so the worker scans/loads gated checkpoints and base models with the same auth as preflight. */
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
  /** A single GGUF quant method or a list (list produces multiple GGUFs from one model load). */
  quantization_method: string | string[];
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  imatrix?: boolean;
  imatrix_path?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/gguf", {
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
  /** Also convert the adapter to a GGUF LoRA file (llama.cpp `--lora`). */
  gguf?: boolean;
  /** GGUF LoRA output float type (f32/f16/bf16/q8_0/auto); only used when gguf=true. */
  gguf_outtype?: string;
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

/**
 * Cancel the in-flight export by terminating its worker subprocess. Training
 * and inference are left running. Always resolves (best-effort) so callers can
 * fire it without guarding for a missing active run.
 */
export async function cancelExport(): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/cancel", { method: "POST" });
  return parseJson<ExportOperationResponse>(response);
}

export interface ExportStatus {
  current_checkpoint: string | null;
  is_vision: boolean;
  is_peft: boolean;
  /** True while a load / export / cleanup operation is running on the backend. */
  is_export_active: boolean;
  /** Kind of the currently running op (load_checkpoint / export_* / cleanup). */
  active_op_kind?: string | null;
  /** Monotonic counter of finished ops; baseline to detect "my op finished". */
  last_op_seq?: number;
  last_op_kind?: string | null;
  /** Outcome of the most recently finished op. */
  last_op_status?: "success" | "error" | "cancelled" | null;
  last_op_output_path?: string | null;
  last_op_error?: string | null;
}

/**
 * Snapshot of the export backend, used to hydrate the runtime store on mount /
 * page reload so a still-running export (started in another tab, or before a
 * refresh) is reflected even though the in-memory store reset.
 */
export async function getExportStatus(): Promise<ExportStatus> {
  const response = await authFetch("/api/export/status");
  return parseJson<ExportStatus>(response);
}

// ─────────────────────────────────────────────────────────────────────
// Live export log stream (Server-Sent Events)
// ─────────────────────────────────────────────────────────────────────

export type ExportLogStream = "stdout" | "stderr" | "status";

export interface ExportLogEntry {
  stream: ExportLogStream;
  line: string;
  ts: number | null;
}

/** A log line from the JSON poll endpoint, carrying its ring-buffer seq. */
export interface ExportLogPollEntry extends ExportLogEntry {
  seq: number;
}

export interface ExportLogsResponse {
  entries: ExportLogPollEntry[];
  /** Highest seq returned; pass back as `since` on the next poll. */
  cursor: number;
  /** True while a load / export / cleanup op is running on the backend. */
  active: boolean;
}

/**
 * Tunnel-safe JSON fallback for {@link streamExportLogs}. Cloudflare quick
 * tunnels (`--secure` mode) buffer `text/event-stream`, so the SSE stream
 * delivers nothing until it closes; this plain-JSON poll is never buffered and
 * carries the same ring-buffer lines. Poll it while a run is active and merge
 * the entries into the store (de-duped by seq), so logs appear over the tunnel.
 */
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
  /** Present on `log` events. */
  entry?: ExportLogEntry;
  /** Present on `error` events. */
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
    // Comment lines (":heartbeat" etc.) are ignored per SSE spec.
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

  const response = await authFetch(url, {
    method: "GET",
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
          // Ignore malformed events, keep reading.
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
