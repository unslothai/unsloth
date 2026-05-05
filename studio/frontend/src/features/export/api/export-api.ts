// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string; message?: string };
    return payload.detail || payload.message || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
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

export interface ExportOperationResponse {
  success: boolean;
  message: string;
  /**
   * Optional extras returned by the backend. The export endpoints set
   * `details.output_path` to the resolved on-disk directory of the
   * saved model when a local save was requested. Hub-only pushes leave
   * `details` undefined.
   */
  details?: { output_path?: string | null } & Record<string, unknown>;
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const response = await authFetch("/api/models/checkpoints");
  return parseJson<CheckpointListResponse>(response);
}

export async function loadCheckpoint(params: {
  checkpoint_path: string;
  max_seq_length?: number;
  load_in_4bit?: boolean;
  /** Allow loading models with custom code. Only enable for checkpoints you trust. */
  trust_remote_code?: boolean;
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
  quantization_method: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
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

// ─────────────────────────────────────────────────────────────────────
// Live export log stream (Server-Sent Events)
// ─────────────────────────────────────────────────────────────────────

export type ExportLogStream = "stdout" | "stderr" | "status";

export interface ExportLogEntry {
  stream: ExportLogStream;
  line: string;
  ts: number | null;
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
  }
}
