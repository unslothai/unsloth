// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { readSseJsonEvents } from "@/lib/sse-json-events";

export const NPU_MODEL_PREFIX = "lemonade:";

export function isNpuModelId(
  value: string | null | undefined,
): value is string {
  return typeof value === "string" && value.startsWith(NPU_MODEL_PREFIX);
}

export interface NpuValidation {
  ready: boolean;
  problems: string[];
}

export interface NpuStatus {
  supported: boolean;
  hardware: {
    present: boolean;
    supported: boolean;
    family?: string;
    name?: string;
    driver?: string | null;
    driver_version?: string | null;
    firmware_version?: string | null;
  };
  runtime_installed: boolean;
  runtime_running: boolean;
  state: string;
  ready: boolean;
  error: string | null;
  validation: NpuValidation | null;
  help_url: string | null;
  loaded_model: string | null;
  context_length: number | null;
  loading_model: string | null;
  /** Installed runtime versions; absent from servers that predate them. */
  versions?: { lemonade: string | null; fastflowlm: string | null };
  /** About how much memory this PC can give the NPU, in GB; null when unknown. */
  memory_window_gb?: number | null;
}

export interface NpuModel {
  id: string;
  model_path: string;
  checkpoint: string;
  size_gb: number | null;
  downloaded: boolean;
  labels: string[];
  supports_vision: boolean;
  supports_reasoning: boolean;
  supports_tools: boolean;
  max_context_length: number | null;
  resume_percent: number | null;
  /** FastFlowLM's Hugging Face repo (owner/name) and revision; absent from older servers. */
  hf_repo?: string | null;
  hf_revision?: string | null;
  /** Size plus load overhead, in GB. */
  memory_needed_gb?: number | null;
  /** false when the model is larger than the NPU memory window; null when unknown. */
  fits_memory?: boolean | null;
}

export function npuRowsFor(
  models: NpuModel[] | null,
  {
    onDevice,
    query,
  }: {
    onDevice: boolean;
    query: string;
  },
): NpuModel[] {
  const needle = query.trim().toLowerCase();
  return (models ?? [])
    .filter(
      (model) =>
        (!onDevice || model.downloaded) &&
        (!needle ||
          `${model.id} ${model.checkpoint}`.toLowerCase().includes(needle)),
    )
    .sort((a, b) => npuRank(a.checkpoint) - npuRank(b.checkpoint));
}

// Newest models that run on a Strix laptop NPU first. Big ones (qwen3.6-moe 35B-A3B, qwen3.8
// 27B) fail to load or take hours to download there, so they keep the backend's order after.
const NPU_RECOMMENDED = [
  "qwen3.5:9b",
  "qwen3.5:4b",
  "gemma4-it:e4b",
  "gpt-oss:20b",
  "lfm3:1.7b",
  "nemotron3.5-bolt:e4b",
  "qwen3vl-it:4b",
  "gemma4-it:e2b",
  "qwen3.5:2b",
  "qwen3.5:0.8b",
  "phi4-mini-it:4b",
  "llama3.2:3b",
];

function npuRank(checkpoint: string): number {
  const rank = NPU_RECOMMENDED.indexOf(checkpoint);
  return rank === -1 ? NPU_RECOMMENDED.length : rank;
}

export interface NpuCapability {
  key: "vision" | "reasoning" | "tools";
  label: string;
}

export function npuCapabilities(model: NpuModel): NpuCapability[] {
  const out: NpuCapability[] = [];
  if (model.supports_vision) out.push({ key: "vision", label: "Vision" });
  if (model.supports_reasoning)
    out.push({ key: "reasoning", label: "Reasoning" });
  if (model.supports_tools) out.push({ key: "tools", label: "Tools" });
  return out;
}

/** "32K context", or null when the server did not report one. */
export function npuContextLabel(tokens: number | null): string | null {
  if (tokens == null || tokens <= 0) return null;
  return tokens >= 1024 && tokens % 1024 === 0
    ? `${tokens / 1024}K context`
    : `${tokens.toLocaleString("en-US")} context`;
}

export function npuSizeLabel(sizeGb: number | null): string | null {
  if (sizeGb == null) return null;
  return sizeGb < 1
    ? `${Math.round(sizeGb * 1000)} MB`
    : `${sizeGb.toFixed(1)} GB`;
}

/** FastFlowLM pins repos by branch, which moves; only a commit revision matches the hash-checked files. */
export function npuReadmeNote(model: NpuModel): string | null {
  return model.hf_repo && !/^[0-9a-f]{40}$/i.test(model.hf_revision ?? "")
    ? "Latest on Hugging Face (may be newer than the installed files)"
    : null;
}

/** Row tag for a model the backend says is larger than the NPU memory window. */
export function npuMemoryTag(model: NpuModel): string | null {
  return model.fits_memory === false ? "May not fit NPU memory" : null;
}

/** The full warning for such a model, or null. It only warns: the window is an estimate. */
export function npuMemoryWarning(
  model: NpuModel,
  status: Pick<NpuStatus, "memory_window_gb">,
): string | null {
  if (model.fits_memory !== false) return null;
  const needed = model.memory_needed_gb;
  const window = status.memory_window_gb;
  if (needed == null || window == null) {
    return "This model may need more NPU memory than this PC can give the NPU.";
  }
  return `Needs about ${Math.round(needed)} GB of NPU memory; this PC can give the NPU about ${Math.round(window)} GB.`;
}

export function npuResumeLabel(model: NpuModel): string | null {
  return model.resume_percent == null
    ? null
    : `${model.resume_percent}% downloaded`;
}

/** Rehashing kept files before a resume, or Lemonade registering the files after the download. */
export type NpuDownloadPhase = "verifying" | "finishing";

export function npuDownloadLabel(
  percent: number | null | undefined,
  reconnecting = false,
  phase?: NpuDownloadPhase,
): string {
  const done = percent == null ? "" : ` ${Math.round(percent)}%`;
  if (reconnecting) return `Reconnecting${done}`;
  if (phase === "finishing") return "Finishing";
  return phase === "verifying" ? `Verifying${done}` : `Downloading${done}`;
}

export interface NpuDownloadEvent {
  event: string;
  percent?: number;
  bytes_downloaded?: number;
  bytes_total?: number;
  file?: string;
  file_index?: number;
  total_files?: number;
  error?: string;
}

export async function getNpuStatus(signal?: AbortSignal): Promise<NpuStatus> {
  const response = await authFetch("/api/npu/status", { signal });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not read the NPU status"),
    );
  }
  return (await response.json()) as NpuStatus;
}

export async function enableNpu(): Promise<NpuStatus> {
  const response = await authFetch("/api/npu/enable", { method: "POST" });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not enable the NPU runtime"),
    );
  }
  return (await response.json()) as NpuStatus;
}

export async function listNpuModels(): Promise<NpuModel[]> {
  const response = await authFetch("/api/npu/models");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not list NPU models"),
    );
  }
  return ((await response.json()) as { models: NpuModel[] }).models;
}

export interface NpuRunningDownload {
  model: string;
  percent: number | null;
}

export async function listNpuDownloads(): Promise<NpuRunningDownload[]> {
  const response = await authFetch("/api/npu/downloads");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not list NPU downloads"),
    );
  }
  return ((await response.json()) as { downloads: NpuRunningDownload[] })
    .downloads;
}

export async function deleteNpuModel(id: string): Promise<void> {
  const response = await authFetch(
    `/api/npu/models/${encodeURIComponent(id)}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not delete the model"),
    );
  }
}

/** Stops the pull at its next chunk; its partial files stay, so Download resumes. */
export async function cancelNpuModelDownload(id: string): Promise<void> {
  const response = await authFetch(
    `/api/npu/models/${encodeURIComponent(id)}/download/cancel`,
    { method: "POST" },
  );
  // 404: the pull already ended.
  if (!response.ok && response.status !== 404) {
    throw new Error(
      await readFastApiError(response, "Could not cancel the download"),
    );
  }
}

export async function downloadNpuModel(
  id: string,
  onProgress: (event: NpuDownloadEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await authFetch(
    `/api/npu/models/${encodeURIComponent(id)}/download`,
    { method: "POST", signal },
  );
  if (!response.ok || !response.body) {
    throw new Error(
      await readFastApiError(response, "Could not download the model"),
    );
  }
  await readDownloadStream(response.body, onProgress);
}

export async function followNpuModelDownload(
  id: string,
  onProgress: (event: NpuDownloadEvent) => void,
): Promise<void> {
  const response = await authFetch(
    `/api/npu/models/${encodeURIComponent(id)}/download`,
  );
  if (response.status === 404) return;
  if (!response.ok || !response.body) {
    throw new Error(
      await readFastApiError(response, "Could not follow the download"),
    );
  }
  await readDownloadStream(response.body, onProgress);
}

export class NpuDownloadError extends Error {}

export class NpuDownloadCancelled extends NpuDownloadError {}

async function readDownloadStream(
  body: ReadableStream<Uint8Array>,
  onProgress: (event: NpuDownloadEvent) => void,
): Promise<void> {
  for await (const event of readSseJsonEvents<NpuDownloadEvent>(body)) {
    if (event.event === "error") {
      throw new NpuDownloadError(event.error || "The download failed");
    }
    if (event.event === "cancelled") {
      throw new NpuDownloadCancelled("The download was cancelled");
    }
    onProgress(event);
    if (event.event === "complete") return;
  }
  throw new Error("The download ended before it completed");
}
