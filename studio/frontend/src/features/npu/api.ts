// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { readSseJsonEvents } from "@/lib/sse-json-events";

export const NPU_MODEL_PREFIX = "lemonade:";

/** Keep in sync with npu_backend.py's DEFAULT_CONTEXT_LENGTH. */
export const NPU_DEFAULT_CONTEXT_LENGTH = 8192;

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
  return (models ?? []).filter(
    (model) =>
      (!onDevice || model.downloaded) &&
      (!needle ||
        `${model.id} ${model.checkpoint}`.toLowerCase().includes(needle)),
  );
}

export function npuSizeLabel(sizeGb: number | null): string | null {
  if (sizeGb == null) return null;
  return sizeGb < 1
    ? `${Math.round(sizeGb * 1000)} MB`
    : `${sizeGb.toFixed(1)} GB`;
}

export function npuResumeLabel(model: NpuModel): string | null {
  return model.resume_percent == null
    ? null
    : `${model.resume_percent}% downloaded`;
}

export function npuDownloadLabel(
  percent: number | null | undefined,
  reconnecting = false,
): string {
  const done = percent == null ? "" : ` ${Math.round(percent)}%`;
  return reconnecting ? `Reconnecting${done}` : `Downloading${done}`;
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

async function readDownloadStream(
  body: ReadableStream<Uint8Array>,
  onProgress: (event: NpuDownloadEvent) => void,
): Promise<void> {
  for await (const event of readSseJsonEvents<NpuDownloadEvent>(body)) {
    if (event.event === "error") {
      throw new NpuDownloadError(event.error || "The download failed");
    }
    onProgress(event);
    if (event.event === "complete") return;
  }
  throw new Error("The download ended before it completed");
}
