// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** Model paths the backend loads onto the NPU (`lemonade:<Lemonade model id>`). */
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
  /** Validated now, or by an earlier Enable of this install. */
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
}

/** The NPU rows a picker section shows: downloaded ones on device, all of them to browse. */
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

/** Download a model, reporting each progress event. Resolves on `complete`, rejects on `error`. */
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
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
      if (!frame.startsWith("data:")) continue;
      const event = JSON.parse(frame.slice(5).trim()) as NpuDownloadEvent;
      if (event.event === "error") {
        throw new Error(event.error || "The download failed");
      }
      onProgress(event);
      if (event.event === "complete") return;
    }
  }
  throw new Error("The download ended before it completed");
}
