// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export interface LinkedInstance {
  id: string;
  name: string;
  base_url: string;
  created_at: string;
  updated_at: string;
}

export interface LinkedInstanceStatus {
  id: string;
  online: boolean;
  error: string | null;
  models: string[];
  loaded: string[];
  latency_ms: number | null;
}

async function detail(res: Response, fallback: string): Promise<Error> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return new Error(body.detail);
  } catch {
    // not JSON
  }
  return new Error(fallback);
}

export async function fetchLinkedInstances(): Promise<LinkedInstance[]> {
  const res = await authFetch("/api/linked-instances");
  if (!res.ok) throw await detail(res, "Failed to load linked instances");
  return res.json();
}

export async function createLinkedInstance(input: {
  name: string;
  base_url: string;
  api_key: string;
}): Promise<LinkedInstance> {
  const res = await authFetch("/api/linked-instances", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw await detail(res, "Failed to link instance");
  return res.json();
}

export async function deleteLinkedInstance(id: string): Promise<void> {
  const res = await authFetch(`/api/linked-instances/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await detail(res, "Failed to remove linked instance");
}

export async function testLinkedInstance(
  id: string,
): Promise<LinkedInstanceStatus> {
  const res = await authFetch(`/api/linked-instances/${id}/test`, {
    method: "POST",
  });
  if (!res.ok) throw await detail(res, "Failed to reach linked instance");
  return res.json();
}

export async function fetchLinkedInstancesStatus(): Promise<
  LinkedInstanceStatus[]
> {
  const res = await authFetch("/api/linked-instances/status");
  if (!res.ok) throw await detail(res, "Failed to load linked instances");
  return res.json();
}

export async function updateLinkedInstance(
  id: string,
  input: { name?: string; base_url?: string; api_key?: string },
): Promise<LinkedInstance> {
  const res = await authFetch(`/api/linked-instances/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw await detail(res, "Failed to update linked instance");
  return res.json();
}

export interface LinkedInstanceGpu {
  name: string;
  vram_total_gb: number | null;
  vram_used_gb: number | null;
  utilization_pct: number | null;
}

/** What a linked instance reports about itself; older releases leave fields null. */
export interface LinkedInstanceInfo {
  id: string;
  online: boolean;
  error: string | null;
  version: string | null;
  install_source: string | null;
  update_available: boolean;
  latest_version: string | null;
  platform: string | null;
  python_version: string | null;
  device_backend: string | null;
  torch: string | null;
  transformers: string | null;
  cuda: string | null;
  rocm: string | null;
  llama_cpp: string | null;
  gpus: LinkedInstanceGpu[];
  cpu_count: number | null;
  memory_total_gb: number | null;
  memory_available_gb: number | null;
  disk_total_gb: number | null;
  disk_free_gb: number | null;
  uptime_seconds: number | null;
  /** Absent on a release that predates it. */
  image_model?: string | null;
}

export async function fetchLinkedInstancesInfo(): Promise<
  LinkedInstanceInfo[]
> {
  const res = await authFetch("/api/linked-instances/info");
  if (!res.ok) throw await detail(res, "Failed to load linked instances");
  return res.json();
}
