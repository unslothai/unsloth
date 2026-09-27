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

export type ColabGpu = "T4" | "L4" | "A100" | "H100";
export type ColabStage =
  "allocating" | "installing" | "starting" | "linking" | "ready";

export interface ColabCapability {
  state:
    "ready" | "unsupported" | "missing_cli" | "kernel_client" | "signed_out";
  ready: boolean;
  message: string;
  setup: string[];
  runner: "native" | "wsl" | null;
  distro: string | null;
  auth: string | null;
  detail: string | null;
}

export interface ColabLaunchJob {
  id: string;
  name: string;
  gpu: ColabGpu;
  session: string;
  stage: ColabStage;
  state: "running" | "ready" | "failed" | "cancelled";
  error: string | null;
  setup: string[];
  instance_id: string | null;
  started_at: string;
  finished_at: string | null;
  log: string[];
}

/** A Colab VM this machine started; it bills until stopped. */
export interface ColabSession {
  session: string;
  name: string;
  gpu: string;
  instance_id: string | null;
  created_at: string;
}

export async function fetchColabCapability(): Promise<ColabCapability> {
  const res = await authFetch("/api/linked-instances/colab/capability");
  if (!res.ok) throw await detail(res, "Failed to check the Colab CLI");
  return res.json();
}

export async function fetchColabLaunch(): Promise<ColabLaunchJob | null> {
  const res = await authFetch("/api/linked-instances/colab/launch");
  if (!res.ok) throw await detail(res, "Failed to load the Colab launch");
  return res.json();
}

export async function startColabLaunch(input: {
  gpu: ColabGpu;
  name: string;
}): Promise<ColabLaunchJob> {
  const res = await authFetch("/api/linked-instances/colab/launch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw await detail(res, "Failed to launch a Colab GPU");
  return res.json();
}

export async function cancelColabLaunch(): Promise<void> {
  const res = await authFetch("/api/linked-instances/colab/launch/cancel", {
    method: "POST",
  });
  if (!res.ok) throw await detail(res, "Failed to cancel the Colab launch");
}

export async function fetchColabSessions(): Promise<ColabSession[]> {
  const res = await authFetch("/api/linked-instances/colab/sessions");
  if (!res.ok) throw await detail(res, "Failed to load Colab sessions");
  return res.json();
}

export async function stopColabSession(session: string): Promise<void> {
  const res = await authFetch(
    `/api/linked-instances/colab/sessions/${encodeURIComponent(session)}/stop`,
    { method: "POST" },
  );
  if (!res.ok) throw await detail(res, "Failed to stop the Colab session");
}
