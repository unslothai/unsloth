// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";
import { readSseJsonEvents } from "@/lib/sse-json-events";
import { isStudioMediaUrl } from "./media-url";
import type {
  FlowDocument,
  FlowRecord,
  FlowRunSummary,
  FlowSummary,
  MediaValue,
  NodeCatalog,
  NodeStatus,
  RunEvent,
} from "./types";

export const FLOWS_API = "/api/flows";

export type FlowsFetch = (
  url: string,
  init?: RequestInit,
  options?: { retryNetworkErrors?: boolean },
) => Promise<Response>;

let transport: FlowsFetch = (url, init, options) =>
  authFetch(url, init, options);

export function setFlowsTransport(next: FlowsFetch | null): void {
  transport = next ?? ((url, init, options) => authFetch(url, init, options));
}

export class FlowsApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await transport(`${FLOWS_API}${path}`, init);
  if (!response.ok) {
    throw new FlowsApiError(await readFastApiError(response), response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const getCatalog = () => request<NodeCatalog>("/nodes");

export const listFlows = () => request<{ flows: FlowSummary[] }>("");

export const listExamples = () =>
  request<{ examples: FlowSummary[] }>("/examples");

export const copyExample = (exampleId: string) =>
  request<FlowRecord>(`/examples/${encodeURIComponent(exampleId)}/copy`, {
    method: "POST",
  });

export const createFlow = (doc: FlowDocument) =>
  request<FlowRecord>("", json("POST", { doc }));

export const getFlow = (flowId: string) =>
  request<FlowRecord>(`/${encodeURIComponent(flowId)}`);

export const saveFlow = (
  flowId: string,
  doc: FlowDocument,
  baseUpdatedAt: number | null,
) =>
  request<FlowRecord>(
    `/${encodeURIComponent(flowId)}`,
    json("PUT", { doc, baseUpdatedAt }),
  );

export const deleteFlow = (flowId: string) =>
  request<void>(`/${encodeURIComponent(flowId)}`, { method: "DELETE" });

export interface ValidationIssue {
  node_id?: string | null;
  edge_id?: string | null;
  message: string;
}

export const validateFlow = (doc: FlowDocument) =>
  request<{
    ok: boolean;
    errors: ValidationIssue[];
    warnings: ValidationIssue[];
  }>("/validate", json("POST", { doc }));

export interface PlanModel {
  ref?: string | null;
  kind: string;
  repo: string;
  variant?: string | null;
  downloaded?: boolean;
  size_bytes?: number | null;
  fits?: boolean | null;
  loaded?: boolean;
}

export interface PlanHint {
  id: string;
  message: string;
}

export interface FlowPlan {
  models: PlanModel[];
  gpu_ops: number;
  cached_nodes: number;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  hints?: PlanHint[];
}

export const planFlow = (
  flowId: string,
  params?: Record<string, unknown>,
  doc?: FlowDocument,
) =>
  request<FlowPlan>(
    `/${encodeURIComponent(flowId)}/plan`,
    json("POST", { params: params ?? {}, ...(doc ? { doc } : {}) }),
  );

export interface StartRunBody {
  params?: Record<string, unknown>;
  target_nodes?: string[];
  force_nodes?: string[];
  bake?: boolean;
  doc?: FlowDocument;
  pinned?: Record<string, Record<string, unknown>>;
}

export const startRun = (flowId: string, body: StartRunBody) =>
  request<{ run_id: string }>(
    `/${encodeURIComponent(flowId)}/runs`,
    json("POST", body),
  );

export const listRuns = (flowId: string) =>
  request<{ runs: FlowRunSummary[] }>(`/${encodeURIComponent(flowId)}/runs`);

export interface RunNodeState {
  status: NodeStatus;
  outputs?: Record<string, unknown> | null;
  error?: string | null;
  progress?: number | null;
}

export interface RunSnapshot {
  run: FlowRunSummary;
  nodes: Record<string, RunNodeState>;
  last_seq: number;
}

export const getRun = (runId: string) =>
  request<RunSnapshot>(`/runs/${encodeURIComponent(runId)}`);

export const cancelRun = (runId: string) =>
  request<{ status: string }>(`/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });

export const latestOutputs = (flowId: string) =>
  request<{ nodes: Record<string, RunNodeState> }>(
    `/${encodeURIComponent(flowId)}/latest-outputs`,
  );

export interface NodeHistoryEntry {
  run_id: string;
  updated_at: number;
  outputs: Record<string, unknown>;
}

export const nodeHistory = (flowId: string, nodeId: string) =>
  request<{ entries: NodeHistoryEntry[] }>(
    `/${encodeURIComponent(flowId)}/nodes/${encodeURIComponent(nodeId)}/history`,
  );

export const recipeRuns = (flowId: string, nodeId: string, limit: number) =>
  request<{ executions: Record<string, unknown>[] }>(
    `/${encodeURIComponent(flowId)}/nodes/${encodeURIComponent(nodeId)}/recipe-runs?limit=${limit}`,
  );

export const recipeRun = (flowId: string, nodeId: string, runId: string) =>
  request<Record<string, unknown> & { dataset?: unknown; datasetTotal?: unknown }>(
    `/${encodeURIComponent(flowId)}/nodes/${encodeURIComponent(nodeId)}/recipe-runs/${encodeURIComponent(runId)}`,
  );

export const datasetRows = (value: unknown, offset: number, limit: number) =>
  request<import("./dataset/dataset-value").DatasetRowsPage>(
    "/dataset-rows",
    json("POST", { value, offset, limit }),
  );

export interface ImportReport {
  missing_models: PlanModel[];
  warnings: string[];
  baked_nodes: string[];
  license?: string | null;
  content_rating?: string | null;
  author?: string | null;
  replication?: string[];
  live_note?: string | null;
  unsupported?: string[];
  colab_url?: string | null;
}

export const importFlow = async (
  data: Blob | ArrayBuffer | string,
  name?: string,
): Promise<{ flow: FlowRecord; report: ImportReport }> => {
  const query = name ? `?${new URLSearchParams({ name }).toString()}` : "";
  return request(`/import${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: data,
  });
};

export async function importNotebook(
  data: Blob | string,
  name: string,
): Promise<{ flow: FlowRecord | null; report: ImportReport }> {
  const query = new URLSearchParams({ name }).toString();
  const result = await request<{
    flow: FlowRecord | null;
    report: {
      approximated?: string[];
      unsupported?: string[];
      colab_url?: string | null;
      missing_models?: PlanModel[];
    };
  }>(`/import/notebook?${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: data,
  });
  const report = result.report;
  return {
    flow: result.flow,
    report: {
      missing_models: report.missing_models ?? [],
      warnings: report.approximated ?? [],
      baked_nodes: [],
      unsupported: report.unsupported ?? [],
      colab_url: report.colab_url ?? null,
    },
  };
}

/** A saved Data Recipe (by id) or recipe JSON as a new flow with one Recipe group. */
export const importRecipe = (
  source: { recipe_id: string } | { payload: unknown; name?: string },
) =>
  request<{ flow: FlowRecord; report: { notices: string[] } }>(
    "/import/recipe",
    json("POST", source),
  );

export interface RecipeMigration {
  created: {
    recipe_id: string;
    flow_id: string;
    name: string;
    notices: string[];
  }[];
  skipped: { recipe_id: string; flow_id: string }[];
  failed: { recipe_id: string; name: string; error: string }[];
}

/** One flow per saved Data Recipe not in Flows yet; repeating it creates nothing new. */
export const importAllRecipes = () =>
  request<RecipeMigration>("/import/recipes", { method: "POST" });

export interface NotebookEntry {
  id: string;
  title: string;
  notebook_file: string;
  category: string;
  featured: boolean;
  studio_model?: string | null;
  studio_task?: string | null;
  colab_url: string;
  github_url: string;
}

export async function listNotebooks(): Promise<NotebookEntry[]> {
  const response = await transport("/api/notebooks");
  if (!response.ok) {
    throw new FlowsApiError(await readFastApiError(response), response.status);
  }
  return ((await response.json()) as { notebooks: NotebookEntry[] }).notebooks;
}

export type ExportMode = "json" | "package" | "baked";

export async function exportFlow(
  flowId: string,
  mode: ExportMode,
): Promise<Blob> {
  const response = await transport(
    `${FLOWS_API}/${encodeURIComponent(flowId)}/export?mode=${mode}`,
  );
  if (!response.ok) {
    throw new FlowsApiError(await readFastApiError(response), response.status);
  }
  return response.blob();
}

export async function uploadAsset(file: File): Promise<MediaValue> {
  const query = new URLSearchParams({
    name: file.name,
    mime: file.type || "application/octet-stream",
  });
  const result = await request<{ media: MediaValue }>(
    `/assets?${query.toString()}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    },
  );
  return result.media;
}

const GALLERY_ROUTES: Record<string, string> = {
  image: "images",
  audio: "audio",
  video: "video",
};

export function mediaPath(media: MediaValue): string | null {
  if (isStudioMediaUrl(media.url)) return media.url;
  if (media.ref.startsWith("sha256:") || media.ref.startsWith("flowasset:")) {
    return `${FLOWS_API}/assets/${encodeURIComponent(media.ref)}`;
  }
  const colon = media.ref.indexOf(":");
  const gallery = GALLERY_ROUTES[media.ref.slice(0, colon)];
  const id = media.ref.slice(colon + 1);
  if (colon > 0 && gallery && id) {
    return `/api/inference/${gallery}/gallery/${encodeURIComponent(id)}/file`;
  }
  return null;
}

export async function fetchMediaBlob(
  path: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await transport(path, { signal });
  if (!response.ok)
    throw new FlowsApiError(
      `Media failed (${response.status})`,
      response.status,
    );
  return response.blob();
}

/**
 * Follow a run's events from `afterSeq`, reconnecting on a dropped stream until a terminal event.
 * Every reconnect starts with a `snapshot`, so a gap is never silent.
 */
export async function followRunEvents(
  runId: string,
  afterSeq: number,
  onEvent: (event: RunEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  let lastSeq = afterSeq;
  let attempt = 0;
  while (!signal.aborted) {
    let terminal = false;
    try {
      const response = await openStreamResponse(
        (url, init, options) => transport(url, init, options),
        `${FLOWS_API}/runs/${encodeURIComponent(runId)}/events?after=${lastSeq}`,
        { headers: { "Last-Event-ID": String(lastSeq) }, signal },
        { retryNetworkErrors: false },
      );
      if (response.status === 404) return;
      if (!response.ok || !response.body)
        throw new Error(`events ${response.status}`);
      attempt = 0;
      for await (const event of readSseJsonEvents<RunEvent>(
        response.body,
        45_000,
      )) {
        if (typeof event.seq === "number" && event.seq > lastSeq)
          lastSeq = event.seq;
        onEvent(event);
        if (isTerminalEvent(event)) terminal = true;
      }
    } catch (error) {
      if (signal.aborted) return;
      void error;
    }
    if (terminal || signal.aborted) return;
    attempt += 1;
    await new Promise((resolve) =>
      setTimeout(resolve, Math.min(8000, 500 * 2 ** attempt)),
    );
  }
}

export function isTerminalEvent(event: RunEvent): boolean {
  if (
    event.type === "run.completed" ||
    event.type === "run.failed" ||
    event.type === "run.cancelled" ||
    event.type === "run.interrupted"
  ) {
    return true;
  }
  if (event.type === "snapshot") {
    const status = (event.data.run as { status?: string } | undefined)?.status;
    return (
      status === "completed" ||
      status === "failed" ||
      status === "cancelled" ||
      status === "interrupted"
    );
  }
  return false;
}

export interface FlowCapabilitySet {
  python: string[];
  hosts: string[];
  mcp: boolean;
  providers: string[];
  triggers: boolean;
}

export interface CapabilityReport {
  needs: FlowCapabilitySet;
  pending: FlowCapabilitySet;
  approved: Partial<FlowCapabilitySet>;
  can_approve: boolean;
}

export const getCapabilities = (flowId: string) =>
  request<CapabilityReport>(`/${encodeURIComponent(flowId)}/capabilities`);

export const approveCapabilities = (flowId: string, shown: FlowCapabilitySet) =>
  request<CapabilityReport>(
    `/${encodeURIComponent(flowId)}/approve`,
    json("POST", shown),
  );

export interface FlowTrigger {
  node_id: string;
  kind: "schedule" | "webhook";
  enabled: boolean;
  path?: string;
  every_minutes?: number;
  cron?: string;
  cron_error?: string;
  last_fired_at?: number | null;
}

export const listTriggers = (flowId: string) =>
  request<{ triggers: FlowTrigger[]; owner_only: boolean }>(
    `/${encodeURIComponent(flowId)}/triggers`,
  );

export const rotateTrigger = (flowId: string, nodeId: string) =>
  request<{ triggers: FlowTrigger[]; owner_only: boolean }>(
    `/${encodeURIComponent(flowId)}/triggers/${encodeURIComponent(nodeId)}/rotate`,
    { method: "POST" },
  );
