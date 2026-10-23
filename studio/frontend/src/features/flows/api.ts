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

/** The Vite harness swaps in an in-memory backend; the app always uses authFetch. */
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

export interface FlowPlan {
  models: PlanModel[];
  gpu_ops: number;
  cached_nodes: number;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
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

export interface ImportReport {
  missing_models: PlanModel[];
  warnings: string[];
  baked_nodes: string[];
  license?: string | null;
  content_rating?: string | null;
  author?: string | null;
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

/** Same-origin URL for a MediaValue; authenticated media is fetched as a blob by `useMediaUrl`. */
export function mediaPath(media: MediaValue): string | null {
  if (isStudioMediaUrl(media.url)) return media.url;
  if (media.ref.startsWith("sha256:") || media.ref.startsWith("flowasset:")) {
    return `${FLOWS_API}/assets/${encodeURIComponent(media.ref)}`;
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
