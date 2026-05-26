// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, hfTokenHeader } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type { TransportMode } from "./constants";

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    const formatted = formatFastApiDetail(detail);
    if (status === 405) {
      return `${formatted || "Method Not Allowed"} - the Studio backend did not accept this API method. Restart Studio so the frontend and backend are on the same build.`;
    }
    if (formatted) return formatted;
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  if (status === 405) {
    return "Method Not Allowed - the Studio backend did not accept this API method. Restart Studio so the frontend and backend are on the same build.";
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

export type DownloadJobState =
  | "idle"
  | "running"
  | "complete"
  | "error"
  | "cancelled"
  | "cancelling";

export interface DownloadJobStatus {
  state: DownloadJobState;
  error?: string | null;
}

export type DownloadStartState = DownloadJobState | "deleting";

export interface DownloadStartResult {
  state: DownloadStartState;
  accepted: boolean;
  generation?: number;
}

export interface ActiveModelDownload {
  variant: string | null;
  state: DownloadJobState;
}

const ACTIVE_MODEL_DOWNLOADS_DEDUPE_MS = 750;
const ACTIVE_MODEL_DOWNLOADS_CACHE_MAX = 200;
const activeModelDownloadsInFlight = new Map<
  string,
  Promise<ActiveModelDownload[]>
>();
const activeModelDownloadsCache = new Map<
  string,
  { expiresAt: number; downloads: ActiveModelDownload[] }
>();

function abortError(signal: AbortSignal): DOMException {
  return signal.reason instanceof DOMException
    ? signal.reason
    : new DOMException("The operation was aborted.", "AbortError");
}

function withAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}

function pruneActiveModelDownloadsCache(now = Date.now()): void {
  for (const [key, entry] of activeModelDownloadsCache) {
    if (entry.expiresAt <= now) activeModelDownloadsCache.delete(key);
  }
  while (activeModelDownloadsCache.size > ACTIVE_MODEL_DOWNLOADS_CACHE_MAX) {
    const oldest = activeModelDownloadsCache.keys().next().value;
    if (oldest === undefined) break;
    activeModelDownloadsCache.delete(oldest);
  }
}

export type TransportMarker = TransportMode | null;

export interface TransportStatus {
  has_partial: boolean;
  last_transport: TransportMarker;
  resumable: boolean;
}

export interface DownloadTransportCapability {
  available: boolean;
  reason: string | null;
}

export interface DownloadTransportCapabilities {
  http: DownloadTransportCapability;
  xet: DownloadTransportCapability;
}

export interface DownloadProgressResponse {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
  cache_path: string | null;
}

export type DownloadProgressOptions = {
  signal?: AbortSignal;
  hfToken?: string | null;
  expectedBytes?: number;
};

export type GgufDownloadProgressOptions = DownloadProgressOptions & {
  variant: string;
};

const DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK: DownloadTransportCapabilities = {
  http: { available: true, reason: null },
  xet: {
    available: false,
    reason: "Couldn't verify Xet support with the Studio backend.",
  },
};
let downloadTransportCapabilitiesCache: DownloadTransportCapabilities | null =
  null;
let downloadTransportCapabilitiesInFlight: Promise<DownloadTransportCapabilities> | null =
  null;

function normalizeDownloadTransportCapability(
  value: unknown,
  fallback: DownloadTransportCapability,
): DownloadTransportCapability {
  if (!value || typeof value !== "object") {
    return fallback;
  }
  const candidate = value as { available?: unknown; reason?: unknown };
  return {
    available:
      typeof candidate.available === "boolean"
        ? candidate.available
        : fallback.available,
    reason:
      typeof candidate.reason === "string"
        ? candidate.reason
        : candidate.reason === null
          ? null
          : fallback.reason,
  };
}

function normalizeDownloadTransportCapabilities(
  value: unknown,
): DownloadTransportCapabilities {
  if (!value || typeof value !== "object") {
    return DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK;
  }
  const candidate = value as { http?: unknown; xet?: unknown };
  return {
    http: normalizeDownloadTransportCapability(candidate.http, {
      available: true,
      reason: null,
    }),
    xet: normalizeDownloadTransportCapability(
      candidate.xet,
      DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK.xet,
    ),
  };
}

export async function getDownloadTransportCapabilities(options: {
  force?: boolean;
} = {}): Promise<DownloadTransportCapabilities> {
  if (!options.force && downloadTransportCapabilitiesCache) {
    return downloadTransportCapabilitiesCache;
  }
  if (!options.force && downloadTransportCapabilitiesInFlight) {
    return downloadTransportCapabilitiesInFlight;
  }
  const request = authFetch("/api/studio/download-transport-capabilities")
    .then(parseJsonOrThrow<unknown>)
    .then(normalizeDownloadTransportCapabilities)
    .then((capabilities) => {
      downloadTransportCapabilitiesCache = capabilities;
      return capabilities;
    })
    .catch(() => DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK)
    .finally(() => {
      if (downloadTransportCapabilitiesInFlight === request) {
        downloadTransportCapabilitiesInFlight = null;
      }
    });
  downloadTransportCapabilitiesInFlight = request;
  return request;
}

export function __resetDownloadTransportCapabilitiesForTests(): void {
  downloadTransportCapabilitiesCache = null;
  downloadTransportCapabilitiesInFlight = null;
}

export async function startModelDownload(payload: {
  repo_id: string;
  gguf_variant?: string | null;
  hf_token?: string | null;
  use_xet?: boolean;
}): Promise<DownloadStartResult & { job_key: string }> {
  const { hf_token, ...body } = payload;
  const headers = {
    "Content-Type": "application/json",
    ...hfTokenHeader(hf_token),
  };
  const response = await authFetch("/api/models/download", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow(response);
}

export async function cancelModelDownload(payload: {
  repo_id: string;
  gguf_variant?: string | null;
  generation?: number | null;
  signal?: AbortSignal;
}): Promise<{ job_key: string; state: DownloadJobState }> {
  const { signal, ...body } = payload;
  const response = await authFetch("/api/models/download/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow(response);
}

export async function getModelDownloadStatus(
  repoId: string,
  ggufVariant?: string | null,
  signal?: AbortSignal,
): Promise<DownloadJobStatus> {
  const params = new URLSearchParams({ repo_id: repoId });
  if (ggufVariant) params.set("gguf_variant", ggufVariant);
  const response = await authFetch(`/api/models/download-status?${params}`, {
    signal,
  });
  return parseJsonOrThrow<DownloadJobStatus>(response);
}

export async function getActiveModelDownloads(
  repoId: string,
  signal?: AbortSignal,
): Promise<ActiveModelDownload[]> {
  const key = repoId.trim();
  const cached = activeModelDownloadsCache.get(key);
  const now = Date.now();
  if (cached && cached.expiresAt > now) {
    return withAbort(Promise.resolve(cached.downloads), signal);
  }

  let request = activeModelDownloadsInFlight.get(key);
  if (!request) {
    request = (async () => {
      const params = new URLSearchParams({ repo_id: key });
      const response = await authFetch(`/api/models/active-downloads?${params}`);
      const data = await parseJsonOrThrow<{
        downloads: ActiveModelDownload[];
      }>(response);
      activeModelDownloadsCache.set(key, {
        expiresAt: Date.now() + ACTIVE_MODEL_DOWNLOADS_DEDUPE_MS,
        downloads: data.downloads,
      });
      pruneActiveModelDownloadsCache();
      return data.downloads;
    })().finally(() => {
      activeModelDownloadsInFlight.delete(key);
    });
    activeModelDownloadsInFlight.set(key, request);
  }
  return withAbort(request, signal);
}

export async function startDatasetDownload(payload: {
  repo_id: string;
  hf_token?: string | null;
  use_xet?: boolean;
}): Promise<DownloadStartResult & { repo_id: string }> {
  const { hf_token, ...body } = payload;
  const headers = {
    "Content-Type": "application/json",
    ...hfTokenHeader(hf_token),
  };
  const response = await authFetch("/api/datasets/download", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow(response);
}

export async function cancelDatasetDownload(payload: {
  repo_id: string;
  generation?: number | null;
  signal?: AbortSignal;
}): Promise<{ repo_id: string; state: DownloadJobState }> {
  const { signal, ...body } = payload;
  const response = await authFetch("/api/datasets/download/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow(response);
}

export async function getDatasetDownloadStatus(
  repoId: string,
  signal?: AbortSignal,
): Promise<DownloadJobStatus> {
  const params = new URLSearchParams({ repo_id: repoId });
  const response = await authFetch(`/api/datasets/download-status?${params}`, {
    signal,
  });
  return parseJsonOrThrow<DownloadJobStatus>(response);
}

export async function getModelTransportStatus(
  repoId: string,
): Promise<TransportStatus> {
  const params = new URLSearchParams({ repo_id: repoId });
  const response = await authFetch(`/api/models/transport-status?${params}`);
  return parseJsonOrThrow<TransportStatus>(response);
}

export async function getDatasetTransportStatus(
  repoId: string,
): Promise<TransportStatus> {
  const params = new URLSearchParams({ repo_id: repoId });
  const response = await authFetch(`/api/datasets/transport-status?${params}`);
  return parseJsonOrThrow<TransportStatus>(response);
}

export async function getGgufDownloadProgress(
  repoId: string,
  options: GgufDownloadProgressOptions,
): Promise<{
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
}> {
  const { expectedBytes = 0, hfToken, signal, variant } = options;
  const params = new URLSearchParams({
    repo_id: repoId,
    variant,
    expected_bytes: String(expectedBytes),
  });
  const response = await authFetch(`/api/models/gguf-download-progress?${params}`, {
    headers: hfTokenHeader(hfToken),
    signal,
  });
  return parseJsonOrThrow(response);
}

export async function getDownloadProgress(
  repoId: string,
  options: DownloadProgressOptions,
): Promise<DownloadProgressResponse> {
  const { expectedBytes = 0, hfToken, signal } = options;
  const params = new URLSearchParams({ repo_id: repoId });
  if (expectedBytes > 0) params.set("expected_bytes", String(expectedBytes));
  const response = await authFetch(`/api/models/download-progress?${params}`, {
    headers: hfTokenHeader(hfToken),
    signal,
  });
  return parseJsonOrThrow(response);
}

export async function getDatasetDownloadProgress(
  repoId: string,
  options: DownloadProgressOptions,
): Promise<DownloadProgressResponse> {
  const { expectedBytes = 0, hfToken, signal } = options;
  const params = new URLSearchParams({ repo_id: repoId });
  if (expectedBytes > 0) params.set("expected_bytes", String(expectedBytes));
  const response = await authFetch(`/api/datasets/download-progress?${params}`, {
    headers: hfTokenHeader(hfToken),
    signal,
  });
  return parseJsonOrThrow(response);
}
