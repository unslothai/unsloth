// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getHfToken } from "../stores/hf-token-store";
import { toast } from "@/lib/toast";
import {
  type DownloadStartResult,
  type DownloadStartState,
  cancelDatasetDownload,
  cancelModelDownload,
  getDatasetDownloadProgress,
  getDatasetDownloadStatus,
  getDatasetTransportStatus,
  getDownloadProgress,
  getDownloadTransportCapabilities,
  getGgufDownloadProgress,
  getModelDownloadStatus,
  getModelTransportStatus,
  startDatasetDownload,
  startModelDownload,
} from "./api";
import {
  POLL_REQUEST_TIMEOUT_MS,
  TRANSPORT_STATUS_RETRY_DELAY_MS,
  TRANSPORT_STATUS_TIMEOUT_MS,
} from "./download-manager-config";
import { DOWNLOAD_KIND, TRANSPORT, type TransportMode } from "./constants";
import type {
  DownloadRequest,
  ManagedDownload,
  ProgressLike,
} from "./download-manager-types";
import {
  type PollSignal,
  disposableTimeoutSignal,
  pollSignal,
} from "../lib/abort-signals";

let lastXetUnavailableWarningReason: string | null = null;

export async function withPollRequestTimeout<T>(
  parent: AbortController | null,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const poll: PollSignal = parent
    ? pollSignal(parent.signal, POLL_REQUEST_TIMEOUT_MS)
    : disposableTimeoutSignal(POLL_REQUEST_TIMEOUT_MS);
  try {
    return await request(poll.signal);
  } finally {
    poll.dispose();
  }
}

export function apiGetStatus(job: ManagedDownload, signal: AbortSignal) {
  return job.kind === DOWNLOAD_KIND.DATASET
    ? getDatasetDownloadStatus(job.repoId, signal)
    : getModelDownloadStatus(job.repoId, job.variant, signal);
}

export function apiGetProgress(
  job: ManagedDownload,
  signal: AbortSignal,
): Promise<ProgressLike> {
  const token = getHfToken() || null;
  if (job.kind === DOWNLOAD_KIND.DATASET) {
    return getDatasetDownloadProgress(job.repoId, {
      signal,
      hfToken: token,
      expectedBytes: job.expectedBytes,
    });
  }
  if (job.variant) {
    return getGgufDownloadProgress(job.repoId, {
      variant: job.variant,
      expectedBytes: job.expectedBytes,
      hfToken: token,
      signal,
    });
  }
  return getDownloadProgress(job.repoId, {
    signal,
    hfToken: token,
    expectedBytes: job.expectedBytes,
  });
}

export function apiStart(
  req: DownloadRequest,
  useXet: boolean,
  hfToken: string | null,
): Promise<DownloadStartResult> {
  return req.kind === DOWNLOAD_KIND.DATASET
    ? startDatasetDownload({
        repo_id: req.repoId,
        hf_token: hfToken,
        use_xet: useXet,
      })
    : startModelDownload({
        repo_id: req.repoId,
        gguf_variant: req.variant,
        hf_token: hfToken,
        use_xet: useXet,
      });
}

export function apiCancel(job: ManagedDownload, signal?: AbortSignal) {
  const generation = Number.isSafeInteger(job.serverGeneration)
    ? job.serverGeneration
    : undefined;
  return job.kind === DOWNLOAD_KIND.DATASET
    ? cancelDatasetDownload({ repo_id: job.repoId, generation, signal })
    : cancelModelDownload({
        repo_id: job.repoId,
        gguf_variant: job.variant,
        generation,
        signal,
      });
}

export function apiCancelRequest(
  req: DownloadRequest,
  generation: number | undefined,
  signal?: AbortSignal,
) {
  if (!Number.isSafeInteger(generation)) return Promise.resolve();
  return req.kind === DOWNLOAD_KIND.DATASET
    ? cancelDatasetDownload({ repo_id: req.repoId, generation, signal })
    : cancelModelDownload({
        repo_id: req.repoId,
        gguf_variant: req.variant,
        generation,
        signal,
      });
}

function apiTransportStatus(
  req: DownloadRequest,
  signal?: AbortSignal,
) {
  const token = getHfToken() || null;
  return req.kind === DOWNLOAD_KIND.DATASET
    ? getDatasetTransportStatus(req.repoId, signal)
    : getModelTransportStatus(req.repoId, req.variant, token, signal);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });
}

export async function apiTransportStatusWithRetry(
  req: DownloadRequest,
): Promise<Awaited<ReturnType<typeof apiTransportStatus>>> {
  const request = async () => {
    const timeout = disposableTimeoutSignal(TRANSPORT_STATUS_TIMEOUT_MS);
    try {
      return await apiTransportStatus(req, timeout.signal);
    } finally {
      timeout.dispose();
    }
  };
  try {
    return await request();
  } catch {
    await wait(TRANSPORT_STATUS_RETRY_DELAY_MS);
    return request();
  }
}

export async function effectiveTransportMode(
  preferred: TransportMode,
): Promise<TransportMode> {
  if (preferred !== TRANSPORT.XET) {
    return preferred;
  }
  const capabilities = await getDownloadTransportCapabilities();
  if (capabilities.xet.available === true) {
    lastXetUnavailableWarningReason = null;
    return preferred;
  }
  if (capabilities.xet.available === null) {
    return preferred;
  }
  const reason =
    capabilities.xet.reason ?? "Unsloth will use HTTP downloads instead.";
  if (lastXetUnavailableWarningReason !== reason) {
    lastXetUnavailableWarningReason = reason;
    toast.warning("Xet download transport unavailable", {
      description: reason,
    });
  }
  return TRANSPORT.HTTP;
}

export function describeUnacceptedStart(state: DownloadStartState): string {
  switch (state) {
    case "deleting":
      return "This repository is being removed. Try again once it finishes.";
    case "running":
    case "cancelling":
      return "Another download for this repository is already in progress. Wait for it to finish or cancel it first.";
    default:
      return "The download could not be started. Try again in a moment.";
  }
}

export function accessErrorMessage(raw: string): string | null {
  const lower = raw.toLowerCase();
  const hasAccessSignal =
    lower.includes("unauthorized") ||
    lower.includes("forbidden") ||
    lower.includes("gated") ||
    lower.includes("repository not found");
  if (!hasAccessSignal) return null;
  return "Couldn't access this Hugging Face repo with the token used for this download. Update the HF token and restart the download, or delete the partial download if you no longer need it.";
}

export const pollAccessErrorMessage = accessErrorMessage;

export function normalizeDownloadError(
  error: unknown,
  fallback = "Failed to start download",
): string {
  const raw =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : fallback;
  return accessErrorMessage(raw) ?? raw;
}

export function isRequestTimeout(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "TimeoutError")
  );
}

export function resetDownloadApiAdapterState(): void {
  lastXetUnavailableWarningReason = null;
}
