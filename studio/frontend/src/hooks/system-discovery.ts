// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Render a verdict only from "ready". */
export type SystemInfoStatus = "pending" | "ready" | "unavailable";

/** A reading on screen stays; only a placeholder moves off "pending". */
export function settledFailureStatus(
  previous: SystemInfoStatus,
): SystemInfoStatus {
  return previous === "ready" ? "ready" : "unavailable";
}

interface InferenceGpuDiscovery {
  available: boolean;
  backend?: string;
}

export function shouldRetrySystemDiscovery(
  cacheIsCold: boolean,
  inferenceGpu: InferenceGpuDiscovery | undefined,
  retrySubscribers: number,
): boolean {
  if (retrySubscribers <= 0) {
    return false;
  }
  if (cacheIsCold) {
    return true;
  }
  return inferenceGpu?.backend === "vulkan" && !inferenceGpu.available;
}

const VULKAN_RETRY_BASE_MS = 3000;
const VULKAN_RETRY_MAX_MS = 60_000;

export function vulkanRetryDelayMs(attempt: number): number {
  return Math.min(
    VULKAN_RETRY_BASE_MS * 2 ** Math.max(0, attempt),
    VULKAN_RETRY_MAX_MS,
  );
}

export const STALE_AFTER_FAILED_POLLS = 2;

export function nextFailedPolls(previous: number, ok: boolean): number {
  return ok ? 0 : previous + 1;
}

export function isReadingStale(
  status: SystemInfoStatus,
  failedPolls: number,
  polling: boolean,
): boolean {
  return (
    polling && status === "ready" && failedPolls >= STALE_AFTER_FAILED_POLLS
  );
}
