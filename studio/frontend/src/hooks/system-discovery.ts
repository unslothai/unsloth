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
