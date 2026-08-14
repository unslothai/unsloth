// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** How far the host description has got: "pending" until the first request
 * settles, "ready" once real data arrives, "unavailable" when it settled with
 * none. Callers rendering a verdict about the host must not draw one from
 * anything but "ready". */
export type SystemInfoStatus = "pending" | "ready" | "unavailable";

/** The status after a request settles empty. Data already on screen stays, so a
 * failed poll cannot blank a good reading; only a placeholder moves, off
 * "pending", because with polling off nothing retries and "still checking"
 * would outlive the check. */
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
