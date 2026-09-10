// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ManagedDownload } from "./download-manager-types";

/** Project plan-wide cache counters onto the sole artifact still transferring. */
export function presentedProgress(
  job: Pick<
    ManagedDownload,
    "downloadedBytes" | "expectedBytes" | "fraction" | "presentation"
  >,
) {
  const presentation = job.presentation;
  if (!presentation) {
    return {
      expectedBytes: job.expectedBytes,
      downloadedBytes: job.downloadedBytes,
      fraction: job.fraction,
    };
  }
  const expectedBytes = presentation.expectedBytes;
  const cachedPlanPrefix = Math.max(0, job.expectedBytes - expectedBytes);
  const downloadedBytes = Math.min(
    expectedBytes,
    Math.max(0, job.downloadedBytes - cachedPlanPrefix),
  );
  return {
    expectedBytes,
    downloadedBytes,
    fraction: expectedBytes > 0 ? downloadedBytes / expectedBytes : job.fraction,
  };
}
