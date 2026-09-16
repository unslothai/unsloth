// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantDetail } from "../inventory";
import type {
  DownloadPresentation,
  ManagedDownload,
} from "./download-manager-types";

/** Describe a sole missing drafter as the transfer, not as the cached model. */
export function pendingDrafterPresentation(
  variant: GgufVariantDetail | null | undefined,
): DownloadPresentation | undefined {
  const filename = variant?.pending_drafter_filename?.trim();
  const expectedBytes = variant?.pending_drafter_size_bytes ?? 0;
  if (!filename || !Number.isFinite(expectedBytes) || expectedBytes <= 0) {
    return undefined;
  }
  const basename = filename.replaceAll("\\", "/").split("/").at(-1) ?? filename;
  const lower = basename.toLowerCase();
  const label = lower.startsWith("mtp-")
    ? "MTP companion"
    : lower.startsWith("dspark-") || lower.startsWith("dflash-")
      ? "Draft companion"
      : "Model companion";
  return { label, filename: basename, expectedBytes };
}

export function stabilizeDownloadPresentation(
  presentation: DownloadPresentation | undefined,
  planExpectedBytes: number,
): DownloadPresentation | undefined {
  if (
    !presentation ||
    presentation.cachedPlanPrefixBytes !== undefined ||
    !Number.isFinite(planExpectedBytes) ||
    planExpectedBytes < presentation.expectedBytes
  ) {
    return presentation;
  }
  return {
    ...presentation,
    cachedPlanPrefixBytes: Math.max(
      0,
      planExpectedBytes - presentation.expectedBytes,
    ),
  };
}

/** A backend-active adoption has no UI metadata, so retain the persisted scope. */
export function presentationForJobStart(
  requested: DownloadPresentation | undefined,
  existing: DownloadPresentation | undefined,
  planExpectedBytes: number,
  adopt: boolean,
): DownloadPresentation | undefined {
  return stabilizeDownloadPresentation(
    requested ?? (adopt ? existing : undefined),
    planExpectedBytes,
  );
}

export function presentationForExpectedBytesUpdate(
  presentation: DownloadPresentation | undefined,
  previousPlanExpectedBytes: number,
  nextPlanExpectedBytes: number,
): DownloadPresentation | undefined {
  if (!presentation || presentation.cachedPlanPrefixBytes !== undefined) {
    return presentation;
  }
  const planExpectedBytes =
    previousPlanExpectedBytes >= presentation.expectedBytes
      ? previousPlanExpectedBytes
      : nextPlanExpectedBytes;
  return stabilizeDownloadPresentation(presentation, planExpectedBytes);
}

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
  const cachedPlanPrefix =
    job.expectedBytes <= expectedBytes
      ? 0
      : (presentation.cachedPlanPrefixBytes ??
        Math.max(0, job.expectedBytes - expectedBytes));
  const downloadedBytes = Math.min(
    expectedBytes,
    Math.max(0, job.downloadedBytes - cachedPlanPrefix),
  );
  return {
    expectedBytes,
    downloadedBytes,
    fraction:
      expectedBytes > 0 ? downloadedBytes / expectedBytes : job.fraction,
  };
}
