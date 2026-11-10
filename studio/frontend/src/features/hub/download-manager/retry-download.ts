// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DownloadRequest,
  ManagedDownload,
} from "./download-manager-types";
import { isTrainingDownloadVariant } from "./training-downloads";

type RetrySubject = Pick<
  ManagedDownload,
  | "kind"
  | "repoId"
  | "variant"
  | "expectedBytes"
  | "inventoryKind"
  | "presentation"
  | "scopedFiles"
  | "checkpoint"
  | "downloadDir"
  | "external"
  | "state"
>;

/** Only Hub-started rows: engine / STT and training rows belong to the flows that started them. */
export function isRetryableDownload(job: RetrySubject): boolean {
  if (job.external || isTrainingDownloadVariant(job.variant)) return false;
  // The backend refuses a scope without its file list.
  return !job.variant?.startsWith("@") || (job.scopedFiles?.length ?? 0) > 0;
}

/** The start that produced *job*; a scoped variant must travel as scope_id, not gguf_variant. */
export function retryRequestFor(job: RetrySubject): DownloadRequest {
  const scopeId = job.variant?.startsWith("@") ? job.variant.slice(1) : null;
  return {
    kind: job.kind,
    repoId: job.repoId,
    variant: job.variant,
    expectedBytes: job.expectedBytes,
    ...(scopeId ? { scopeId } : {}),
    ...(job.inventoryKind ? { inventoryKind: job.inventoryKind } : {}),
    ...(job.presentation ? { presentation: job.presentation } : {}),
    ...(job.scopedFiles ? { files: job.scopedFiles } : {}),
    ...(job.checkpoint !== undefined ? { checkpoint: job.checkpoint } : {}),
    // Same rule as withDownloadDir: only a cancel pins its folder; a failure may have been the folder.
    ...(job.state === "cancelled" && job.downloadDir !== undefined
      ? { downloadDir: job.downloadDir }
      : {}),
  };
}
