// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { DOWNLOAD_KIND, type DownloadKind } from "./constants";
import { ACTIVE_STATES } from "./download-manager-config";
import { jobKeyOf } from "./download-manager-state";
import type {
  DownloadPresentation,
  DownloadRequest,
  ManagedDownload,
} from "./download-manager-types";

export interface PendingStart {
  key: string;
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  presentation?: DownloadPresentation;
  checkpoint?: boolean;
  scopedFiles?: string[];
}

export const usePendingStarts = create<{
  starts: Record<string, PendingStart>;
}>(() => ({ starts: {} }));

export function publishPendingStart(
  key: string,
  req: DownloadRequest,
): PendingStart {
  const start: PendingStart = {
    key,
    kind: req.kind,
    repoId: req.repoId,
    variant: req.variant,
    ...(req.presentation ? { presentation: req.presentation } : {}),
    ...(req.checkpoint !== undefined ? { checkpoint: req.checkpoint } : {}),
    ...(req.files ? { scopedFiles: req.files } : {}),
  };
  usePendingStarts.setState(({ starts }) => ({
    starts: { ...starts, [key]: start },
  }));
  return start;
}

export function retirePendingStart(start: PendingStart): void {
  usePendingStarts.setState(({ starts }) => {
    if (starts[start.key] !== start) return { starts };
    const next = { ...starts };
    delete next[start.key];
    return { starts: next };
  });
}

export function resetPendingStarts(): void {
  usePendingStarts.setState({ starts: {} });
}

export function visiblePendingStarts(
  starts: Record<string, PendingStart>,
  jobs: Record<string, Pick<ManagedDownload, "state">>,
  queued: readonly { repoId: string; variant: string | null }[],
): PendingStart[] {
  const queuedKeys = new Set(
    queued.map((entry) =>
      jobKeyOf(DOWNLOAD_KIND.MODEL, entry.repoId, entry.variant),
    ),
  );
  return Object.values(starts).filter((start) => {
    const job = jobs[start.key];
    if (job && ACTIVE_STATES.has(job.state)) return false;
    return !queuedKeys.has(start.key);
  });
}
