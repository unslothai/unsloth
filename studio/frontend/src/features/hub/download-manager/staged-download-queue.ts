// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { scopedVariant, type ManagedDownload } from "./download-manager-types";
import type { StagedDownloadEntry } from "./use-staged-download";

export interface StagedQueue {
  scopeId: string;
  entries: StagedDownloadEntry[];
}

// Display state only. The owning hook still controls advancement and loading.
export const useStagedDownloadQueues = create<{
  queues: Record<string, StagedQueue>;
}>(() => ({ queues: {} }));

export function publishStagedQueue(owner: string, queue: StagedQueue | null) {
  useStagedDownloadQueues.setState(({ queues }) => {
    if (queue?.entries.length) return { queues: { ...queues, [owner]: queue } };
    if (!(owner in queues)) return { queues };
    const next = { ...queues };
    delete next[owner];
    return { queues: next };
  });
}

export function queuedStagedEntries(
  queues: Record<string, StagedQueue>,
  jobs: Record<
    string,
    Pick<
      ManagedDownload,
      "kind" | "repoId" | "variant" | "state" | "scopedFiles"
    >
  >,
) {
  return Object.entries(queues).flatMap(([owner, { scopeId, entries }]) =>
    entries.flatMap((entry, index) => {
      const active = Object.values(jobs).some(
        (job) =>
          job.kind === "model" &&
          job.repoId === entry.repoId &&
          job.variant === (entry.ggufVariant ?? scopedVariant(scopeId)) &&
          (job.state === "running" || job.state === "cancelling") &&
          (entry.ggufVariant ||
            (entry.files.length === job.scopedFiles?.length &&
              entry.files.every((file) => job.scopedFiles?.includes(file)))),
      );
      return active ? [] : [{ ...entry, planId: `staged:${owner}:${index}` }];
    }),
  );
}
