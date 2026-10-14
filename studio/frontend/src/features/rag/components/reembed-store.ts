// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { cancelReembed, getReembedStatus, startReembed } from "../api/rag-api";
import type { ReembedJob, ReembedStatus } from "../types/rag";

const POLL_MS = 1500;

interface ReembedStore {
  status: ReembedStatus | null;
  load: () => Promise<ReembedStatus | null>;
  start: () => Promise<void>;
  /** Resolves false when the request failed, so Stop can be offered again. */
  cancel: () => Promise<boolean>;
}

let pollTimer: ReturnType<typeof setTimeout> | null = null;
/** A job seen running here, so its end is announced once. */
let watchedJobId: string | null = null;
let clock = 0;
/** Replies sent before this may describe an older server state: it is when the last applied reply
 * arrived, or when a failed start or cancel left the server's state unknown. */
let freshAfter = 0;
/** The shown state is known to be outdated (a dropped reply, a failed start or cancel, or the counts
 * a finished job changed) until a reply applies; failed reads keep retrying meanwhile. */
let reconciling = false;

/** Applies the reply unless it was sent before `freshAfter`, which would roll the job back or
 * announce its end twice. Such a reply returns false, and the caller reads the current state. */
async function request<T>(
  send: () => Promise<T>,
  apply: (value: T) => void,
): Promise<boolean> {
  const sentAt = ++clock;
  try {
    const value = await send();
    const arrivedAt = ++clock;
    if (sentAt <= freshAfter) {
      reconciling = true;
      return false;
    }
    freshAfter = arrivedAt;
    reconciling = false;
    apply(value);
    return true;
  } catch (error) {
    // A failed request must not end the polling that reports the job's progress and end.
    if (watchedJobId !== null || reconciling) schedulePoll();
    throw error;
  }
}

function setJob(job: ReembedJob): void {
  useReembedStore.setState((state) => ({
    status: { staleDocuments: 0, staleArchived: 0, ...state.status, job },
  }));
  follow(job);
  if (job.status !== "running") {
    // Only a status read carries the stale counts a finished job changed.
    reconciling = true;
    void useReembedStore.getState().load();
  }
}

/** Shared by every surface, so a job started from a toast shows its progress in Settings. */
export const useReembedStore = create<ReembedStore>(() => ({
  status: null,
  load: async (): Promise<ReembedStatus | null> => {
    try {
      const applied = await request(getReembedStatus, (status) => {
        useReembedStore.setState({ status });
        follow(status.job);
      });
      return applied
        ? useReembedStore.getState().status
        : await useReembedStore.getState().load();
    } catch {
      return null;
    }
  },
  start: async () => {
    try {
      if (!(await request(startReembed, ({ job }) => setJob(job)))) {
        await useReembedStore.getState().load();
      }
    } catch (error) {
      markOutdated();
      toast.error(translate("settings.general.rag.reembed.startFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  },
  cancel: async () => {
    try {
      if (!(await request(cancelReembed, ({ job }) => setJob(job)))) {
        await useReembedStore.getState().load();
      }
      return true;
    } catch {
      markOutdated();
      return false;
    }
  },
}));

/** A failed start or cancel may still have reached the server, so read status until a read applies. */
function markOutdated(): void {
  freshAfter = ++clock;
  reconciling = true;
  schedulePoll();
}

function schedulePoll(): void {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = setTimeout(() => void useReembedStore.getState().load(), POLL_MS);
}

function follow(job: ReembedJob | null): void {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
  if (job?.status === "running") {
    watchedJobId = job.id;
    schedulePoll();
    return;
  }
  if (!job || job.id !== watchedJobId) return;
  watchedJobId = null;
  if (job.status === "failed") {
    toast.error(translate("settings.general.rag.reembed.stopped"), {
      description: job.error ?? undefined,
    });
  } else if (job.status === "completed" && job.failed > 0) {
    toast.warning(
      translate("settings.general.rag.reembed.finishedWithFailures", {
        count: job.failed,
      }),
      { description: job.error ?? undefined },
    );
  } else if (job.status === "completed") {
    toast.success(translate("settings.general.rag.reembed.finished"));
  }
}

/** What is stale, in words; null when nothing is. */
export function staleDescription(status: ReembedStatus): string | null {
  if (status.staleDocuments === 1) {
    return translate("settings.general.rag.reembed.staleOne");
  }
  if (status.staleDocuments > 1) {
    return translate("settings.general.rag.reembed.staleMany", {
      count: status.staleDocuments,
    });
  }
  return status.staleArchived > 0
    ? translate("settings.general.rag.reembed.staleArchived")
    : null;
}

/** Success toast for a saved embedding model, offering Re-embed when existing documents are now stale. */
export async function toastEmbeddingModelSaved(title: string): Promise<void> {
  const status = await useReembedStore.getState().load();
  if (status === null) {
    toast.success(title, {
      description: translate("settings.general.rag.reindexWarning"),
    });
    return;
  }
  const description = staleDescription(status);
  if (description === null || status.job?.status === "running") {
    // A running job follows the current model, so it already covers the switch.
    toast.success(title, description ? { description } : undefined);
    return;
  }
  toast.success(title, {
    description,
    duration: 10_000,
    action: {
      label: translate("settings.general.rag.reembed.action"),
      onClick: () => void useReembedStore.getState().start(),
    },
  });
}
