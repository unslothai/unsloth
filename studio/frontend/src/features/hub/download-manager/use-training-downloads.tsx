// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_SESSION_STORED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth/session";
import {
  ModelLoadDescription,
  MODEL_LOAD_TOAST_CLASSNAMES,
} from "@/features/chat/components/model-load-status";
import { toast } from "@/lib/toast";
import { formatBytes } from "../lib/format";
import { getAllActiveModelDownloads } from "./api";
import { DOWNLOAD_KIND } from "./constants";
import {
  getState,
  jobKeyOf,
  removeJob,
  useDownloadManagerStore,
} from "./download-manager-state";
import { adoptJob } from "./poll-loop";
import {
  isTrainingDownloadVariant,
  TrainingDownloadDiscovery,
} from "./training-downloads";

/** Discover registry jobs globally; the regular manager owns polling and cancellation. */
export function useTrainingDownloads(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const session = () =>
      `${getAuthSessionEpoch()}:${localStorage.getItem(AUTH_SESSION_MARK_KEY)}`;
    const notices = new Set<string>();
    const dismissed = new Set<string>();
    const clearNotices = () => {
      for (const id of notices) toast.dismiss(id);
      notices.clear();
      dismissed.clear();
    };
    const render = () => {
      const live = new Set<string>();
      for (const job of Object.values(getState().jobs)) {
        if (!isTrainingDownloadVariant(job.variant)) continue;
        const id = `training-download:${job.key}`;
        if (job.state !== "running" && job.state !== "cancelling") continue;
        live.add(id);
        if (dismissed.has(id)) continue;
        notices.add(id);
        const known = job.expectedBytes > 0;
        toast(null, {
          id,
          duration: Infinity,
          closeButton: true,
          classNames: MODEL_LOAD_TOAST_CLASSNAMES,
          description: (
            <ModelLoadDescription
              title={
                job.state === "cancelling"
                  ? "Stopping training download…"
                  : known
                    ? `Downloading ${job.repoId.split("/").at(-1)}…`
                    : `Preparing ${job.repoId.split("/").at(-1)} download…`
              }
              message="Preparing model files for training…"
              progressPercent={known ? job.fraction * 100 : null}
              progressLabel={
                known
                  ? `${formatBytes(job.downloadedBytes)} of ${formatBytes(job.expectedBytes)} · current files`
                  : null
              }
            />
          ),
          onDismiss: () => {
            dismissed.add(id);
            notices.delete(id);
          },
        });
      }
      for (const id of notices) {
        if (!live.has(id)) {
          toast.dismiss(id);
          notices.delete(id);
        }
      }
    };
    const unsubscribe = useDownloadManagerStore.subscribe(render);
    const discovery = new TrainingDownloadDiscovery({
      fetch: (signal) => getAllActiveModelDownloads(signal, { fresh: true }),
      session,
      adopt: (job) => {
        const repoId = job.repo_id!.trim();
        const key = jobKeyOf(DOWNLOAD_KIND.MODEL, repoId, job.variant);
        const existing = getState().jobs[key];
        if (
          existing &&
          existing.serverGeneration === job.generation &&
          (existing.state === "running" || existing.state === "cancelling")
        )
          return;
        adoptJob(
          {
            kind: DOWNLOAD_KIND.MODEL,
            repoId,
            variant: job.variant,
            expectedBytes: 0,
          },
          job.generation,
          job.state,
        );
      },
    });
    const poll = async () => {
      if ((await discovery.poll()) && !disposed)
        timer = setTimeout(() => void poll(), 1500);
    };
    const reset = () => {
      discovery.reset();
      clearTimeout(timer);
      for (const job of Object.values(getState().jobs)) {
        if (isTrainingDownloadVariant(job.variant)) removeJob(job.key);
      }
      clearNotices();
      void poll();
    };
    const storage = (event: StorageEvent) => {
      if (event.key === AUTH_SESSION_MARK_KEY) reset();
    };
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, reset);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, reset);
    window.addEventListener("storage", storage);
    render();
    void poll();
    return () => {
      disposed = true;
      discovery.dispose();
      clearTimeout(timer);
      unsubscribe();
      clearNotices();
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, reset);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, reset);
      window.removeEventListener("storage", storage);
    };
  }, [enabled]);
}
