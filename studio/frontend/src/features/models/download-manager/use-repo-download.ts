// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import type { TransportConflictInfo } from "../components/transport-conflict-dialog";
import {
  type DownloadKind,
  type JobListeners,
  downloadManager,
  jobKeyOf,
  repoKeyOf,
  selectActiveJob,
  subscribeJobListeners,
  useDownloadManagerStore,
} from "./download-manager-store";

export interface DownloadJobProgress {
  variant: string | null;
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

export interface DownloadJob {
  progress: DownloadJobProgress | null;
  bytesPerSec: number;
  cancelling: boolean;
  transportConflict: TransportConflictInfo | null;
  startDownload: (
    variant: string | null,
    expectedBytes: number,
    useXetOverride?: boolean,
  ) => void;
  requestStartDownload: (variant: string | null, expectedBytes: number) => void;
  cancelDownload: (variant: string | null) => void;
  adoptRunningJob: (variant: string | null, expectedBytes: number) => void;
  setExpectedBytes: (bytes: number) => void;
  resumeConflict: () => void;
  restartConflict: () => void;
  cancelConflict: () => void;
}

export interface RepoDownloadConfig {
  kind: DownloadKind;
  repoId: string;
  onComplete?: JobListeners["onComplete"];
  onCancelled?: JobListeners["onCancelled"];
  onError?: JobListeners["onError"];
  // Attach to a no-variant download already running on the backend. Per-variant
  // surfaces (GGUF) resolve their variant and adopt themselves.
  autoAdopt?: boolean;
}

/**
 * Binds a single download surface (one repo, optionally per GGUF variant) to the
 * global download manager. Returns the same shape the card UI consumed before
 * the manager existed, but the job state and polling now live in the store, so a
 * download keeps running and stays visible after the card unmounts.
 */
export function useRepoDownload(config: RepoDownloadConfig): DownloadJob {
  const { kind, repoId, onComplete, onCancelled, onError, autoAdopt } = config;

  // Keep the latest callbacks in a ref so the subscription only re-binds when
  // the repo identity changes, not on every render that passes fresh closures.
  const handlersRef = useRef<JobListeners>({});
  useEffect(() => {
    handlersRef.current = { onComplete, onCancelled, onError };
  });
  useEffect(() => {
    return subscribeJobListeners(kind, repoId, {
      onComplete: (variant, bytes) =>
        handlersRef.current.onComplete?.(variant, bytes),
      onCancelled: (variant) => handlersRef.current.onCancelled?.(variant),
      onError: (variant) => handlersRef.current.onError?.(variant),
    });
  }, [kind, repoId]);

  useEffect(() => {
    if (!autoAdopt) return;
    const controller = new AbortController();
    void downloadManager.probeAndAdopt(kind, repoId, controller.signal);
    return () => controller.abort();
  }, [autoAdopt, kind, repoId]);

  const active = useDownloadManagerStore(
    useShallow((state) => selectActiveJob(state, kind, repoId)),
  );
  const transportConflict = useDownloadManagerStore(
    (state) => state.conflicts[repoKeyOf(kind, repoId)]?.info ?? null,
  );

  const startDownload = useCallback(
    (
      variant: string | null,
      expectedBytes: number,
      useXetOverride?: boolean,
    ) => {
      void downloadManager.start(
        { kind, repoId, variant, expectedBytes },
        useXetOverride === undefined ? {} : { useXet: useXetOverride },
      );
    },
    [kind, repoId],
  );

  const requestStartDownload = useCallback(
    (variant: string | null, expectedBytes: number) => {
      void downloadManager.requestStart({
        kind,
        repoId,
        variant,
        expectedBytes,
      });
    },
    [kind, repoId],
  );

  const cancelDownload = useCallback(
    (variant: string | null) => {
      void downloadManager.cancel(jobKeyOf(kind, repoId, variant));
    },
    [kind, repoId],
  );

  const adoptRunningJob = useCallback(
    (variant: string | null, expectedBytes: number) => {
      downloadManager.adopt({ kind, repoId, variant, expectedBytes });
    },
    [kind, repoId],
  );

  const setExpectedBytes = useCallback(
    (bytes: number) => downloadManager.setExpected(kind, repoId, bytes),
    [kind, repoId],
  );

  const resumeConflict = useCallback(
    () => downloadManager.resumeConflict(repoKeyOf(kind, repoId)),
    [kind, repoId],
  );
  const restartConflict = useCallback(
    () => downloadManager.restartConflict(repoKeyOf(kind, repoId)),
    [kind, repoId],
  );
  const cancelConflict = useCallback(
    () => downloadManager.cancelConflict(repoKeyOf(kind, repoId)),
    [kind, repoId],
  );

  const progress = useMemo<DownloadJobProgress | null>(
    () =>
      active
        ? {
            variant: active.variant,
            expectedBytes: active.expectedBytes,
            downloadedBytes: active.downloadedBytes,
            fraction: active.fraction,
          }
        : null,
    [active],
  );

  return {
    progress,
    bytesPerSec: active?.bytesPerSec ?? 0,
    cancelling: active?.state === "cancelling",
    transportConflict,
    startDownload,
    requestStartDownload,
    cancelDownload,
    adoptRunningJob,
    setExpectedBytes,
    resumeConflict,
    restartConflict,
    cancelConflict,
  };
}
