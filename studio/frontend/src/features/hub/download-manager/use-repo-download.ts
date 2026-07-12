// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useLatestRef } from "../hooks/use-latest-ref";
import type { TransportConflictInfo } from "./types";
import {
  type DownloadKind,
  type JobListeners,
  downloadManager,
  jobKeyOf,
  selectActiveJob,
  subscribeJobListeners,
  useDownloadManagerStore,
} from "./download-manager-controller";

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
  repoPeerActive: boolean;
  transportConflict: TransportConflictInfo | null;
  requestStartDownload: (
    variant: string | null,
    expectedBytes: number,
  ) => Promise<void>;
  cancelDownload: (variant: string | null) => void;
  setExpectedBytes: (bytes: number, variant?: string | null) => void;
  resumeConflict: () => void;
  restartConflict: () => void;
  cancelConflict: () => void;
}

export interface RepoDownloadConfig {
  kind: DownloadKind;
  repoId: string;
  activeVariant?: string | null;
  onComplete?: JobListeners["onComplete"];
  onCancelled?: JobListeners["onCancelled"];
  onError?: JobListeners["onError"];
  // Attach to a no-variant backend download already running (GGUF surfaces adopt their own variant).
  autoAdopt?: boolean;
}

/**
 * Binds a single download surface (one repo, optionally per GGUF variant) to the
 * global download manager. Job state and polling live in the store, so a
 * download keeps running and stays visible after the card unmounts.
 */
export function useRepoDownload(config: RepoDownloadConfig): DownloadJob {
  const {
    kind,
    repoId,
    activeVariant,
    onComplete,
    onCancelled,
    onError,
    autoAdopt,
  } = config;

  const handlersRef = useLatestRef<JobListeners>({
    onComplete,
    onCancelled,
    onError,
  });
  useEffect(() => {
    return subscribeJobListeners(kind, repoId, {
      onComplete: (variant, bytes) =>
        handlersRef.current.onComplete?.(variant, bytes),
      onCancelled: (variant) => handlersRef.current.onCancelled?.(variant),
      onError: (variant) => handlersRef.current.onError?.(variant),
    });
  }, [handlersRef, kind, repoId]);

  useEffect(() => {
    if (!autoAdopt) return;
    const controller = new AbortController();
    void downloadManager.probeAndAdopt(kind, repoId, controller.signal);
    return () => controller.abort();
  }, [autoAdopt, kind, repoId]);

  const activeState = useDownloadManagerStore(
    useShallow((state) => {
      if (activeVariant === undefined) {
        return {
          active: selectActiveJob(state, kind, repoId),
          repoPeerActive: false,
        };
      }
      const active = selectActiveJob(state, kind, repoId, activeVariant);
      const repoActive = selectActiveJob(state, kind, repoId);
      return {
        active,
        repoPeerActive: Boolean(repoActive && repoActive.key !== active?.key),
      };
    }),
  );
  const active = activeState.active;
  const conflictKey = useMemo(
    () => jobKeyOf(kind, repoId, activeVariant ?? null),
    [activeVariant, kind, repoId],
  );
  const transportConflict = useDownloadManagerStore(
    (state) => state.conflicts[conflictKey]?.info ?? null,
  );

  useEffect(
    () => () => {
      downloadManager.cancelConflict(conflictKey);
    },
    [conflictKey],
  );

  const requestStartDownload = useCallback(
    async (variant: string | null, expectedBytes: number) => {
      // This surface renders the conflict resolver (transportConflict), so the
      // start outcome is handled by the card UI; the awaited result is ignored.
      await downloadManager.requestStart({
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
      const state = useDownloadManagerStore.getState();
      const activeJob = selectActiveJob(state, kind, repoId, variant);
      void downloadManager.cancel(
        activeJob?.key ?? jobKeyOf(kind, repoId, variant),
      );
    },
    [kind, repoId],
  );

  const setExpectedBytes = useCallback(
    (bytes: number, variant: string | null = null) =>
      downloadManager.setExpected(kind, repoId, variant, bytes),
    [kind, repoId],
  );

  const resumeConflict = useCallback(
    () => downloadManager.resumeConflict(conflictKey),
    [conflictKey],
  );
  const restartConflict = useCallback(
    () => downloadManager.restartConflict(conflictKey),
    [conflictKey],
  );
  const cancelConflict = useCallback(
    () => downloadManager.cancelConflict(conflictKey),
    [conflictKey],
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
    repoPeerActive: activeState.repoPeerActive,
    transportConflict,
    requestStartDownload,
    cancelDownload,
    setExpectedBytes,
    resumeConflict,
    restartConflict,
    cancelConflict,
  };
}
