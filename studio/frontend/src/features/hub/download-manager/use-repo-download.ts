// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import { useLatestRef } from "../hooks/use-latest-ref";
import type { ResolvedTransport } from "./constants";
import type { TransportConflictInfo } from "./types";
import {
  conflictInfoForOwner,
  type DownloadKind,
  type JobListeners,
  downloadManager,
  jobKeyOf,
  repoKeyOf,
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
  etaSeconds: number;
  /** Transport the running job resolved to, when it started on this frontend. */
  transport: ResolvedTransport | null;
  /** Its cancel marker, when a Xet run fell back to HTTP: stopping it is still
   * a restart, so this and not `transport` decides the stop control. */
  cancelTransport: ResolvedTransport | null;
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
  const repoConflictKey = useMemo(
    () => repoKeyOf(kind, repoId),
    [kind, repoId],
  );
  const visibleConflict = useDownloadManagerStore(
    useShallow((state) => {
      const exact = state.conflicts[conflictKey];
      const exactInfo = conflictInfoForOwner(exact, "caller");
      if (exactInfo) return { key: conflictKey, info: exactInfo };
      const scoped = Object.entries(state.conflicts).find(
        ([key, entry]) =>
          key.startsWith(`${repoConflictKey}#`) && entry.owner === "caller",
      );
      return scoped
        ? { key: scoped[0], info: scoped[1].info }
        : { key: conflictKey, info: null };
    }),
  );
  const visibleConflictKey = visibleConflict.key;
  const transportConflict = visibleConflict.info;

  // Chat and Video staging park this hook on an idle repo id when the queue
  // clears. Keep that conflict for Hub, but remember its key so a later real
  // repo replacement clears the superseded request.
  const repoIdRef = useRef(repoId);
  const preservedConflictKeyRef = useRef<string | null>(null);
  repoIdRef.current = repoId;
  useEffect(
    () => () => {
      const parked = repoIdRef.current;
      if (
        parked === "__staged_download_idle__" ||
        parked === "__hub_autoload_idle__"
      ) {
        preservedConflictKeyRef.current = conflictKey;
        return;
      }
      if (preservedConflictKeyRef.current) {
        downloadManager.cancelConflict(preservedConflictKeyRef.current);
        preservedConflictKeyRef.current = null;
      }
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
    () => downloadManager.resumeConflict(visibleConflictKey),
    [visibleConflictKey],
  );
  const restartConflict = useCallback(
    () => downloadManager.restartConflict(visibleConflictKey),
    [visibleConflictKey],
  );
  const cancelConflict = useCallback(
    () => downloadManager.cancelConflict(visibleConflictKey),
    [visibleConflictKey],
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
    etaSeconds: active?.etaSeconds ?? 0,
    transport: active?.transport ?? null,
    cancelTransport: active?.cancelTransport ?? null,
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
