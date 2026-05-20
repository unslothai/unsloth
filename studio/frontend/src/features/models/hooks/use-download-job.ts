// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import type {
  DownloadJobStatus,
  TransportStatus,
} from "@/features/chat/api/chat-api";
import { formatBytes } from "../lib/format";
import { useTransportMode } from "../lib/transport-preference";
import type { TransportConflictInfo } from "../components/transport-conflict-dialog";

export interface DownloadJobProgress {
  variant: string | null;
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

interface ProgressResponse {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
}

export interface DownloadJobAdapter {
  repoId: string;
  label: (variant: string | null) => string;
  toastId: (variant: string | null) => string;
  startErrorTitle: string;
  errorTitle: string;
  getProgress: (
    variant: string | null,
    expectedBytes: number,
  ) => Promise<ProgressResponse>;
  getStatus: (variant: string | null) => Promise<DownloadJobStatus>;
  start: (variant: string | null, useXet: boolean) => Promise<unknown>;
  cancel: (variant: string | null) => Promise<unknown>;
  getTransportStatus: () => Promise<TransportStatus>;
  onComplete?: (variant: string | null, bytes: number) => void | Promise<unknown>;
  onCancelled?: (variant: string | null) => void | Promise<unknown>;
  onError?: (variant: string | null) => void | Promise<unknown>;
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
  ) => Promise<void>;
  requestStartDownload: (
    variant: string | null,
    expectedBytes: number,
  ) => Promise<void>;
  cancelDownload: (variant: string | null) => Promise<void>;
  adoptRunningJob: (variant: string | null, expectedBytes: number) => void;
  resumeConflict: () => void;
  restartConflict: () => void;
  cancelConflict: () => void;
}

// Polls fail transiently (a dropped request, a backend hiccup). Only give up
// after this many in a row so a persistent 5xx surfaces a terminal error
// instead of leaving the progress bar frozen forever.
const MAX_CONSECUTIVE_POLL_FAILURES = 6;

export function useDownloadJob(adapter: DownloadJobAdapter): DownloadJob {
  const { repoId } = adapter;
  const [transportMode] = useTransportMode();
  const adapterRef = useRef(adapter);
  const transportModeRef = useRef(transportMode);
  useEffect(() => {
    adapterRef.current = adapter;
    transportModeRef.current = transportMode;
  });

  const [progress, setProgress] = useState<DownloadJobProgress | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [bytesPerSec, setBytesPerSec] = useState(0);
  const [transportConflict, setTransportConflict] =
    useState<TransportConflictInfo | null>(null);

  const pollRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const startEpochRef = useRef(0);
  const speedSampleRef = useRef<{ bytes: number; tMs: number } | null>(null);
  const activeToastIdRef = useRef<string | null>(null);
  const pendingRef = useRef<{ variant: string | null; expectedBytes: number } | null>(
    null,
  );

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      // Invalidate any in-flight tick so it can't fire setState/onComplete
      // after unmount once its awaited request resolves.
      startEpochRef.current++;
      stopPolling();
      if (activeToastIdRef.current) {
        toast.dismiss(activeToastIdRef.current);
        activeToastIdRef.current = null;
      }
    };
  }, [stopPolling]);

  useEffect(() => {
    startEpochRef.current++;
    setProgress(null);
    setCancelling(false);
    setBytesPerSec(0);
    speedSampleRef.current = null;
    inFlightRef.current = false;
    stopPolling();
  }, [repoId, stopPolling]);

  const startDownload = useCallback(
    async (
      variant: string | null,
      expectedBytes: number,
      useXetOverride?: boolean,
      opts?: { adopt?: boolean },
    ) => {
      const ad = adapterRef.current;
      const jobRepoId = ad.repoId;
      const epoch = ++startEpochRef.current;
      stopPolling();
      inFlightRef.current = false;
      setProgress({
        variant,
        expectedBytes,
        downloadedBytes: 0,
        fraction: 0,
      });
      setBytesPerSec(0);
      speedSampleRef.current = null;

      const label = ad.label(variant);
      const toastId = ad.toastId(variant);
      activeToastIdRef.current = toastId;

      if (!opts?.adopt) {
        toast(`Downloading ${label}`, {
          id: toastId,
          description: "Starting download…",
          duration: 10_000,
        });

        try {
          await ad.start(
            variant,
            useXetOverride ?? transportModeRef.current === "xet",
          );
        } catch (err) {
          if (startEpochRef.current !== epoch) return;
          setProgress(null);
          toast.error(ad.startErrorTitle, {
            id: toastId,
            description: err instanceof Error ? err.message : undefined,
            duration: 4000,
          });
          return;
        }
      }

      let consecutiveFailures = 0;
      const tick = async () => {
        const a = adapterRef.current;
        if (a.repoId !== jobRepoId || startEpochRef.current !== epoch) return;
        if (inFlightRef.current) return;
        inFlightRef.current = true;
        const teardown = () => {
          stopPolling();
          setProgress(null);
          setBytesPerSec(0);
          speedSampleRef.current = null;
          activeToastIdRef.current = null;
        };
        try {
          const [progressResp, status] = await Promise.all([
            a.getProgress(variant, expectedBytes),
            a.getStatus(variant),
          ]);
          if (
            adapterRef.current.repoId !== jobRepoId ||
            startEpochRef.current !== epoch
          )
            return;
          consecutiveFailures = 0;

          const reported = progressResp.expected_bytes;
          const nextExpected = reported > 0 ? reported : expectedBytes;
          const derivedFraction =
            progressResp.progress > 0
              ? progressResp.progress
              : nextExpected > 0
                ? Math.min(progressResp.downloaded_bytes / nextExpected, 0.99)
                : 0;
          setProgress({
            variant,
            expectedBytes: nextExpected,
            downloadedBytes: progressResp.downloaded_bytes,
            fraction: derivedFraction,
          });

          const nowMs = Date.now();
          const lastSample = speedSampleRef.current;
          if (lastSample) {
            const dt = (nowMs - lastSample.tMs) / 1000;
            const db = progressResp.downloaded_bytes - lastSample.bytes;
            if (dt > 0 && db >= 0) {
              const sample = db / dt;
              setBytesPerSec((prev) =>
                prev > 0 ? prev * 0.7 + sample * 0.3 : sample,
              );
            }
          }
          speedSampleRef.current = {
            bytes: progressResp.downloaded_bytes,
            tMs: nowMs,
          };

          const percent = Math.round(Math.min(derivedFraction, 1) * 100);
          const downloadedLabel = formatBytes(progressResp.downloaded_bytes);
          const totalLabel = nextExpected > 0 ? formatBytes(nextExpected) : null;
          toast(`Downloading ${label}`, {
            id: toastId,
            description: totalLabel
              ? `${downloadedLabel} / ${totalLabel} · ${percent}%`
              : `${downloadedLabel} · ${percent}%`,
            duration: 10_000,
          });

          const settle = (result: void | Promise<unknown>) =>
            void Promise.resolve(result).finally(() => setCancelling(false));

          if (status.state === "complete") {
            teardown();
            toast.success(`Downloaded ${label}`, {
              id: toastId,
              description: undefined,
              duration: 3000,
            });
            settle(
              a.onComplete?.(
                variant,
                progressResp.downloaded_bytes || nextExpected || 0,
              ),
            );
          } else if (status.state === "cancelled") {
            teardown();
            toast(`Cancelled download of ${label}`, {
              id: toastId,
              description: "Partial files kept. Click Resume to continue.",
              duration: 3000,
            });
            settle(a.onCancelled?.(variant));
          } else if (status.state === "error") {
            teardown();
            toast.error(a.errorTitle, {
              id: toastId,
              description: status.error ?? undefined,
              duration: 5000,
            });
            settle(a.onError?.(variant));
          }
        } catch {
          consecutiveFailures++;
          if (consecutiveFailures >= MAX_CONSECUTIVE_POLL_FAILURES) {
            teardown();
            toast.error(a.errorTitle, {
              id: toastId,
              description:
                "Lost contact with the download. Check your connection and try again.",
              duration: 5000,
            });
            void Promise.resolve(a.onError?.(variant)).finally(() =>
              setCancelling(false),
            );
          }
        } finally {
          inFlightRef.current = false;
        }
      };

      void tick();
      pollRef.current = window.setInterval(() => {
        void tick();
      }, 500);
    },
    [stopPolling],
  );

  const adoptRunningJob = useCallback(
    (variant: string | null, expectedBytes: number) => {
      if (pollRef.current != null) return;
      void startDownload(variant, expectedBytes, undefined, { adopt: true });
    },
    [startDownload],
  );

  const requestStartDownload = useCallback(
    async (variant: string | null, expectedBytes: number) => {
      const ad = adapterRef.current;
      const mode = transportModeRef.current;
      try {
        const status = await ad.getTransportStatus();
        if (
          status.has_partial &&
          status.last_transport &&
          status.last_transport !== mode
        ) {
          pendingRef.current = { variant, expectedBytes };
          setTransportConflict({
            previous: status.last_transport,
            next: mode,
            resumable: status.resumable,
          });
          return;
        }
      } catch {
        void 0;
      }
      void startDownload(variant, expectedBytes);
    },
    [startDownload],
  );

  const cancelDownload = useCallback(async (variant: string | null) => {
    const ad = adapterRef.current;
    setCancelling(true);
    try {
      await ad.cancel(variant);
    } catch (err) {
      setCancelling(false);
      toast.error("Failed to cancel download", {
        description: err instanceof Error ? err.message : undefined,
        duration: 4000,
      });
    }
  }, []);

  const resumeConflict = useCallback(() => {
    const conflict = transportConflict;
    const pending = pendingRef.current;
    if (!conflict || !pending) return;
    pendingRef.current = null;
    setTransportConflict(null);
    void startDownload(
      pending.variant,
      pending.expectedBytes,
      conflict.previous === "xet",
    );
  }, [transportConflict, startDownload]);

  const restartConflict = useCallback(() => {
    const conflict = transportConflict;
    const pending = pendingRef.current;
    if (!conflict || !pending) return;
    pendingRef.current = null;
    setTransportConflict(null);
    void startDownload(
      pending.variant,
      pending.expectedBytes,
      conflict.next === "xet",
    );
  }, [transportConflict, startDownload]);

  const cancelConflict = useCallback(() => {
    pendingRef.current = null;
    setTransportConflict(null);
  }, []);

  return {
    progress,
    bytesPerSec,
    cancelling,
    transportConflict,
    startDownload,
    requestStartDownload,
    cancelDownload,
    adoptRunningJob,
    resumeConflict,
    restartConflict,
    cancelConflict,
  };
}
