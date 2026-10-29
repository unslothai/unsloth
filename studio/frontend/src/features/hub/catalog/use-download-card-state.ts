// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadJob, DownloadPresentation } from "../download-manager";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DownloadStopMode } from "./download-cancel-indicator";

/** Never "Redownload": finished files are kept. `partialResumable` is the backend's verdict
 * on THIS partial, not on the installed writer. */
export function partialResumeLabel(partialResumable = false): string {
  return partialResumable ? "Resume" : "Continue";
}

/** Leads with the restart because the unit is the file: one-file quants keep nothing. */
export function partialDownloadHint(partialResumable = false): string {
  const label = partialResumeLabel(partialResumable);
  return partialResumable
    ? `Partial download. Click ${label} to pick up where it stopped.`
    : `Partial download. Click ${label} to finish it. The interrupted file starts over; other files already on disk are kept.`;
}

/** Uses the running job's transport, not the partial's, which may describe neither. */
export function downloadStopMode(
  activeTransport: string | null | undefined,
  partialTransport?: string | null,
  cancelTransport?: string | null,
  partialsResumable = false,
): DownloadStopMode {
  // The cancel marker wins: a Xet run that fell back to HTTP still leaves a restart-only partial.
  const transport = cancelTransport ?? activeTransport ?? partialTransport;
  // Capability, not row verdict: this machine's own writer decides whether stopping keeps it.
  return transport === "http" && partialsResumable ? "pause" : "cancel";
}

// Also match scoped ("@scope") jobs, or the row shows "Resume" mid-download.
export function isRepoDownloadProgress(
  progress: { variant: string | null } | null | undefined,
): boolean {
  if (!progress) return false;
  return progress.variant === null || progress.variant.startsWith("@");
}

export function downloadActionAriaLabel(
  downloading: boolean,
  cancelling: boolean,
  stopMode: DownloadStopMode = "cancel",
): string | undefined {
  if (cancelling) return "Cancelling…";
  if (!downloading) return undefined;
  return stopMode === "pause" ? "Pause download" : "Cancel download";
}

export function downloadActionLabel(
  isPartial: boolean,
  partialResumable = false,
): string {
  return isPartial ? partialResumeLabel(partialResumable) : "Download";
}

export function useDownloadCardState({
  job,
  variant,
  expectedBytes,
  presentation,
  downloading,
  cancelling = job.cancelling,
  disabled,
  isPartial = false,
  partialTransport = null,
  partialResumable = false,
  partialsResumable = false,
}: {
  job: DownloadJob;
  variant: string | null;
  expectedBytes: number;
  presentation?: DownloadPresentation;
  downloading: boolean;
  cancelling?: boolean;
  disabled: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  partialsResumable?: boolean;
}) {
  const [starting, setStarting] = useState(false);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  useEffect(() => {
    if (downloading || cancelling || disabled) {
      setStarting(false);
    }
  }, [cancelling, disabled, downloading]);
  const progressPercent =
    job.progress != null
      ? Math.round(Math.min(job.progress.fraction, 1) * 100)
      : null;
  const effectiveDisabled = disabled || starting;
  const onClick = useCallback(() => {
    if (disabled || cancelling || starting) return;
    if (downloading) {
      void job.cancelDownload(variant);
      return;
    }
    setStarting(true);
    void job
      .requestStartDownload(variant, expectedBytes, presentation)
      .finally(() => {
        if (mountedRef.current) setStarting(false);
      });
  }, [
    cancelling,
    disabled,
    downloading,
    expectedBytes,
    job,
    presentation,
    starting,
    variant,
  ]);
  const stopMode = downloadStopMode(
    job.transport,
    partialTransport,
    job.cancelTransport,
    partialsResumable,
  );
  return {
    downloading,
    cancelling,
    starting,
    isPartial,
    partialTransport,
    partialResumable,
    progressPercent,
    stopMode,
    disabled: effectiveDisabled,
    ariaLabel: downloadActionAriaLabel(downloading, cancelling, stopMode),
    downloadLabel: downloadActionLabel(isPartial, partialResumable),
    partialHint: partialDownloadHint(partialResumable),
    onClick,
  };
}
