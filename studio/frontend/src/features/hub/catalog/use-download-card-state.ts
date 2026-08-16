// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadJob } from "../download-manager";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DownloadStopMode } from "./download-cancel-indicator";

/** Whether a partial written by `transport` can be picked up byte for byte.
 *
 * Two conditions, and both are needed. Xet rewrites the destination from
 * scratch, so only an HTTP partial is a candidate; and only a huggingface_hub
 * that still reopens `.incomplete` (<= 1.17) can append to it, which is what
 * `partialsResumable` reports. */
export function partialIsResumable(
  transport: string | null | undefined,
  partialsResumable: boolean,
): boolean {
  return transport === "http" && partialsResumable;
}

/** What the button on a partial row does. Never "Redownload": whichever
 * transport wrote the partial, files already on disk are kept and only the
 * interrupted one is fetched again. */
export function partialResumeLabel(
  transport: string | null | undefined,
  partialsResumable = false,
): string {
  return partialIsResumable(transport, partialsResumable) ? "Resume" : "Continue";
}

/** Tooltip for a "Partial" badge. The badge is not a control, so it names the
 * button that is, and says what continuing actually costs. */
export function partialDownloadHint(
  transport: string | null | undefined,
  partialsResumable = false,
): string {
  const label = partialResumeLabel(transport, partialsResumable);
  return partialIsResumable(transport, partialsResumable)
    ? `Partial download. Click ${label} to pick up where it stopped.`
    : `Partial download. Click ${label} to finish it. Files already downloaded are kept; the interrupted one starts over.`;
}

/** Stopping a download that can be resumed is a pause; anything else is a
 * cancel, since the interrupted file has to start over.
 *
 * Reads the running job's transport, not the partial's: a fresh HTTP download
 * has no partial yet, and a restarted conflict switches transport, so the
 * partial describes neither. */
export function downloadStopMode(
  activeTransport: string | null | undefined,
  partialTransport?: string | null,
  cancelTransport?: string | null,
  partialsResumable = false,
): DownloadStopMode {
  // The cancel marker wins where there is one: a Xet run that fell back to
  // HTTP still cancels into a restart-only partial, so Pause would promise a
  // resume the marker does not allow.
  const transport = cancelTransport ?? activeTransport ?? partialTransport;
  return partialIsResumable(transport, partialsResumable) ? "pause" : "cancel";
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
  partialTransport: string | null | undefined,
  partialsResumable = false,
): string {
  return isPartial
    ? partialResumeLabel(partialTransport, partialsResumable)
    : "Download";
}

export function useDownloadCardState({
  job,
  variant,
  expectedBytes,
  downloading,
  cancelling = job.cancelling,
  disabled,
  isPartial = false,
  partialTransport = null,
  partialsResumable = false,
}: {
  job: DownloadJob;
  variant: string | null;
  expectedBytes: number;
  downloading: boolean;
  cancelling?: boolean;
  disabled: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  /** Backend capability: see partialIsResumable. */
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
    void job.requestStartDownload(variant, expectedBytes).finally(() => {
      if (mountedRef.current) setStarting(false);
    });
  }, [
    cancelling,
    disabled,
    downloading,
    expectedBytes,
    job,
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
    partialsResumable,
    progressPercent,
    stopMode,
    disabled: effectiveDisabled,
    ariaLabel: downloadActionAriaLabel(downloading, cancelling, stopMode),
    downloadLabel: downloadActionLabel(
      isPartial,
      partialTransport,
      partialsResumable,
    ),
    partialHint: partialDownloadHint(partialTransport, partialsResumable),
    onClick,
  };
}
