


import type { DownloadJob } from "../download-manager";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DownloadStopMode } from "./download-cancel-indicator";

export function partialResumeLabel(transport: string | null | undefined): string {
  if (transport === "xet") return "Redownload";
  if (transport === "http") return "Continue";
  return "Retry";
}

/** Stopping an HTTP download leaves a partial to continue from, so it is a
 * pause. Xet has to start over, so it is a cancel. Unknown assumes the costlier
 * one rather than promising a resume that may not exist.
 *
 * Reads the running job's transport, not the partial's: a fresh HTTP download
 * has no partial yet, and a restarted conflict switches transport, so the
 * partial describes neither. */
export function downloadStopMode(
  activeTransport: string | null | undefined,
  partialTransport?: string | null,
  cancelTransport?: string | null,
): DownloadStopMode {
  // The cancel marker wins where there is one: a Xet run that fell back to
  // HTTP still cancels into a restart-only partial, so Pause would promise a
  // resume the marker does not allow.
  const transport = cancelTransport ?? activeTransport ?? partialTransport;
  return transport === "http" ? "pause" : "cancel";
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
): string {
  return isPartial ? partialResumeLabel(partialTransport) : "Download";
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
}: {
  job: DownloadJob;
  variant: string | null;
  expectedBytes: number;
  downloading: boolean;
  cancelling?: boolean;
  disabled: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
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
  );
  return {
    downloading,
    cancelling,
    starting,
    isPartial,
    partialTransport,
    progressPercent,
    stopMode,
    disabled: effectiveDisabled,
    ariaLabel: downloadActionAriaLabel(downloading, cancelling, stopMode),
    downloadLabel: downloadActionLabel(isPartial, partialTransport),
    onClick,
  };
}
