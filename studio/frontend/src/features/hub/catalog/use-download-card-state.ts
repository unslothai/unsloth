


import type { DownloadJob } from "../download-manager";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DownloadStopMode } from "./download-cancel-indicator";

/** What the button on a partial row does. Never "Redownload": whichever
 * transport wrote the partial, files already on disk are kept and only the
 * interrupted one is fetched again.
 *
 * `partialResumable` is the backend's verdict on THIS partial, not on the
 * installed writer: a cache shared with a newer environment holds partials that
 * even a resuming huggingface_hub will not reopen. */
export function partialResumeLabel(partialResumable = false): string {
  return partialResumable ? "Resume" : "Continue";
}

/** Tooltip for a "Partial" badge. The badge is not a control, so it names the
 * button that is, and says what continuing actually costs.
 *
 * The restart leads, because the unit is the FILE: a sharded repo keeps the
 * shards it finished, but a one-file quant has nothing to keep and fetches
 * every byte again. Leading with what survives reads as a promise the
 * single-file case cannot honour. */
export function partialDownloadHint(partialResumable = false): string {
  const label = partialResumeLabel(partialResumable);
  return partialResumable
    ? `Partial download. Click ${label} to pick up where it stopped.`
    : `Partial download. Click ${label} to finish it. The interrupted file starts over; other files already on disk are kept.`;
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
  // Capability, not a row verdict: the partial being written right now is this
  // machine's own, so the installed writer decides whether stopping keeps it.
  return transport === "http" && partialsResumable ? "pause" : "cancel";
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
  downloading: boolean;
  cancelling?: boolean;
  disabled: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  /** This row's partial can be continued byte for byte (backend verdict). */
  partialResumable?: boolean;
  /** Whether the installed writer resumes at all, for the running job's stop control. */
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
