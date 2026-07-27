// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadJob } from "../download-manager";
import { useCallback, useEffect, useRef, useState } from "react";

export function partialResumeLabel(transport: string | null | undefined): string {
  if (transport === "xet") return "Redownload";
  if (transport === "http") return "Continue";
  return "Retry";
}

export function downloadActionAriaLabel(
  downloading: boolean,
  cancelling: boolean,
): string | undefined {
  return cancelling ? "Cancelling…" : downloading ? "Cancel download" : undefined;
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
  return {
    downloading,
    cancelling,
    starting,
    isPartial,
    partialTransport,
    progressPercent,
    disabled: effectiveDisabled,
    ariaLabel: downloadActionAriaLabel(downloading, cancelling),
    downloadLabel: downloadActionLabel(isPartial, partialTransport),
    onClick,
  };
}
