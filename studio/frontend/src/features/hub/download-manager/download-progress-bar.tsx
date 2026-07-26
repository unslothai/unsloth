// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatBytes, formatEta, formatRate } from "@/features/hub/lib/format";

export interface DownloadProgress {
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

export function DownloadProgressBar({
  progress,
  bytesPerSec,
}: {
  progress: DownloadProgress;
  bytesPerSec: number;
}) {
  const exactPercent = Math.min(Math.max(progress.fraction, 0), 1) * 100;
  const totalLabel =
    progress.expectedBytes > 0 ? formatBytes(progress.expectedBytes) : null;
  const rateLabel = formatRate(bytesPerSec);
  const remainingBytes = Math.max(
    progress.expectedBytes - progress.downloadedBytes,
    0,
  );
  const etaLabel =
    bytesPerSec > 0 && remainingBytes > 0
      ? formatEta(remainingBytes / bytesPerSec)
      : "";
  return (
    <div className="flex flex-col gap-1.5 pb-1">
      <div className="relative h-[3px] overflow-hidden rounded-full bg-foreground/[0.06] dark:bg-white/[0.06]">
        <div
          className="h-full rounded-full bg-status-warning/80 transition-[width] duration-500 ease-linear"
          style={{ width: `${exactPercent}%` }}
        />
        <span
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-status-warning ring-2 ring-status-warning/30 transition-[left] duration-500 ease-linear"
          style={{ left: `${exactPercent}%` }}
        />
      </div>
      <div className="flex items-center justify-between gap-2 text-ui-10p5 text-muted-foreground tabular-nums">
        <span>
          {formatBytes(progress.downloadedBytes)}
          {totalLabel && ` / ${totalLabel}`}
        </span>
        <span className="flex items-center gap-2">
          {rateLabel && <span>{rateLabel}</span>}
          {etaLabel && <span>{etaLabel}</span>}
        </span>
      </div>
    </div>
  );
}
