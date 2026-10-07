// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatBytes, formatEta, formatRate } from "@/features/hub/lib/format";
import type { DownloadPart, DownloadPartKind } from "./download-breakdown";
import { isIndeterminateProgress } from "./progress-reconcile";

const PART_STYLE: Record<
  DownloadPartKind,
  { label: string; fill: string; track: string; text: string }
> = {
  model: {
    label: "model",
    fill: "bg-amber-500",
    track: "bg-amber-500/25",
    text: "text-amber-600 dark:text-amber-400",
  },
  encoder: {
    label: "text encoder",
    fill: "bg-blue-500",
    track: "bg-blue-500/25",
    text: "text-blue-600 dark:text-blue-400",
  },
  vae: {
    label: "VAE",
    fill: "bg-emerald-500",
    track: "bg-emerald-500/25",
    text: "text-emerald-600 dark:text-emerald-400",
  },
  other: {
    label: "other",
    fill: "bg-muted-foreground/70",
    track: "bg-muted-foreground/20",
    text: "text-muted-foreground",
  },
};

/** The bar split by what each part of the download is, the cached model included. */
function PartsBar({ parts }: { parts: DownloadPart[] }) {
  const total = parts.reduce((n, p) => n + p.bytes, 0);
  const widths = parts.map((p) => Math.max(2, (p.bytes / total) * 100));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex h-[5px] gap-0.5">
        {parts.map((part, i) => (
          <div
            key={part.kind}
            className={`relative overflow-hidden rounded-full ${PART_STYLE[part.kind].track}`}
            style={{ width: `${widths[i]}%` }}
          >
            <div
              className={`h-full ${PART_STYLE[part.kind].fill} transition-[width] duration-500 ease-linear`}
              style={{
                width: `${part.bytes > 0 ? (part.doneBytes / part.bytes) * 100 : 0}%`,
              }}
            />
          </div>
        ))}
      </div>
      <div className="flex gap-0.5 text-ui-10p5">
        {parts.map((part, i) => (
          <div
            key={part.kind}
            className={`whitespace-nowrap ${PART_STYLE[part.kind].text} ${i === parts.length - 1 && i > 0 ? "text-right" : ""}`}
            style={{ width: `${widths[i]}%` }}
          >
            {PART_STYLE[part.kind].label}
            {widths[i] >= 30 ? ` ${formatBytes(part.bytes)}` : ""}
          </div>
        ))}
      </div>
    </div>
  );
}

export interface DownloadProgress {
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

export function DownloadProgressBar({
  progress,
  bytesPerSec,
  cancelling = false,
  etaSeconds = 0,
  activity,
  parts,
}: {
  progress: DownloadProgress;
  bytesPerSec: number;
  cancelling?: boolean;
  /**
   * Seconds remaining, from the estimator that produced ``bytesPerSec``.
   * Deriving it here gave every caller its own ETA semantics and left the
   * shared estimator's ``etaSeconds`` unused on this path.
   */
  etaSeconds?: number;
  activity?: string;
  /** From `downloadParts`; null or absent keeps the single bar. */
  parts?: DownloadPart[] | null;
}) {
  const exactPercent = Math.min(Math.max(progress.fraction, 0), 1) * 100;
  const indeterminate = isIndeterminateProgress(progress, cancelling);
  const totalLabel =
    progress.expectedBytes > 0 ? formatBytes(progress.expectedBytes) : null;
  const rateLabel = formatRate(bytesPerSec);
  const etaLabel = etaSeconds > 0 ? formatEta(etaSeconds) : "";
  return (
    <div className="flex flex-col gap-1.5 pb-1">
      {parts && !indeterminate ? (
        <PartsBar parts={parts} />
      ) : (
        <div className="relative h-[3px] overflow-hidden rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
          {indeterminate ? (
            <div className="loading-bar-slide h-full w-1/3 rounded-full bg-status-warning/80" />
          ) : (
            <>
              <div
                className="h-full rounded-full bg-status-warning/80 transition-[width] duration-500 ease-linear"
                style={{ width: `${exactPercent}%` }}
              />
              <span
                aria-hidden="true"
                className="pointer-events-none absolute top-1/2 size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-status-warning ring-2 ring-status-warning/30 transition-[left] duration-500 ease-linear"
                style={{ left: `${exactPercent}%` }}
              />
            </>
          )}
        </div>
      )}
      <div className="flex items-center justify-between gap-2 text-ui-10p5 text-muted-foreground tabular-nums">
        <span>
          {indeterminate
            ? (activity ?? "Transferring…")
            : formatBytes(progress.downloadedBytes)}
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
