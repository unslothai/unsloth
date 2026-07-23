// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import type { DownloadJob, DownloadJobProgress } from "../download-manager";
import { DotTag } from "./dot-tag";
import {
  CardDivider,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";

export function GgufDownloadStatusCard({
  job,
  message,
  tone = "muted",
  loading = false,
  partial = false,
  actionLabel,
  onAction,
}: {
  job: DownloadJob;
  message: string;
  tone?: "muted" | "danger";
  loading?: boolean;
  partial?: boolean;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="flex w-full flex-col gap-2">
      <DownloadCard job={job} progress={null}>
        <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
          <span
            className={cn(
              "flex min-w-0 items-center gap-2 text-[0.78125rem]",
              tone === "danger" ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {loading && <Spinner className="size-3.5 shrink-0" />}
            {partial && <DotTag tone="warning" label="Partial" />}
            <span className="truncate">{message}</span>
          </span>
        </div>
        <CardDivider />
        {actionLabel && onAction ? (
          <button
            type="button"
            onClick={onAction}
            className="hub-action-btn w-28"
          >
            {actionLabel}
          </button>
        ) : (
          <button
            type="button"
            disabled
            className="hub-action-btn w-28 opacity-70"
          >
            {loading ? (
              <>
                <Spinner />
                Loading
              </>
            ) : (
              "Unavailable"
            )}
          </button>
        )}
      </DownloadCard>
    </div>
  );
}

/**
 * Shown when a download is in flight but the variant list isn't loaded. Keeps the
 * live job as source of truth so progress + cancel survive a remount instead of
 * being replaced by the variant-status card.
 */
export function GgufDownloadingFallbackCard({
  job,
  progress,
  cancelling,
}: {
  job: DownloadJob;
  progress: DownloadJobProgress;
  cancelling: boolean;
}) {
  return (
    <div className="flex w-full flex-col gap-2">
      <DownloadCard job={job} progress={progress}>
        <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
          <span className="flex min-w-0 items-center gap-2 text-[0.78125rem] text-muted-foreground">
            {progress.variant && <DotTag tone="gguf" label={progress.variant} />}
            <span className="truncate">Downloading…</span>
          </span>
        </div>
        <CardDivider />
        <DownloadActionButton
          downloading
          cancelling={cancelling}
          progressPercent={Math.round(Math.min(progress.fraction, 1) * 100)}
          disabled={cancelling}
          onClick={() => void job.cancelDownload(progress.variant)}
        />
      </DownloadCard>
    </div>
  );
}
