// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  getDatasetDownloadProgress,
  getDatasetDownloadStatus,
  startDatasetDownload,
} from "@/features/chat/api/chat-api";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  DownloadCircle02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { formatBytes } from "../lib/format";

interface ProgressState {
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

export function DatasetDownloadSection({
  repoId,
  isDownloaded,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  onChange?: () => void;
}) {
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [totalBytes, setTotalBytes] = useState<number | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    setProgress(null);
    setTotalBytes(null);
    stopPolling();

    let cancelled = false;
    void getDatasetDownloadProgress(repoId)
      .then((res) => {
        if (cancelled) return;
        if (res.expected_bytes > 0) {
          setTotalBytes(res.expected_bytes);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [repoId, stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const startDownload = useCallback(async () => {
    stopPolling();
    setProgress({ expectedBytes: 0, downloadedBytes: 0, fraction: 0 });

    const toastId = `dataset-download-${repoId}`;
    toast(`Downloading ${repoId}`, {
      id: toastId,
      description: "Starting download…",
      duration: Number.POSITIVE_INFINITY,
    });

    try {
      await startDatasetDownload({ repo_id: repoId });
    } catch (err) {
      setProgress(null);
      toast.error("Failed to start dataset download", {
        id: toastId,
        description: err instanceof Error ? err.message : undefined,
        duration: 4000,
      });
      return;
    }

    const tick = async () => {
      try {
        const [progressResp, status] = await Promise.all([
          getDatasetDownloadProgress(repoId),
          getDatasetDownloadStatus(repoId),
        ]);

        const next: ProgressState = {
          expectedBytes: progressResp.expected_bytes,
          downloadedBytes: progressResp.downloaded_bytes,
          fraction: progressResp.progress,
        };
        setProgress(next);

        const percent = Math.round(Math.min(progressResp.progress, 1) * 100);
        const downloadedLabel = formatBytes(progressResp.downloaded_bytes);
        const totalLabel =
          progressResp.expected_bytes > 0
            ? formatBytes(progressResp.expected_bytes)
            : null;
        toast(`Downloading ${repoId}`, {
          id: toastId,
          description: totalLabel
            ? `${downloadedLabel} / ${totalLabel} · ${percent}%`
            : `${downloadedLabel} · ${percent}%`,
          duration: Number.POSITIVE_INFINITY,
        });

        if (status.state === "complete") {
          stopPolling();
          setProgress(null);
          toast.success(`Downloaded ${repoId}`, {
            id: toastId,
            description: undefined,
            duration: 3000,
          });
          onChange?.();
        } else if (status.state === "error") {
          stopPolling();
          setProgress(null);
          toast.error("Dataset download failed", {
            id: toastId,
            description: status.error ?? undefined,
            duration: 5000,
          });
        }
      } catch {
        // transient; next tick will retry
      }
    };

    void tick();
    pollRef.current = window.setInterval(() => {
      void tick();
    }, 750);
  }, [repoId, onChange, stopPolling]);

  const downloading = progress !== null;

  return (
    <div className="flex flex-col gap-3 rounded-[24px] border border-border/60 bg-background/80 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Download
            {totalBytes && totalBytes > 0 && (
              <span className="rounded-[5px] border border-border/60 px-1.5 py-px text-[10px] font-medium normal-case tracking-normal text-foreground/80 tabular-nums">
                {formatBytes(totalBytes)}
              </span>
            )}
          </span>
          <span className="text-[13px] text-foreground">
            {isDownloaded
              ? "Cached locally. Ready for fine-tuning runs."
              : downloading
                ? "Pulling dataset blobs into the HF cache."
                : "Pulls the full snapshot via huggingface_hub. Reuses the same cache the fine-tuning runner uses."}
          </span>
        </div>
        <button
          type="button"
          disabled={downloading || isDownloaded}
          onClick={() => void startDownload()}
          className={cn(
            "inline-flex h-10 shrink-0 items-center gap-1.5 rounded-[16px] px-4 text-[12.5px] font-medium transition-colors",
            isDownloaded
              ? "cursor-default bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
              : "bg-foreground text-background hover:bg-foreground/85",
            downloading && "opacity-70",
          )}
        >
          {downloading ? (
            <>
              <Spinner className="size-3.5" />
              Downloading…
            </>
          ) : isDownloaded ? (
            <>
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                strokeWidth={2}
                className="size-3.5"
              />
              On device
            </>
          ) : (
            <>
              <HugeiconsIcon
                icon={DownloadCircle02Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              Download
            </>
          )}
        </button>
      </div>
      {downloading && progress && (
        <div className="flex flex-col gap-1.5">
          <div className="h-1 overflow-hidden rounded-full bg-border/40">
            <div
              className="h-full rounded-full bg-foreground/80 transition-[width] duration-300"
              style={{
                width: `${Math.round(Math.min(progress.fraction, 1) * 100)}%`,
              }}
            />
          </div>
          <div className="flex items-center justify-between text-[10.5px] text-muted-foreground tabular-nums">
            <span>
              {formatBytes(progress.downloadedBytes)}
              {progress.expectedBytes > 0 &&
                ` / ${formatBytes(progress.expectedBytes)}`}
            </span>
            <span>
              {Math.round(Math.min(progress.fraction, 1) * 100)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
