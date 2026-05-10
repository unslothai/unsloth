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
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { fetchDatasetSize } from "../lib/dataset-size";
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

    void fetchDatasetSize(repoId).then((info) => {
      if (cancelled || !info) return;
      const upstream = info.numBytesParquet ?? info.numBytesOriginal;
      if (upstream && upstream > 0) {
        setTotalBytes((prev) => prev ?? upstream);
      }
    });

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
    <div className="download-card">
      <div className="flex items-center">
        <div className="flex h-9 min-w-0 flex-1 items-center pl-3">
          <span className="flex items-center gap-2.5 text-[12px] text-muted-foreground">
            <span className="font-medium text-amber-600 dark:text-amber-400">
              Dataset
            </span>
            {totalBytes && totalBytes > 0 && (
              <span className="tabular-nums">{formatBytes(totalBytes)}</span>
            )}
            {isDownloaded && (
              <span className="inline-flex items-center gap-1 font-medium text-emerald-600 dark:text-emerald-400">
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  strokeWidth={2.5}
                  className="size-3"
                />
                On device
              </span>
            )}
          </span>
        </div>
        <div
          aria-hidden="true"
          className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] dark:bg-white/[0.04]"
        />
        <button
          type="button"
          disabled={downloading || isDownloaded}
          onClick={() => void startDownload()}
          className={cn(
            "inline-flex h-9 w-24 shrink-0 cursor-pointer items-center justify-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12.5px] font-medium tracking-tight transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]",
            isDownloaded
              ? "cursor-default text-emerald-700 dark:text-emerald-400"
              : "text-foreground",
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
                icon={Download01Icon}
                strokeWidth={1.75}
                className="size-4"
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
