// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { getDatasetDownloadProgress } from "@/features/chat";
import { deleteCachedDataset } from "@/features/training";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { TrainIcon } from "@/components/icons/train-icon";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { fetchDatasetSize } from "../lib/dataset-size";
import { formatBytes } from "@/lib/format";
import { useRepoDownload } from "../download-manager";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";
import type { InventoryHint } from "./download-types";

export function DatasetDownloadSection({
  repoId,
  isDownloaded,
  isPartial = false,
  cachePath,
  onTrain,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  cachePath?: string | null;
  onTrain?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const sizeKey = `${repoId}::${hfToken ?? ""}`;
  const [sizeState, setSizeState] = useState<{
    key: string;
    bytes: number | null;
  }>(() => ({ key: sizeKey, bytes: null }));
  const totalBytes = sizeState.key === sizeKey ? sizeState.bytes : null;
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const job = useRepoDownload({
    kind: "dataset",
    repoId,
    autoAdopt: true,
    onComplete: (_variant, bytes) =>
      onChange?.({ kind: "dataset", repoId, bytes: bytes || undefined }),
    onCancelled: () => onChange?.(),
  });

  const progress = job.progress;
  const cancelling = job.cancelling;

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    void getDatasetDownloadProgress(repoId, signal, hfToken || undefined)
      .then((res) => {
        if (signal.aborted) return;
        if (res.expected_bytes > 0) {
          setSizeState({ key: sizeKey, bytes: res.expected_bytes });
        }
      })
      .catch(() => {});

    void fetchDatasetSize(repoId).then((info) => {
      if (signal.aborted || !info) return;
      const upstream = info.numBytesParquet ?? info.numBytesOriginal;
      if (upstream && upstream > 0) {
        setSizeState((prev) =>
          prev.key === sizeKey && prev.bytes != null
            ? prev
            : { key: sizeKey, bytes: upstream },
        );
      }
    });

    return () => {
      controller.abort();
    };
  }, [repoId, hfToken, sizeKey]);

  const handleConfirmDelete = useCallback(async () => {
    setDeleting(true);
    try {
      await deleteCachedDataset(repoId);
      toast.success(`Deleted ${repoId}`);
      setDeleteOpen(false);
      onChange?.();
    } catch (err) {
      toast.error("Failed to delete dataset", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setDeleting(false);
    }
  }, [repoId, onChange]);

  const downloading = progress !== null;
  const canDelete =
    (isDownloaded || isPartial) && !downloading && !cancelling && !deleting;
  const progressPercent =
    progress != null ? Math.round(Math.min(progress.fraction, 1) * 100) : null;

  return (
    <DownloadCard
      job={job}
      progress={downloading ? progress : null}
      dialogs={
        <DeleteConfirmDialog
          open={deleteOpen}
          onOpenChange={(o) => {
            if (!o && !deleting) setDeleteOpen(false);
          }}
          title="Delete cached dataset?"
          deleting={deleting}
          onConfirm={() => void handleConfirmDelete()}
          description={
            <>
              This will remove{" "}
              <span className="font-medium text-foreground">{repoId}</span> and
              its downloaded files
              {totalBytes && totalBytes > 0
                ? ` (${formatBytes(totalBytes)})`
                : ""}{" "}
              from disk. You can re-download it later.
            </>
          }
        />
      }
    >
      <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
        <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
          {isDownloaded && <DotTag tone="success" label="On device" />}
          {!isDownloaded && isPartial && !downloading && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <DotTag tone="warning" label="Partial" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" sideOffset={4}>
                Partial download. Click to continue.
              </TooltipContent>
            </Tooltip>
          )}
          {totalBytes && totalBytes > 0 && (
            <span className="tabular-nums">{formatBytes(totalBytes)}</span>
          )}
        </span>
        <div className="ml-auto flex items-center gap-0.5">
          {canDelete && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label={`Delete ${repoId}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteOpen(true);
                  }}
                  className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-rose-500/10 hover:text-rose-600 focus-visible:opacity-100 group-hover/dl:opacity-100 dark:hover:bg-rose-500/15 dark:hover:text-rose-400"
                >
                  <HugeiconsIcon
                    icon={Delete02Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="top" sideOffset={4}>
                Delete from device
              </TooltipContent>
            </Tooltip>
          )}
          {isDownloaded && cachePath && (
            <PathInfoButton
              path={cachePath}
              title="On-device location"
              description={`Where ${repoId} lives on disk.`}
            />
          )}
        </div>
      </div>
      <CardDivider />
      {isDownloaded && !downloading ? (
        <button
          type="button"
          onClick={() => onTrain?.()}
          className="hub-action-btn w-28"
        >
          <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
          Train
        </button>
      ) : (
        <DownloadActionButton
          downloading={downloading}
          cancelling={cancelling}
          isPartial={isPartial}
          progressPercent={progressPercent}
          disabled={cancelling || deleting}
          onClick={() => {
            if (downloading) {
              void job.cancelDownload(null);
              return;
            }
            void job.requestStartDownload(null, totalBytes ?? 0);
          }}
        />
      )}
    </DownloadCard>
  );
}
