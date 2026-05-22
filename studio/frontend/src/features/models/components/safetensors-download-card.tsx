// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { deleteCachedModel } from "@/features/chat/api/chat-api";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { TrainIcon } from "@/components/icons/train-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { fetchModelSize } from "../lib/dataset-size";
import { formatBytes } from "../lib/format";
import { useRepoDownload } from "../download-manager";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";
import type { InventoryHint } from "./download-types";

export function SafetensorsDownloadCard({
  repoId,
  isDownloaded,
  isPartial = false,
  isActive,
  isLoadingThisModel,
  cachePath,
  onLoad,
  onUseInChat,
  onTrain,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  isActive: boolean;
  isLoadingThisModel: boolean;
  cachePath?: string | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onTrain?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [modelTotalBytes, setModelTotalBytes] = useState<number | null>(null);
  const [deleteRepoOpen, setDeleteRepoOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    setModelTotalBytes(null);
    if (!repoId) return;
    let cancelled = false;
    void fetchModelSize(repoId, hfToken || undefined).then((info) => {
      if (cancelled || !info) return;
      const upstream = info.weightsBytes ?? info.totalBytes;
      if (upstream && upstream > 0) setModelTotalBytes(upstream);
    });
    return () => {
      cancelled = true;
    };
  }, [repoId, hfToken]);

  const job = useRepoDownload({
    kind: "model",
    repoId,
    autoAdopt: true,
    onComplete: (_variant, bytes) =>
      onChange?.({ kind: "model", repoId, bytes: bytes || undefined }),
    onCancelled: () => onChange?.(),
  });

  const progress = job.progress;
  const cancelling = job.cancelling;

  const setJobExpectedBytes = job.setExpectedBytes;
  useEffect(() => {
    if (modelTotalBytes && modelTotalBytes > 0) {
      setJobExpectedBytes(modelTotalBytes);
    }
  }, [modelTotalBytes, setJobExpectedBytes]);

  async function handleRepoDeleteConfirm() {
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, undefined, hfToken || undefined);
      toast.success(`Deleted ${repoId}`);
      setDeleteRepoOpen(false);
      onChange?.();
    } catch (err) {
      toast.error("Failed to delete model", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setDeleting(false);
    }
  }

  const downloading = progress !== null && progress.variant === null;
  const showActionPair = isDownloaded && !downloading;
  const canDelete =
    isDownloaded && !downloading && !isActive && !isLoadingThisModel;
  const progressPercent =
    progress != null ? Math.round(Math.min(progress.fraction, 1) * 100) : null;

  return (
    <DownloadCard
      job={job}
      progress={downloading ? progress : null}
      dialogs={
        <DeleteConfirmDialog
          open={deleteRepoOpen}
          onOpenChange={(o) => {
            if (!o && !deleting) setDeleteRepoOpen(false);
          }}
          title="Delete cached model?"
          deleting={deleting}
          onConfirm={() => void handleRepoDeleteConfirm()}
          description={
            <>
              This will remove{" "}
              <span className="font-medium text-foreground">{repoId}</span> and
              its downloaded files
              {modelTotalBytes && modelTotalBytes > 0
                ? ` (${formatBytes(modelTotalBytes)})`
                : ""}{" "}
              from disk. You can re-download it later.
            </>
          }
        />
      }
    >
      <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
        <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
          {(isActive || isDownloaded) && (
            <DotTag tone="success" label={isActive ? "Loaded" : "On device"} />
          )}
          {!isDownloaded && !isActive && isPartial && !downloading && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <DotTag tone="warning" label="Partial" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" sideOffset={4}>
                Partial download. Click Resume to continue.
              </TooltipContent>
            </Tooltip>
          )}
          <DotTag tone="checkpoint" label="Safetensors" />
          {modelTotalBytes && modelTotalBytes > 0 && (
            <span className="tabular-nums">{formatBytes(modelTotalBytes)}</span>
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
                    setDeleteRepoOpen(true);
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
      {showActionPair ? (
        <div className="group/pair flex h-9 shrink-0 items-stretch gap-1.5">
          <button
            type="button"
            onClick={() => onTrain?.()}
            className="hub-action-btn w-24"
          >
            <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
            Train
          </button>
          <button
            type="button"
            disabled={isLoadingThisModel}
            onClick={() => {
              if (isActive) {
                onUseInChat?.();
                return;
              }
              onLoad({});
            }}
            className={cn(
              isLoadingThisModel || isActive
                ? "hub-action-btn w-24"
                : "run-action-btn w-24",
              isLoadingThisModel && "opacity-70",
            )}
          >
            {isLoadingThisModel ? (
              <>
                <Spinner />
                Loading…
              </>
            ) : isActive ? (
              <>
                <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
                Chat
              </>
            ) : (
              <>
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                Run
              </>
            )}
          </button>
        </div>
      ) : (
        <DownloadActionButton
          downloading={downloading}
          cancelling={cancelling}
          loading={isLoadingThisModel}
          isPartial={isPartial}
          progressPercent={progressPercent}
          disabled={isLoadingThisModel || cancelling}
          onClick={() => {
            if (downloading) {
              void job.cancelDownload(null);
              return;
            }
            void job.requestStartDownload(null, modelTotalBytes ?? 0);
          }}
        />
      )}
    </DownloadCard>
  );
}
