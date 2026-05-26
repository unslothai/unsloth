// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useRepoDownload } from "@/features/download-jobs";
import { deleteCachedModel } from "@/features/inventory";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
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
import { formatBytes } from "@/lib/format";
import { fingerprintToken } from "@/lib/token-fingerprint";
import { useOnlineStatus } from "@/hooks";
import { notifyInventoryEntryDeleted } from "../delete-notifications";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";
import type { InventoryHint } from "./download-types";
import { PerModelConfigNotice } from "./per-model-config-notice";
import type { ModelInventoryFormat } from "@/features/inventory";

function formatModelLabel(modelFormat?: ModelInventoryFormat | null): string {
  if (modelFormat === "adapter") return "Adapter";
  if (modelFormat === "checkpoint") return "Checkpoint";
  if (!modelFormat || modelFormat === "safetensors") return "Safetensors";
  return "Model";
}

function formatModelTone(
  modelFormat?: ModelInventoryFormat | null,
): "checkpoint" | "adapter" {
  return modelFormat === "adapter" ? "adapter" : "checkpoint";
}

export function SafetensorsDownloadCard({
  repoId,
  isDownloaded,
  isPartial = false,
  modelFormat,
  canRun = true,
  isActive,
  isLoadingThisModel,
  cachePath,
  knownBytes,
  onLoad,
  onUseInChat,
  onTrain,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  canRun?: boolean;
  isActive: boolean;
  isLoadingThisModel: boolean;
  cachePath?: string | null;
  knownBytes?: number | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onTrain?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const sizeKey = `${repoId}::${fingerprintToken(hfToken)}`;
  const [modelSize, setModelSize] = useState<{
    key: string;
    bytes: number | null;
  }>(() => ({ key: sizeKey, bytes: null }));
  const modelTotalBytes =
    knownBytes && knownBytes > 0
      ? knownBytes
      : modelSize.key === sizeKey
        ? modelSize.bytes
        : null;
  const [deleteRepoOpen, setDeleteRepoOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const job = useRepoDownload({
    kind: "model",
    repoId,
    autoAdopt: true,
  });

  const progress = job.progress;
  const cancelling = job.cancelling;
  const setJobExpectedBytes = job.setExpectedBytes;

  useEffect(() => {
    if (knownBytes && knownBytes > 0) {
      setJobExpectedBytes(knownBytes);
      return;
    }
    if (!online) return;
    if (!repoId) return;
    const controller = new AbortController();
    const { signal } = controller;
    void fetchModelSize(repoId, hfToken || undefined, signal)
      .then((info) => {
        if (signal.aborted || !info) return;
        const upstream = info.weightsBytes ?? info.totalBytes;
        if (upstream && upstream > 0) {
          setModelSize({ key: sizeKey, bytes: upstream });
          setJobExpectedBytes(upstream);
        }
      })
      .catch((err) => {
        if (!signal.aborted && import.meta.env.DEV) {
          console.debug("Model size lookup failed", err);
        }
      });
    return () => {
      controller.abort();
    };
  }, [repoId, hfToken, sizeKey, setJobExpectedBytes, knownBytes, online]);

  async function handleRepoDeleteConfirm() {
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, undefined, hfToken || undefined);
      notifyInventoryEntryDeleted({ kind: "model", id: repoId });
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
  const showActionPair = isDownloaded && !downloading && (canRun || !!onTrain);
  const showUnavailableAction =
    isDownloaded && !downloading && !canRun && !onTrain;
  const canDelete =
    isDownloaded && !downloading && !isActive && !isLoadingThisModel;
  const progressPercent =
    progress != null ? Math.round(Math.min(progress.fraction, 1) * 100) : null;

  return (
    <div className="flex w-full flex-col gap-2">
      <PerModelConfigNotice modelId={repoId} />
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
                <span className="font-medium text-foreground">{repoId}</span>{" "}
                and its downloaded files
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
              <DotTag
                tone="success"
                label={isActive ? "Loaded" : "On device"}
              />
            )}
            {!isDownloaded && !isActive && isPartial && !downloading && (
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
            <DotTag
              tone={formatModelTone(modelFormat)}
              label={formatModelLabel(modelFormat)}
            />
            {modelTotalBytes && modelTotalBytes > 0 && (
              <span className="tabular-nums">
                {formatBytes(modelTotalBytes)}
              </span>
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
            {onTrain && (
              <button
                type="button"
                onClick={onTrain}
                className="hub-action-btn w-24"
              >
                <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
                Train
              </button>
            )}
            <button
              type="button"
              disabled={isLoadingThisModel || !canRun}
              onClick={() => {
                if (!canRun) return;
                if (isActive) {
                  onUseInChat?.();
                  return;
                }
                onLoad({});
              }}
              className={cn(
                isLoadingThisModel || isActive || !canRun
                  ? "hub-action-btn w-24"
                  : "run-action-btn w-24",
                (isLoadingThisModel || !canRun) && "opacity-70",
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
              ) : !canRun ? (
                <>
                  <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
                  No run
                </>
              ) : (
                <>
                  <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                  Run
                </>
              )}
            </button>
          </div>
        ) : showUnavailableAction ? (
          <button
            type="button"
            disabled
            className="hub-action-btn w-28 opacity-70"
          >
            <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
            Unavailable
          </button>
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
    </div>
  );
}
