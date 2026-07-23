// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useRepoDownload } from "../download-manager";
import { deleteCachedDataset } from "../inventory";
import { cn } from "@/lib/utils";
import { TrainIcon } from "../components/train-icon";
import { HUB_POST_DOWNLOAD_ACTIONS_VISIBLE } from "../lib/hub-feature-flags";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { useHfTokenStore } from "../stores/hf-token-store";
import { formatBytes } from "../lib/format";
import { useDatasetSize } from "../hooks/use-dataset-size";
import {
  CardDivider,
  CardDeleteButton,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";
import { useCardDelete } from "./use-card-delete";
import { useDownloadCardState } from "./use-download-card-state";

export function DatasetDownloadSection({
  repoId,
  isDownloaded,
  isPartial = false,
  partialTransport = null,
  cachePath,
  knownBytes,
  onTrain,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  cachePath?: string | null;
  knownBytes?: number | null;
  onTrain?: () => void;
  onChange?: () => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const { deleting, runDelete } = useCardDelete({
    action: () => deleteCachedDataset(repoId),
    resourceName: "dataset",
    successMessage: () => `Deleted ${repoId}`,
    onSuccess: () => {
      setDeleteOpen(false);
      onChange?.();
    },
  });

  const job = useRepoDownload({
    kind: "dataset",
    repoId,
    autoAdopt: true,
  });

  const progress = job.progress;
  const cancelling = job.cancelling;
  const upstreamSize = useDatasetSize(repoId, {
    enabled:
      progress === null && !isDownloaded && !(knownBytes && knownBytes > 0),
    token: hfToken || undefined,
  });
  const upstreamBytes =
    upstreamSize?.numBytesParquet ?? upstreamSize?.numBytesOriginal ?? null;
  const progressBytes =
    progress && progress.expectedBytes > 0 ? progress.expectedBytes : null;
  const totalBytes =
    progressBytes && progressBytes > 0
      ? progressBytes
      : knownBytes && knownBytes > 0
        ? knownBytes
        : upstreamBytes;

  const downloading = progress !== null;
  const canDelete =
    (isDownloaded || isPartial) && !downloading && !cancelling && !deleting;
  const downloadAction = useDownloadCardState({
    job,
    variant: null,
    // The datasets-server size above is a parquet/original estimate, not the raw
    // repo bytes snapshot_download fetches; 0 lets the backend resolve the true total.
    expectedBytes: 0,
    downloading,
    disabled: cancelling || deleting,
    isPartial,
    partialTransport,
  });

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
          onConfirm={() => void runDelete()}
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
        <span className="flex items-center gap-1.5 text-[0.75rem] text-muted-foreground">
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
            <CardDeleteButton
              label={`Delete ${repoId}`}
              onClick={() => setDeleteOpen(true)}
            />
          )}
          {isDownloaded && cachePath && (
            <PathInfoButton path={cachePath} />
          )}
        </div>
      </div>
      {/* Train CTA hidden until Hub→train picker ships; divider pairs with it. */}
      {(!isDownloaded || downloading || HUB_POST_DOWNLOAD_ACTIONS_VISIBLE) && (
        <CardDivider />
      )}
      {isDownloaded && !downloading ? (
        <button
          type="button"
          onClick={() => onTrain?.()}
          className={cn(
            "hub-action-btn w-28",
            !HUB_POST_DOWNLOAD_ACTIONS_VISIBLE && "hidden",
          )}
        >
          <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
          Train
        </button>
      ) : (
        <DownloadActionButton
          downloading={downloadAction.downloading}
          cancelling={downloadAction.cancelling}
          loading={downloadAction.starting}
          isPartial={downloadAction.isPartial}
          partialTransport={downloadAction.partialTransport}
          progressPercent={downloadAction.progressPercent}
          disabled={downloadAction.disabled}
          onClick={downloadAction.onClick}
        />
      )}
    </DownloadCard>
  );
}
