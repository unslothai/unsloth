// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useHttpPartialsResumable, useRepoDownload } from "../download-manager";
import { deleteCachedDataset } from "../inventory";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";
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
  partialResumable = false,
  cachePath,
  knownBytes,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  cachePath?: string | null;
  knownBytes?: number | null;
  onChange?: () => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const { deleting, runDelete } = useCardDelete({
    action: () => deleteCachedDataset(repoId, cachePath ?? undefined),
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
  const partialsResumable = useHttpPartialsResumable();
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
    partialResumable,
    partialsResumable,
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
        <span className="flex items-center gap-1.5 text-ui-12 text-muted-foreground">
          {isDownloaded && <DotTag tone="success" label="On device" />}
          {!isDownloaded && isPartial && !downloading && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <DotTag tone="warning" label="Partial" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" sideOffset={4}>
                {/* The badge is a status dot, not a control. */}
                {downloadAction.partialHint}
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
      {(!isDownloaded || downloading) && <CardDivider />}
      {(!isDownloaded || downloading) && (
        <DownloadActionButton
          downloading={downloadAction.downloading}
          cancelling={downloadAction.cancelling}
          loading={downloadAction.starting}
          isPartial={downloadAction.isPartial}
          partialResumable={downloadAction.partialResumable}
          stopMode={downloadAction.stopMode}
          progressPercent={downloadAction.progressPercent}
          disabled={downloadAction.disabled}
          onClick={downloadAction.onClick}
        />
      )}
    </DownloadCard>
  );
}
