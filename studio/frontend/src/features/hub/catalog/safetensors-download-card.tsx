// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEffect, useState } from "react";
import { useHttpPartialsResumable, useRepoDownload } from "../download-manager";
import { useOnlineStatus } from "../hooks/use-online-status";
import { deleteCachedModel } from "../inventory";
import type { ModelInventoryFormat } from "../inventory";
import { fetchModelSize } from "../lib/dataset-size";
import { formatBytes } from "../lib/format";
import { fingerprintToken } from "../lib/token-fingerprint";
import { useHfTokenStore } from "../stores/hf-token-store";
import { DotTag } from "./dot-tag";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
} from "./download-card";
import { QuantOptionsMenu } from "./gguf-download-card";
import { useCardDelete } from "./use-card-delete";
import { DeleteImpactSummary, useDeleteImpact } from "./delete-impact";
import { useDownloadCardState } from "./use-download-card-state";

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
  partialTransport = null,
  partialResumable = false,
  modelFormat,
  isActive,
  isLoadingThisModel,
  cachePath,
  knownBytes,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  isActive: boolean;
  isLoadingThisModel: boolean;
  /** Owning cache dir, threaded into delete so it targets this copy. */
  cachePath?: string | null;
  knownBytes?: number | null;
  onChange?: () => void;
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
  const { deleting, runDelete } = useCardDelete({
    action: () =>
      deleteCachedModel(repoId, undefined, hfToken || undefined, cachePath ?? undefined),
    resourceName: "model",
    successMessage: () => `Deleted ${repoId}`,
    onSuccess: () => {
      setDeleteRepoOpen(false);
      onChange?.();
    },
  });

  const job = useRepoDownload({
    kind: "model",
    repoId,
    activeVariant: null,
    autoAdopt: true,
  });

  const progress = job.progress;
  const cancelling = job.cancelling;
  const repoPeerActive = job.repoPeerActive;
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

  const downloading = progress !== null && progress.variant === null;
  const partialsResumable = useHttpPartialsResumable();
  const downloadAction = useDownloadCardState({
    job,
    variant: null,
    expectedBytes: modelTotalBytes ?? 0,
    downloading,
    disabled: isLoadingThisModel || cancelling || repoPeerActive,
    isPartial,
    partialTransport,
    partialResumable,
    partialsResumable,
  });
  const showDownloadAction =
    !isDownloaded || downloading || cancelling || downloadAction.starting;
  const canDelete =
    (isDownloaded || isPartial) &&
    !downloading &&
    !repoPeerActive &&
    !isActive &&
    !isLoadingThisModel;

  // Same preview the On Device and picker rows run: without it this card kept an enabled Delete
  // for a companion base an installed image GGUF still needs, and the refusal arrived as a 400
  // after the user confirmed.
  const deleteImpact = useDeleteImpact(
    deleteRepoOpen && Boolean(repoId),
    repoId ?? "",
  );

  return (
    <div className="flex w-full flex-col gap-2">
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
            blocked={(deleteImpact?.blocked_by.length ?? 0) > 0}
            onConfirm={() => void runDelete()}
            description={
              <>
                This will remove{" "}
                <span className="font-medium text-foreground">{repoId}</span>{" "}
                and its downloaded files
                {modelTotalBytes && modelTotalBytes > 0
                  ? ` (${formatBytes(modelTotalBytes)})`
                  : ""}{" "}
                from disk. You can re-download it later.
                <DeleteImpactSummary impact={deleteImpact} />
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
                <TooltipTrigger asChild={true}>
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
            {/* Same 3-dots menu as GGUF, at repo level (no quant); pinning is
                omitted here. Managed HF-cache repos only. */}
            {(isDownloaded || (isPartial && !downloading)) &&
              !/^([/\\~.]|[A-Za-z]:)/.test(repoId) && (
              <QuantOptionsMenu
                repoId={repoId}
                label={repoId}
                downloaded={isDownloaded}
                canDelete={canDelete}
                onDelete={() => setDeleteRepoOpen(true)}
                showPin={false}
                buttonClassName="ml-0.5 size-7"
                iconClassName="size-4"
              />
            )}
          </div>
        </div>
        {showDownloadAction && <CardDivider />}
        {showDownloadAction && (
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
            className={repoPeerActive ? "opacity-70" : undefined}
          />
        )}
      </DownloadCard>
    </div>
  );
}
