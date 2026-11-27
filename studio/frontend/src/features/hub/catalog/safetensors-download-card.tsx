// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  confirmedSdxlCheckpoints,
  rememberedSdxlCheckpoints,
  rememberSdxlCheckpoints,
} from "../lib/sdxl-checkpoints";
import { imageCheckpointSelection, type ImageCheckpointFile } from "../download-manager/required-assets";
import { useHubDownloadPlan, useHubQueue } from "../download-manager/use-hub-download-queue";
import { HubPlanProgress } from "../download-manager/hub-plan-progress";
import { useRequiredAssetsDownload, type AssetRuntime } from "./use-required-assets-download";
import { useHfEndpoint } from "@/lib/hf-endpoint";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEffect, useMemo, useState } from "react";
import { useHttpPartialsResumable, useRepoDownload } from "../download-manager";
import { useOnlineStatus } from "../hooks/use-online-status";
import { deleteCachedModel } from "../inventory";
import type { ModelInventoryFormat } from "../inventory";
import {
  groupForRepoId,
  IMAGE_CATALOG,
} from "@/features/model-picker/components/model-selector/model-catalog";
import { bumpInventoryVersion } from "../stores/inventory-events";
import { fetchModelSize, type ModelSizeInfo } from "../lib/dataset-size";
import { formatBytes } from "../lib/format";
import { fingerprintToken } from "../lib/token-fingerprint";
import { useHfTokenStore } from "../stores/hf-token-store";
import { DotTag } from "./dot-tag";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadActionButton,
  DownloadCard,
  ModelRunActionButton,
} from "./download-card";
import { QuantOptionsMenu } from "./gguf-download-card";
import { useCardDelete } from "./use-card-delete";
import { DeleteImpactSummary, useDeleteImpact } from "./delete-impact";
import {
  isRepoDownloadProgress,
  useDownloadCardState,
} from "./use-download-card-state";

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
  companionPrefetch = false,
  modelFormat,
  isActive,
  isLoadingThisModel,
  cachePath,
  knownBytes,
  onRun,
  onRunCheckpoint,
  runPending = false,
  onChange,
  assetRuntime,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  companionPrefetch?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  isActive: boolean;
  isLoadingThisModel: boolean;
  cachePath?: string | null;
  knownBytes?: number | null;
  onRun?: () => void;
  onRunCheckpoint?: (filename: string) => void;
  runPending?: boolean;
  onChange?: () => void;
  assetRuntime?: AssetRuntime;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const discoverCheckpoints =
    assetRuntime === "images" &&
    !repoId.toLowerCase().startsWith("unsloth/") &&
    !groupForRepoId(repoId, IMAGE_CATALOG);
  const hfEndpoint = useHfEndpoint();
  const sizeKey = `${hfEndpoint}::${repoId}::${fingerprintToken(hfToken)}`;
  const [metadata, setMetadata] = useState<{ key: string; info: ModelSizeInfo | null; choices?: ImageCheckpointFile[] }>();
  const info = metadata?.key === sizeKey ? metadata.info : undefined;
  const remembered = useMemo(
    () => (discoverCheckpoints && !online ? rememberedSdxlCheckpoints(repoId) : undefined),
    [discoverCheckpoints, online, repoId],
  );
  const choices = !discoverCheckpoints
    ? undefined
    : metadata?.key === sizeKey
      ? metadata.choices
      : remembered;
  const job = useRepoDownload({
    kind: "model",
    repoId,
    activeVariant: null,
    autoAdopt: true,
    includeScopedJobs: true,
  });

  const progress = job.progress;
  const cancelling = job.cancelling;
  const setJobExpectedBytes = job.setExpectedBytes;

  const [picked, setPicked] = useState<{ key: string; filename: string }>();
  const queuedFilename = useHubQueue((state) =>
    state.plans.find((plan) => plan.origin?.repoId === repoId && plan.origin.filename)?.origin?.filename,
  );
  const { active, selected: checkpoint } = imageCheckpointSelection(
    repoId, choices, picked?.key === sizeKey ? picked.filename : undefined,
    queuedFilename, progress?.scopedFiles,
  );
  const activeFilename = active?.filename;
  // A reopened card keeps the transferred file selected when its queue entry disappears.
  useEffect(() => {
    if (activeFilename) setPicked({ key: sizeKey, filename: activeFilename });
  }, [sizeKey, activeFilename]);
  const discovering = discoverCheckpoints && online && info === undefined;
  const assets = useRequiredAssetsDownload({
    repoId,
    filename: checkpoint?.filename,
    runtime: discovering ? undefined : assetRuntime,
    modelLabel: checkpoint?.filename,
  });
  const installed = checkpoint ? assets.checkpointInstalled : isDownloaded;
  const partial = !checkpoint && isPartial;
  const run = checkpoint ? onRunCheckpoint?.bind(null, checkpoint.filename) : onRun;
  const modelTotalBytes =
    knownBytes && knownBytes > 0
      ? knownBytes
      : info?.weightsBytes ?? info?.totalBytes ?? null;
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

  useEffect(() => {
    if (knownBytes && knownBytes > 0) {
      setJobExpectedBytes(knownBytes);
      if (!discoverCheckpoints) return;
    }
    if (!online) return;
    if (!repoId) return;
    const controller = new AbortController();
    const { signal } = controller;
    void fetchModelSize(repoId, hfToken || undefined, signal)
      .then(async (info) => {
        if (signal.aborted) return;
        const choices = discoverCheckpoints && info
          ? await confirmedSdxlCheckpoints(repoId, info, hfToken || undefined, signal)
          : undefined;
        if (signal.aborted) return;
        if (choices?.length) rememberSdxlCheckpoints(repoId, choices);
        setMetadata({ key: sizeKey, info, choices });
        const upstream = info?.weightsBytes ?? info?.totalBytes;
        if (!knownBytes && upstream) setJobExpectedBytes(upstream);
      })
      .catch((err) => {
        if (signal.aborted) return;
        setMetadata({ key: sizeKey, info: null });
        if (import.meta.env.DEV) {
          console.debug("Model size lookup failed", err);
        }
      });
    return () => {
      controller.abort();
    };
  }, [repoId, hfToken, sizeKey, setJobExpectedBytes, knownBytes, online, discoverCheckpoints]);

  const repoDownloading = isRepoDownloadProgress(progress);
  const downloading = repoDownloading &&
    (!checkpoint || progress?.variant === null || progress?.scopedFiles?.includes(checkpoint.filename) === true);
  const repoPeerActive = job.repoPeerActive || (repoDownloading && !downloading);
  const partialsResumable = useHttpPartialsResumable();
  const downloadAction = useDownloadCardState({
    job,
    variant: downloading ? (progress?.variant ?? null) : null,
    expectedBytes: modelTotalBytes ?? 0,
    downloading,
    disabled: isLoadingThisModel || cancelling || repoPeerActive,
    isPartial,
    partialTransport,
    partialResumable,
    partialsResumable,
  });
  const hubPlan = useHubDownloadPlan(repoId, checkpoint?.filename);
  const displayBytes = checkpoint ? checkpoint.bytes : modelTotalBytes;
  const showDownloadAction = !hubPlan && (
    !installed || downloading || cancelling || downloadAction.starting);
  const showRunAction = !hubPlan &&
    Boolean(run) &&
    !discovering &&
    installed &&
    !partial &&
    !downloading &&
    !cancelling &&
    !downloadAction.starting &&
    !isLoadingThisModel;
  const hasCachedFiles = isDownloaded || isPartial || companionPrefetch;
  const canDelete =
    hasCachedFiles &&
    !downloading &&
    !repoPeerActive &&
    !isActive &&
    !isLoadingThisModel &&
    !runPending;

  // Without the preview, Delete stayed enabled for a base an installed image GGUF needs.
  const deleteImpact = useDeleteImpact(
    deleteRepoOpen && Boolean(repoId),
    repoId ?? "",
  );

  return (
    <div className="flex w-full flex-col gap-2">
      <DownloadCard
        footer={hubPlan ? <HubPlanProgress plan={hubPlan} /> : undefined}
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
          <span className="flex min-w-0 items-center gap-1.5 text-ui-12 text-muted-foreground">
            {checkpoint && (
              <select
                aria-label="Image checkpoint"
                className="min-w-0 truncate rounded bg-background text-foreground"
                value={checkpoint.filename}
                disabled={downloading || Boolean(hubPlan) || assets.checking}
                onChange={(event) => setPicked({ key: sizeKey, filename: event.target.value })}
              >
                {choices?.map((file) => (
                  <option key={file.filename}>{file.filename}</option>
                ))}
              </select>
            )}
            {installed && <DotTag tone="success" label="On device" />}
            {!installed && downloading && (
              <DotTag tone="downloading" label="Downloading" />
            )}
            {!installed && partial && !downloading && (
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <span className="inline-flex">
                    <DotTag tone="warning" label="Partial" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top" sideOffset={4}>
                  {downloadAction.partialHint}
                </TooltipContent>
              </Tooltip>
            )}
            {!checkpoint && (
              <DotTag tone={formatModelTone(modelFormat)} label={formatModelLabel(modelFormat)} />
            )}
            {displayBytes && displayBytes > 0 && (
              <span className="tabular-nums">
                {formatBytes(displayBytes)}
              </span>
            )}
          </span>
          <div className="ml-auto flex items-center gap-0.5">
            {/* Same 3-dots menu as GGUF, at repo level (no quant); pinning is
                omitted here. Managed HF-cache repos only. */}
            {(isDownloaded ||
              ((isPartial || companionPrefetch) && !downloading)) &&
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
        {showDownloadAction && !showRunAction && <CardDivider />}
        {showDownloadAction && (
          <DownloadActionButton
            downloading={downloadAction.downloading}
            cancelling={downloadAction.cancelling}
            loading={discovering || downloadAction.starting || assets.checking}
            isPartial={!checkpoint && downloadAction.isPartial}
            partialResumable={downloadAction.partialResumable}
            stopMode={downloadAction.stopMode}
            progressPercent={downloadAction.progressPercent}
            disabled={discovering || downloadAction.disabled || assets.checking}
            onClick={() => {
              if (downloading) downloadAction.onClick();
              else void assets.request(checkpoint ? bumpInventoryVersion : downloadAction.onClick);
            }}
            className={repoPeerActive ? "opacity-70" : undefined}
          />
        )}
        {hubPlan && <span className="px-3 text-ui-12 text-muted-foreground">Downloading…</span>}
        {showRunAction && run && (
          <ModelRunActionButton
            label={`Configure and run ${checkpoint?.filename ?? repoId}`}
            onClick={run}
            loading={runPending}
          />
        )}
      </DownloadCard>
      {assets.dialog}
    </div>
  );
}
