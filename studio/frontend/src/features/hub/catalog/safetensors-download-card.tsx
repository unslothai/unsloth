// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  PlayIcon,
  RemoveCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { TrainIcon } from "../components/train-icon";
import { useRepoDownload } from "../download-manager";
import { useOnlineStatus } from "../hooks/use-online-status";
import { deleteCachedModel } from "../inventory";
import type { ModelInventoryFormat } from "../inventory";
import { fetchModelSize } from "../lib/dataset-size";
import { formatBytes } from "../lib/format";
import {
  HUB_NON_GGUF_RUN_ACTIONS_VISIBLE,
  HUB_POST_DOWNLOAD_ACTIONS_VISIBLE,
} from "../lib/hub-feature-flags";
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
import { SamplingSettingsButton } from "./sampling-settings-dialog";
import { useCardDelete } from "./use-card-delete";
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
  modelFormat,
  canRun = true,
  isActive,
  isLoadingThisModel,
  knownBytes,
  onLoad,
  onEject,
  onTrain,
  onChange,
}: {
  repoId: string;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  modelFormat?: ModelInventoryFormat | null;
  canRun?: boolean;
  isActive: boolean;
  isLoadingThisModel: boolean;
  /** Accepted for API parity; the options menu resolves the path itself. */
  cachePath?: string | null;
  knownBytes?: number | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  /** Accepted for API parity; the run bar ejects instead of opening chat. */
  onUseInChat?: () => void;
  onEject?: () => void;
  onTrain?: () => void;
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
    action: () => deleteCachedModel(repoId, undefined, hfToken || undefined),
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
  const downloadAction = useDownloadCardState({
    job,
    variant: null,
    expectedBytes: modelTotalBytes ?? 0,
    downloading,
    disabled: isLoadingThisModel || cancelling || repoPeerActive,
    isPartial,
    partialTransport,
  });
  const showActionPair = isDownloaded && !downloading && (canRun || !!onTrain);
  const showUnavailableAction =
    isDownloaded && !downloading && !canRun && !onTrain;
  const trainActionVisible = !!onTrain && HUB_POST_DOWNLOAD_ACTIONS_VISIBLE;
  const canDelete =
    (isDownloaded || isPartial) &&
    !downloading &&
    !repoPeerActive &&
    !isActive &&
    !isLoadingThisModel;

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
                <TooltipTrigger asChild={true}>
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
            {/* Gear before the 3-dots menu. */}
            <SamplingSettingsButton className="ml-0.5" />
            {/* Same 3-dots menu as GGUF, at repo level (no quant); pinning is
                omitted in the run bar. Managed HF-cache repos only. */}
            {isDownloaded && !/^([/\\~.]|[A-Za-z]:)/.test(repoId) && (
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
        {/* Info/actions hairline; dropped for the run action row (no divider before
            Run, as in the GGUF card's Run CTA), restored when the Train pair ships. */}
        {(!showActionPair || trainActionVisible) && <CardDivider />}
        {showActionPair ? (
          <div
            className={cn(
              "group/pair flex h-9 shrink-0 items-stretch gap-1.5",
              !HUB_NON_GGUF_RUN_ACTIONS_VISIBLE && "hidden",
            )}
          >
            {trainActionVisible && (
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
                  onEject?.();
                  return;
                }
                onLoad({});
              }}
              className={cn(
                isLoadingThisModel || isActive || !canRun
                  ? "hub-action-btn w-24"
                  : "hub-run-action-btn w-24",
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
                  <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} />
                  Eject
                </>
              ) : canRun ? (
                <>
                  <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                  Run
                </>
              ) : (
                <>
                  <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
                  No run
                </>
              )}
            </button>
          </div>
        ) : showUnavailableAction ? (
          <button
            type="button"
            disabled={true}
            className="hub-action-btn w-28 opacity-70"
          >
            <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
            Unavailable
          </button>
        ) : (
          <DownloadActionButton
            downloading={downloadAction.downloading}
            cancelling={downloadAction.cancelling}
            loading={isLoadingThisModel || downloadAction.starting}
            isPartial={downloadAction.isPartial}
            partialTransport={downloadAction.partialTransport}
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
