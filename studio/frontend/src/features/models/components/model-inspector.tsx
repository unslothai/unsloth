// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn, formatCompact } from "@/lib/utils";
import {
  Calendar03Icon,
  Copy01Icon,
  CpuIcon,
  CubeIcon,
  Database02Icon,
  Download01Icon,
  FavouriteIcon,
  Globe02Icon,
  LayersLogoIcon,
  LicenseIcon,
  PackageIcon,
  RamMemoryIcon,
  Share05Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { useCopyFeedback } from "../hooks/use-copy-feedback";
import { DatasetDownloadSection } from "./dataset-download-section";
import { LocalDatasetCard } from "./local-dataset-card";
import { LocalOnDeviceCard } from "./local-on-device-card";
import { ModelReadme } from "./model-readme";
import { OwnerAvatar } from "./owner-avatar";
import { CapabilityPill } from "./shared";
import { DownloadSection, type InventoryHint } from "./download-section";
import {
  type DatasetSizeInfo,
  fetchDatasetSize,
} from "../lib/dataset-size";
import { formatBytes, formatRelativeShort } from "../lib/format";
import {
  formatLocalUpdated,
  formatPipelineTag,
  parseLanguageTags,
} from "../lib/view-models";
import type { SelectedModelView } from "../types";
import { type UnslothSupport, classifyUnslothSupport } from "@/hooks";
import { usePlatformStore } from "@/config/env";

function ViewRepositoryButton({
  repoId,
  isDataset,
}: {
  repoId: string;
  isDataset: boolean;
}) {
  const url = `https://huggingface.co/${isDataset ? "datasets/" : ""}${repoId}`;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="View repository"
          className="inline-flex size-7 shrink-0 items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <HugeiconsIcon
            icon={Share05Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </a>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        Open on Hugging Face
      </TooltipContent>
    </Tooltip>
  );
}

function CopyRepoButton({ repoId }: { repoId: string }) {
  const { copied, copy } = useCopyFeedback();

  const handleCopy = async (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    await copy(repoId);
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="Copy repository ID"
          onClick={handleCopy}
          className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {repoId}
      </TooltipContent>
    </Tooltip>
  );
}

function StatRow({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: IconSvgElement;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="tag-meta inline-flex cursor-default items-center gap-1.5 px-2.5 py-1 text-[11.5px] text-muted-foreground transition-colors hover:text-foreground/80">
          <HugeiconsIcon
            icon={icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
          />
          <span className="font-medium tabular-nums">{value}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="tooltip-compact">{label}</TooltipContent>
    </Tooltip>
  );
}

function StatGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">{children}</div>
  );
}

function StatusChip({
  tone,
  label,
  className,
}: {
  tone: "success" | "warning" | "danger";
  label: string;
  className?: string;
}) {
  const toneClass =
    tone === "danger"
      ? "border-red-500/35 text-red-700 dark:border-red-400/40 dark:text-red-300"
      : tone === "warning"
        ? "border-amber-500/40 text-amber-700 dark:border-amber-400/45 dark:text-amber-300"
        : "border-emerald-500/40 text-emerald-700 dark:border-emerald-400/45 dark:text-emerald-300";
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center whitespace-nowrap rounded-[7px] border bg-transparent px-1.5 text-[11px] font-medium leading-none",
        toneClass,
        className,
      )}
    >
      {label}
    </span>
  );
}

type VramInfo = { est: number; status: "fits" | "tight" | "exceeds" } | null;

function ModelStatusChips({
  isDataset,
  isGguf,
  unslothSupport,
  vramInfo,
}: {
  isDataset: boolean;
  isGguf: boolean;
  unslothSupport: UnslothSupport;
  vramInfo: VramInfo;
}) {
  const showUnsupported = !isDataset && unslothSupport.status === "unsupported";
  const showVram = !isDataset && vramInfo && !isGguf;
  if (!showUnsupported && !showVram) return null;

  const vramTone = !vramInfo
    ? "success"
    : vramInfo.status === "exceeds"
      ? "danger"
      : vramInfo.status === "tight"
        ? "warning"
        : "success";
  const vramLabel = !vramInfo
    ? ""
    : vramInfo.status === "exceeds"
      ? `Over VRAM budget · ~${vramInfo.est} GB`
      : vramInfo.status === "tight"
        ? `Tight fit · ~${vramInfo.est} GB`
        : `Likely fits · ~${vramInfo.est} GB`;
  const vramDetail = !vramInfo
    ? ""
    : vramInfo.status === "exceeds"
      ? "A 4-bit load is likely to exceed the current GPU budget. Higher-precision loads need even more."
      : vramInfo.status === "tight"
        ? "A 4-bit load should fit, with limited headroom for context and activations."
        : "A 4-bit load should fit comfortably on the current GPU.";

  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      {showUnsupported && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span tabIndex={0} className="inline-flex outline-none">
              <StatusChip tone="danger" label="May not be supported" />
            </span>
          </TooltipTrigger>
          <TooltipContent
            side="bottom"
            sideOffset={6}
            className="tooltip-compact max-w-xs"
          >
            This model may not be supported yet.
            {unslothSupport.reason && (
              <span className="mt-1 block text-[10.5px] font-normal text-white/75">
                {unslothSupport.reason}
              </span>
            )}
            <span className="mt-1 block text-[10.5px] font-normal text-white/75">
              Still downloadable to your Hugging Face cache, shared with every
              framework that reads it.
            </span>
          </TooltipContent>
        </Tooltip>
      )}
      {showVram && vramInfo && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span tabIndex={0} className="inline-flex outline-none">
              <StatusChip tone={vramTone} label={vramLabel} />
            </span>
          </TooltipTrigger>
          <TooltipContent
            side="bottom"
            sideOffset={6}
            className="tooltip-compact max-w-xs"
          >
            Estimated 4-bit memory load is around {vramInfo.est} GB.
            <span className="mt-1 block text-[10.5px] font-normal text-white/75">
              {vramDetail}
            </span>
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

export function ModelInspector({
  model,
  isActive,
  activeGgufVariant,
  isLoadingThisModel,
  loadingPhase,
  minMemory,
  vramInfo,
  gpuGb,
  systemRamGb,
  onLoad,
  onLoadLocal,
  onUseInChat,
  onTrain,
  onInventoryChange,
  isDataset = false,
  metadataUnavailable = false,
}: {
  model: SelectedModelView | null;
  isDataset?: boolean;
  metadataUnavailable?: boolean;
  isActive: boolean;
  activeGgufVariant: string | null;
  isLoadingThisModel: boolean;
  loadingPhase?: "downloading" | "starting";
  minMemory: string | null;
  vramInfo: {
    est: number;
    status: "fits" | "tight" | "exceeds";
  } | null;
  gpuGb?: number;
  systemRamGb?: number;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onLoadLocal: () => void;
  onUseInChat: () => void;
  onTrain?: () => void;
  onInventoryChange?: (hint?: InventoryHint) => void;
}) {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const datasetRepoId =
    isDataset && model?.hubRepoId ? model.hubRepoId : null;
  const [datasetSize, setDatasetSize] = useState<DatasetSizeInfo | null>(null);
  useEffect(() => {
    if (!datasetRepoId) {
      setDatasetSize(null);
      return;
    }
    let cancelled = false;
    void fetchDatasetSize(datasetRepoId).then((res) => {
      if (cancelled) return;
      setDatasetSize(res);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetRepoId]);

  const descScrollRef = useRef<HTMLDivElement | null>(null);
  const [descScrolled, setDescScrolled] = useState(false);
  useEffect(() => {
    const el = descScrollRef.current;
    if (!el) return;
    const onScroll = () => setDescScrolled(el.scrollTop > 0);
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [model?.id]);

  if (!model) {
    return (
      <div className="flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-3 px-6 py-10 text-center">
        <div className="inline-flex size-12 items-center justify-center rounded-[14px] bg-muted text-muted-foreground">
          <HugeiconsIcon icon={CubeIcon} strokeWidth={1.5} className="size-5" />
        </div>
        <div className="space-y-1">
          <p className="text-[15px] font-semibold tracking-tight text-foreground">
            Select a model
          </p>
          <p className="max-w-sm text-[12.5px] leading-5 text-muted-foreground">
            Choose an item from the catalog to inspect its runtime fit, download
            state, and model card.
          </p>
        </div>
      </div>
    );
  }

  const updatedRaw = model.updatedAt
    ? formatRelativeShort(model.updatedAt)
    : formatLocalUpdated(model.localUpdatedAt);
  const updatedLabel = updatedRaw === "Unknown update" ? "N/A" : updatedRaw;
  const taskLabel = formatPipelineTag(model.pipelineTag) ?? "General";
  const licenseLabel = model.license ?? "N/A";
  const paramsLabel = model.totalParams
    ? formatCompact(model.totalParams)
    : "N/A";
  const unslothSupport = classifyUnslothSupport({
    modelId: model.hubRepoId ?? model.id,
    pipelineTag: model.pipelineTag,
    tags: model.tags,
    libraryName: model.libraryName,
    deviceType,
    quantMethod: model.quantMethod,
  });

  // Mirror the training picker's filter so a model the Hub flags as
  // unsupported cannot be sent to Studio with a config that later fails.
  const trainingSupported = unslothSupport.status !== "unsupported";

  const languages = parseLanguageTags(model.tags);
  const datasetSizeBytes =
    datasetSize?.numBytesParquet ?? datasetSize?.numBytesOriginal ?? null;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      <div className="shrink-0 px-6 pb-2 pt-0">
        <div className="flex items-center gap-3.5">
          <OwnerAvatar
            owner={model.owner}
            repoName={model.title}
            className="size-[52px] rounded-[14px] text-[16px]"
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <h2 className="truncate text-[22px] font-semibold leading-[28px] tracking-[-0.025em] text-foreground">
                {model.title}
              </h2>
              {model.hubRepoId && (
                <div className="flex shrink-0 items-center gap-0.5">
                  <CopyRepoButton repoId={model.hubRepoId} />
                  <ViewRepositoryButton
                    repoId={model.hubRepoId}
                    isDataset={isDataset}
                  />
                </div>
              )}
            </div>
            <div className="mt-0.5 flex min-w-0 items-center gap-1 text-[14px] leading-[22px] text-muted-foreground">
              <span className="truncate">{model.owner}</span>
              {model.owner.toLowerCase() === "unsloth" && (
                <span
                  aria-label="Verified Unsloth"
                  className="verified-badge size-4 shrink-0 text-primary"
                />
              )}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-1.5">
          {isDataset && (
            <span className="inline-flex shrink-0 items-center rounded-full border border-violet-500/40 bg-transparent px-2 py-0.5 text-[11.5px] font-medium text-violet-600 dark:text-violet-400">
              Dataset
            </span>
          )}
          {!isDataset && (
            <span className="inline-flex h-6 items-center gap-1.5 rounded-full bg-muted px-2.5 text-[11.5px] font-medium text-foreground dark:bg-[rgba(255,255,255,0.04)]">
              <HugeiconsIcon
                icon={CubeIcon}
                strokeWidth={1.75}
                className="size-3 text-muted-foreground"
              />
              {taskLabel}
            </span>
          )}
          {model.capabilities.map((capability) => (
            <CapabilityPill key={capability.key} capability={capability} />
          ))}
        </div>

      </div>

      {isDataset && model.hubRepoId && !model.isLocal && (
        <div className="shrink-0 px-6 pt-3">
          <DatasetDownloadSection
            repoId={model.hubRepoId}
            isDownloaded={model.isDownloaded}
            isPartial={model.isPartial ?? false}
            cachePath={model.path}
            onTrain={onTrain}
            onChange={onInventoryChange}
          />
        </div>
      )}
      {isDataset && model.isLocal && model.path && (
        <div className="shrink-0 px-6 pt-3">
          <LocalDatasetCard
            sourceLabel={model.sourceLabel}
            source={model.localSource ?? "custom"}
            path={model.path}
            onTrain={onTrain}
          />
        </div>
      )}
      {!isDataset && (
        <div className="shrink-0 px-6 pt-3">
          {model.isLocal ? (
            <LocalOnDeviceCard
              repoId={model.hubRepoId}
              sourceLabel={model.sourceLabel}
              source={model.localSource ?? "custom"}
              path={model.path ?? model.displayId}
              isGguf={model.isGguf}
              isActive={isActive}
              isLoading={isLoadingThisModel}
              loadingPhase={loadingPhase}
              unsupportedReason={
                unslothSupport.status === "unsupported"
                  ? (unslothSupport.reason ?? "Unsupported format")
                  : null
              }
              onLoad={onLoadLocal}
              onUseInChat={onUseInChat}
              onTrain={!model.isGguf && trainingSupported ? onTrain : undefined}
              onChange={onInventoryChange}
            />
          ) : (
            <DownloadSection
              repoId={model.id}
              isGguf={model.isGguf}
              isDownloaded={model.isDownloaded}
              isPartial={model.isPartial ?? false}
              isActive={isActive}
              activeQuant={isActive ? (activeGgufVariant ?? null) : null}
              isLoadingThisModel={isLoadingThisModel}
              gpuGb={gpuGb}
              systemRamGb={systemRamGb}
              cachePath={model.path}
              onLoad={onLoad}
              onUseInChat={onUseInChat}
              onTrain={trainingSupported ? onTrain : undefined}
              onChange={onInventoryChange}
            />
          )}
        </div>
      )}

      <div className="shrink-0 px-6 pb-5 pt-5">
        {metadataUnavailable && (
          <p className="mb-3 text-[11.5px] leading-snug text-muted-foreground">
            Couldn't load full details from Hugging Face. Some fields may be
            incomplete.
          </p>
        )}
        <StatGrid>
          <StatRow
            label="Updated"
            value={updatedLabel}
            icon={Calendar03Icon}
          />
          <StatRow
            label="Downloads"
            value={
              model.downloads !== undefined
                ? formatCompact(model.downloads)
                : "N/A"
            }
            icon={Download01Icon}
          />
          <StatRow
            label="Likes"
            value={
              model.likes !== undefined ? formatCompact(model.likes) : "N/A"
            }
            icon={FavouriteIcon}
          />
          {!isDataset && (
            <StatRow
              label="Parameters"
              value={paramsLabel}
              icon={CpuIcon}
            />
          )}
          {!isDataset && (
            <StatRow
              label="Memory"
              value={minMemory ?? "N/A"}
              icon={RamMemoryIcon}
            />
          )}
          {isDataset && datasetSize?.numRows != null && (
            <StatRow
              label="Rows"
              value={formatCompact(datasetSize.numRows)}
              icon={Database02Icon}
            />
          )}
          {isDataset && datasetSizeBytes != null && (
            <StatRow
              label="Size"
              value={formatBytes(datasetSizeBytes)}
              icon={PackageIcon}
            />
          )}
          {isDataset && datasetSize?.numSplits != null && datasetSize.numSplits > 0 && (
            <StatRow
              label="Splits"
              value={String(datasetSize.numSplits)}
              icon={LayersLogoIcon}
            />
          )}
          {isDataset && languages.length > 0 && (
            <StatRow
              label="Languages"
              value={
                languages.length > 3
                  ? `${languages.slice(0, 3).join(", ")} +${languages.length - 3}`
                  : languages.join(", ")
              }
              icon={Globe02Icon}
            />
          )}
          <StatRow
            label="License"
            value={licenseLabel}
            icon={LicenseIcon}
          />
        </StatGrid>
        <ModelStatusChips
          isDataset={isDataset}
          isGguf={model.isGguf}
          unslothSupport={unslothSupport}
          vramInfo={vramInfo}
        />
      </div>

      <div
        aria-hidden="true"
        className={cn(
          "mx-6 shrink-0 border-t transition-colors",
          descScrolled ? "border-border" : "border-transparent",
        )}
      />

      <div
        ref={descScrollRef}
        className="min-h-0 flex-1 space-y-4 overflow-y-auto pl-6 pr-4 pt-5 pb-0 [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
        {model.hubRepoId && (
          <ModelReadme
            repoId={model.hubRepoId}
            kind={isDataset ? "dataset" : "model"}
          />
        )}
      </div>

      <div aria-hidden="true" className="h-3 shrink-0" />
    </div>
  );
}
