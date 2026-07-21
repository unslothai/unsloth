// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  type UnslothSupport,
  classifyUnslothSupport,
} from "@/features/hub/hooks/use-hub-model-search";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import {
  formatBytes,
  formatRelativeShort,
  formatShortDate,
} from "@/features/hub/lib/format";
import { cn, formatCompact } from "@/lib/utils";
import { confirmExternalLink } from "../stores/external-link-confirm";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import {
  Calendar03Icon,
  CalendarAdd01Icon,
  Copy01Icon,
  CpuIcon,
  CubeIcon,
  Database02Icon,
  Download01Icon,
  FavouriteIcon,
  Globe02Icon,
  LayersLogoIcon,
  LibraryIcon,
  LicenseIcon,
  PackageIcon,
  RamMemoryIcon,
  Share05Icon,
} from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useDeferredValue, useMemo } from "react";
import { useCopyFeedback } from "../hooks/use-copy-feedback";
import { useDatasetSize } from "../hooks/use-dataset-size";
import {
  formatLibrary,
  formatLocalUpdated,
  formatPipelineTag,
  parseLanguageTags,
} from "../lib/view-models";
import type { SelectedModelView } from "../types";
import { selectActiveJob, useDownloadManagerStore } from "../download-manager";
import { DatasetDownloadSection } from "./dataset-download-section";
import { DownloadSection } from "./download-section";
import { LocalDatasetCard } from "./local-dataset-card";
import { LocalOnDeviceCard } from "./local-on-device-card";
import { ModelReadme } from "./model-readme";
import { OwnerAvatar } from "./owner-avatar";
import { AccessChip, CapabilityPill } from "./shared";

// HF pipeline_tag values authoritative for embedding-only repos; capability
// labels (code/vision/audio) can leak onto them via name or tags.
const EMBEDDING_PIPELINE_TAGS: ReadonlySet<string> = new Set([
  "feature-extraction",
  "sentence-similarity",
]);

function ViewRepositoryButton({
  repoId,
  isDataset,
}: {
  repoId: string;
  isDataset: boolean;
}) {
  const online = useOnlineStatus();
  const url = `https://huggingface.co/${isDataset ? "datasets/" : ""}${repoId}`;
  const baseClass =
    "inline-flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors";
  const icon = (
    <HugeiconsIcon
      icon={Share05Icon}
      strokeWidth={1.75}
      className="size-[13px]"
    />
  );
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        {online ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View repository"
            className={cn(baseClass, "hover:bg-muted hover:text-foreground")}
            onClick={(event) => {
              event.stopPropagation();
              if (confirmExternalLink(url)) {
                event.preventDefault();
              }
            }}
          >
            {icon}
          </a>
        ) : (
          <span
            aria-label="View repository unavailable offline"
            aria-disabled="true"
            className={cn(baseClass, "cursor-not-allowed opacity-45")}
          >
            {icon}
          </span>
        )}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {online ? "Open on Hugging Face" : "Unavailable offline"}
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
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label="Copy repository ID"
          onClick={handleCopy}
          className="inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-[13px]"
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
  tooltip,
}: {
  label: string;
  value: string;
  icon: IconSvgElement;
  tooltip?: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <span className="hub-tag-meta inline-flex cursor-default items-center gap-1.5 px-2.5 py-1 text-[11.5px] text-muted-foreground transition-colors hover:text-foreground/80">
          <HugeiconsIcon
            icon={icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
          />
          <span className="font-medium tabular-nums">{value}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="tooltip-compact">
        {tooltip ?? label}
      </TooltipContent>
    </Tooltip>
  );
}

function StatGrid({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-wrap items-center gap-1.5">{children}</div>;
}

function InspectorDownloadSlot({ children }: { children: React.ReactNode }) {
  return <div className="max-w-[680px] pt-3">{children}</div>;
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
        "inline-flex h-5 shrink-0 items-center whitespace-nowrap rounded-full border bg-transparent px-2 text-[11px] font-medium leading-none",
        toneClass,
        className,
      )}
    >
      {label}
    </span>
  );
}

function BaseModelSearchChip({
  baseModel,
  searchTerm,
  onSearchHub,
}: {
  baseModel: string;
  searchTerm: string;
  onSearchHub?: (query: string) => void;
}) {
  const content = (
    <>
      <HugeiconsIcon
        icon={LayersLogoIcon}
        strokeWidth={1.75}
        className="size-3.5 shrink-0 text-muted-foreground"
      />
      <span className="shrink-0 text-muted-foreground">Base</span>
      <span className="min-w-0 truncate font-medium text-foreground">
        {baseModel}
      </span>
    </>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        {onSearchHub ? (
          <button
            type="button"
            onClick={() => onSearchHub(searchTerm)}
            className="inline-flex h-6 max-w-full cursor-pointer items-center gap-1.5 rounded-full bg-muted px-2.5 text-[11.5px] transition-colors hover:bg-muted/80 dark:bg-[rgba(255,255,255,0.04)]"
          >
            {content}
          </button>
        ) : (
          <span className="inline-flex h-6 max-w-full items-center gap-1.5 rounded-full bg-muted px-2.5 text-[11.5px] dark:bg-[rgba(255,255,255,0.04)]">
            {content}
          </span>
        )}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        Search this base model in Hub
      </TooltipContent>
    </Tooltip>
  );
}

type VramInfo = { est: number; status: "fits" | "tight" | "exceeds" } | null;

function ModelStatusChips({
  isDataset,
  isGguf,
  chatOnly,
  unslothSupport,
  vramInfo,
}: {
  isDataset: boolean;
  isGguf: boolean;
  chatOnly: boolean;
  unslothSupport: UnslothSupport;
  vramInfo: VramInfo;
}) {
  const showUnsupported = !isDataset && unslothSupport.status === "unsupported";
  // The format-unsupported chip already explains itself; this one covers the
  // supported-format model a chat-only host still can't run.
  const showChatOnly = !isDataset && !isGguf && chatOnly && !showUnsupported;
  const showVram = !isDataset && vramInfo && !isGguf;
  if (!showUnsupported && !showChatOnly && !showVram) return null;

  const vramTone = vramInfo
    ? vramInfo.status === "exceeds"
      ? "danger"
      : vramInfo.status === "tight"
        ? "warning"
        : "success"
    : "success";
  const vramLabel = vramInfo
    ? vramInfo.status === "exceeds"
      ? `Over VRAM budget · ~${vramInfo.est} GB`
      : vramInfo.status === "tight"
        ? `Tight fit · ~${vramInfo.est} GB`
        : `Likely fits · ~${vramInfo.est} GB`
    : "";
  const vramDetail = vramInfo
    ? vramInfo.status === "exceeds"
      ? "A 4-bit load is likely to exceed the current GPU budget. Higher-precision loads need even more."
      : vramInfo.status === "tight"
        ? "A 4-bit load should fit, with limited headroom for context and activations."
        : "A 4-bit load should fit comfortably on the current GPU."
    : "";

  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      {showUnsupported && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
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
              Still downloadable to your Hugging Face cache.
            </span>
          </TooltipContent>
        </Tooltip>
      )}
      {showChatOnly && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <span tabIndex={0} className="inline-flex outline-none">
              <StatusChip tone="warning" label="GGUF-only device" />
            </span>
          </TooltipTrigger>
          <TooltipContent
            side="bottom"
            sideOffset={6}
            className="tooltip-compact max-w-xs"
          >
            This device has no supported GPU or usable MLX, so only GGUF models
            can run here.
            <span className="mt-1 block text-[10.5px] font-normal text-white/75">
              Still downloadable to your Hugging Face cache.
            </span>
          </TooltipContent>
        </Tooltip>
      )}
      {showVram && vramInfo && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
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

export type ModelInspectorRuntime = {
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
};

export type ModelInspectorActions = {
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onLoadLocal: (opts?: {
    ggufVariant?: string;
    expectedBytes?: number;
  }) => void;
  onUseInChat: () => void;
  onTrain?: () => void;
  onInventoryChange?: () => void;
  onSearchHub?: (query: string) => void;
};

export const ModelInspector = memo(function ModelInspector({
  model,
  runtime,
  actions,
  isDataset = false,
  metadataUnavailable = false,
  selectionHiddenByFilters = false,
}: {
  model: SelectedModelView | null;
  isDataset?: boolean;
  metadataUnavailable?: boolean;
  selectionHiddenByFilters?: boolean;
  runtime: ModelInspectorRuntime;
  actions: ModelInspectorActions;
}) {
  const {
    isActive,
    activeGgufVariant,
    isLoadingThisModel,
    loadingPhase,
    minMemory,
    vramInfo,
    gpuGb,
    systemRamGb,
  } = runtime;
  const {
    onLoad,
    onLoadLocal,
    onUseInChat,
    onTrain,
    onInventoryChange,
    onSearchHub,
  } = actions;
  const deviceType = usePlatformStore((s) => s.deviceType);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const hfToken = useHfTokenStore((s) => s.token);
  const datasetRepoId = isDataset && model?.hubRepoId ? model.hubRepoId : null;
  const datasetSize = useDatasetSize(datasetRepoId, {
    token: hfToken || undefined,
  });
  // Inventory rows are snapshots; the download manager is the live source of
  // truth. When a download is in flight, route through the download-aware section
  // so progress/cancel stays visible across refreshes.
  const activeDownloadRepoId = model?.hubRepoId ?? null;
  const hasActiveHubDownload = useDownloadManagerStore((state) =>
    activeDownloadRepoId
      ? selectActiveJob(
          state,
          isDataset ? "dataset" : "model",
          activeDownloadRepoId,
        ) !== null
      : false,
  );
  const readmeRepoId =
    model?.hubRepoId ?? (isDataset ? null : (model?.baseModelHubId ?? null));
  const readmeSubject = isDataset
    ? "dataset"
    : model?.hubRepoId
      ? "model"
      : "baseModel";
  const supportModelId =
    model?.hubRepoId ??
    model?.baseModelHubId ??
    (model?.baseModelSource === "huggingface" ? model.baseModel : null);
  const supportPipelineTag = model?.pipelineTag;
  const supportTagsKey = model?.tags?.join("\0") ?? "";
  const supportLibraryName = model?.libraryName;
  const supportQuantMethod = model?.quantMethod;
  const unslothSupport = useMemo<UnslothSupport>(() => {
    return classifyUnslothSupport({
      modelId: supportModelId,
      pipelineTag: supportPipelineTag,
      tags: supportTagsKey ? supportTagsKey.split("\0") : undefined,
      libraryName: supportLibraryName,
      deviceType,
      quantMethod: supportQuantMethod,
    });
  }, [
    deviceType,
    supportLibraryName,
    supportModelId,
    supportPipelineTag,
    supportQuantMethod,
    supportTagsKey,
  ]);

  const deferredReadmeRepoId = useDeferredValue(readmeRepoId);
  const readmeReady = deferredReadmeRepoId === readmeRepoId;

  if (!model) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3 py-16 text-center">
        <div className="inline-flex size-12 items-center justify-center rounded-[14px] bg-muted text-muted-foreground">
          <HugeiconsIcon icon={CubeIcon} strokeWidth={1.5} className="size-5" />
        </div>
        <div className="space-y-1">
          <p className="text-[15px] font-semibold tracking-tight text-foreground">
            Select a {isDataset ? "dataset" : "model"}
          </p>
          <p className="max-w-sm text-[12.5px] leading-5 text-muted-foreground">
            {isDataset
              ? "Choose a dataset from the catalog to inspect its download state and details."
              : "Choose an item from the catalog to inspect its runtime fit, download state, and model card."}
          </p>
        </div>
      </div>
    );
  }

  const updatedRaw = model.updatedAt
    ? formatRelativeShort(model.updatedAt)
    : formatLocalUpdated(model.localUpdatedAt);
  const updatedLabel = updatedRaw === "Unknown update" ? "N/A" : updatedRaw;
  const createdLabel = model.createdAt ? formatShortDate(model.createdAt) : null;
  const libraryLabel = isDataset ? null : formatLibrary(model.libraryName);
  const gatedAccess = model.gated !== false && model.gated !== undefined;
  const downloadsTooltip =
    model.downloadsAllTime != null ? (
      <>
        Downloads (30 days)
        <span className="mt-1 block text-[10.5px] font-normal text-white/75">
          {formatCompact(model.downloadsAllTime)} all time
        </span>
      </>
    ) : (
      "Downloads"
    );
  const taskLabel = formatPipelineTag(model.pipelineTag) ?? "General";
  const licenseLabel = model.license ?? "N/A";
  const baseModelSearchTerm = model.baseModelHubId ?? model.baseModel ?? null;
  const paramsLabel = model.totalParams
    ? formatCompact(model.totalParams)
    : "N/A";
  const unslothSupported = unslothSupport.status !== "unsupported";
  // Embedding-only non-GGUF repos have no generative head, so keep them out of
  // the Run gate. Prefer the pipeline tag, else the capability heuristic.
  const isEmbeddingOnly =
    !model.isGguf &&
    model.capabilities.some((c) => c.key === "embedding") &&
    (EMBEDDING_PIPELINE_TAGS.has(model.pipelineTag?.toLowerCase() ?? "") ||
      !model.capabilities.some(
        (c) =>
          c.key === "conversational" ||
          c.key === "tools" ||
          c.key === "reasoning" ||
          c.key === "code" ||
          c.key === "vision" ||
          c.key === "audio",
      ));
  // Chat-only hosts (no supported GPU / usable MLX) run inference only through
  // llama.cpp, so only GGUF is loadable.
  const canRunModel =
    !isDataset &&
    (model.runtimeCapabilities?.canChat ?? true) &&
    !isEmbeddingOnly &&
    (model.isGguf || (!chatOnly && unslothSupported));
  const canTrainModel =
    !isDataset &&
    (model.runtimeCapabilities?.canTrain ?? false) &&
    model.modelFormat !== "gguf" &&
    model.modelFormat !== "adapter" &&
    unslothSupported;

  const languages = parseLanguageTags(model.tags);
  const datasetSizeBytes =
    datasetSize?.numBytesParquet ?? datasetSize?.numBytesOriginal ?? null;

  return (
    <div className="flex flex-col">
      <div className="pb-2 pt-1">
        <div className="flex items-center gap-4">
          <OwnerAvatar
            owner={model.owner}
            repoName={model.title}
            className="size-[60px] rounded-[18px] text-[19px]"
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <h2 className="truncate text-[25px] font-semibold leading-[31px] tracking-normal text-foreground">
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
            <div className="mt-0.5 flex min-w-0 items-center gap-1 text-[15px] leading-[24px] text-muted-foreground">
              <span className="truncate">{model.owner}</span>
              {model.owner.toLowerCase() === "unsloth" && (
                <span
                  aria-label="Verified Unsloth"
                  className="hub-verified-badge size-[18px] shrink-0 text-verified"
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
          {model.private && <AccessChip label="Private" />}
          {gatedAccess && <AccessChip label="Gated" />}
          {model.capabilities.map((capability) => (
            <CapabilityPill key={capability.key} capability={capability} />
          ))}
          {!isDataset && model.baseModel && baseModelSearchTerm && (
            <BaseModelSearchChip
              baseModel={model.baseModel}
              searchTerm={baseModelSearchTerm}
              onSearchHub={onSearchHub}
            />
          )}
        </div>
      </div>

      {isDataset &&
        model.hubRepoId &&
        (!model.isLocal || hasActiveHubDownload) && (
          <InspectorDownloadSlot>
            <DatasetDownloadSection
              repoId={model.hubRepoId}
              isDownloaded={model.isDownloaded}
              isPartial={model.isPartial ?? false}
              partialTransport={model.partialTransport ?? null}
              cachePath={model.path}
              knownBytes={model.cachedBytes}
              onTrain={onTrain}
              onChange={onInventoryChange}
            />
          </InspectorDownloadSlot>
        )}
      {isDataset && model.isLocal && !hasActiveHubDownload && model.path && (
        <InspectorDownloadSlot>
          <LocalDatasetCard
            sourceLabel={model.sourceLabel}
            source={model.localSource ?? "custom"}
            path={model.path}
            onTrain={onTrain}
          />
        </InspectorDownloadSlot>
      )}
      {!isDataset && (
        <InspectorDownloadSlot>
          {model.isLocal && !hasActiveHubDownload ? (
            <LocalOnDeviceCard
              modelId={model.id}
              repoId={model.hubRepoId}
              sourceLabel={model.sourceLabel}
              source={model.localSource ?? "custom"}
              path={model.path ?? model.displayId}
              isGguf={model.isGguf}
              requiresVariant={model.requiresVariant}
              modelFormat={model.modelFormat}
              baseModel={model.baseModel}
              baseModelSource={model.baseModelSource}
              baseModelHubId={model.baseModelHubId}
              baseModelSummary={model.baseModelSummary}
              adapterType={model.adapterType}
              trainingMethod={model.trainingMethod}
              canRun={canRunModel}
              isActive={isActive}
              activeGgufVariant={activeGgufVariant}
              isLoading={isLoadingThisModel}
              loadingPhase={loadingPhase}
              gpuGb={gpuGb}
              systemRamGb={systemRamGb}
              unsupportedReason={
                unslothSupport.status === "unsupported"
                  ? (unslothSupport.reason ?? "Unsupported format")
                  : null
              }
              onLoad={onLoadLocal}
              onUseInChat={onUseInChat}
              onTrain={
                model.isDownloaded && canTrainModel ? onTrain : undefined
              }
              onChange={onInventoryChange}
            />
          ) : (
            <DownloadSection
              repoId={model.isLocal ? (model.hubRepoId ?? model.id) : model.id}
              isGguf={model.isGguf}
              isDownloaded={model.isDownloaded}
              isPartial={model.isPartial ?? false}
              partialTransport={model.partialTransport ?? null}
              modelFormat={model.modelFormat}
              canRun={canRunModel}
              isActive={isActive}
              activeQuant={isActive ? (activeGgufVariant ?? null) : null}
              isLoadingThisModel={isLoadingThisModel}
              gpuGb={gpuGb}
              systemRamGb={systemRamGb}
              cachePath={model.path}
              knownBytes={model.cachedBytes}
              onLoad={model.isLocal ? onLoadLocal : onLoad}
              onUseInChat={onUseInChat}
              onTrain={
                model.isDownloaded && canTrainModel ? onTrain : undefined
              }
              onChange={onInventoryChange}
            />
          )}
        </InspectorDownloadSlot>
      )}

      <div className="pb-5 pt-5">
        {selectionHiddenByFilters && (
          <p className="mb-3 rounded-[8px] border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11.5px] leading-snug text-muted-foreground">
            Current selection is hidden by the active filters or search.
          </p>
        )}
        {metadataUnavailable && (
          <p className="mb-3 text-[11.5px] leading-snug text-muted-foreground">
            Couldn't load full details from Hugging Face. Some fields may be
            incomplete.
          </p>
        )}
        <StatGrid>
          <StatRow label="Updated" value={updatedLabel} icon={Calendar03Icon} />
          {createdLabel && (
            <StatRow
              label="Created"
              value={createdLabel}
              icon={CalendarAdd01Icon}
            />
          )}
          <StatRow
            label="Downloads"
            tooltip={downloadsTooltip}
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
            <StatRow label="Parameters" value={paramsLabel} icon={CpuIcon} />
          )}
          {libraryLabel && (
            <StatRow label="Library" value={libraryLabel} icon={LibraryIcon} />
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
          {isDataset &&
            datasetSize?.numSplits != null &&
            datasetSize.numSplits > 0 && (
              <StatRow
                label="Splits"
                value={String(datasetSize.numSplits)}
                icon={LayersLogoIcon}
              />
            )}
          {languages.length > 0 && (
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
          <StatRow label="License" value={licenseLabel} icon={LicenseIcon} />
        </StatGrid>
        <ModelStatusChips
          isDataset={isDataset}
          isGguf={model.isGguf}
          chatOnly={chatOnly}
          unslothSupport={unslothSupport}
          vramInfo={vramInfo}
        />
      </div>

      <div className="max-w-[860px] space-y-4 pt-4">
        {readmeReady && readmeRepoId && (
          <ModelReadme
            repoId={readmeRepoId}
            kind={isDataset ? "dataset" : "model"}
            subject={readmeSubject}
          />
        )}
      </div>
    </div>
  );
});
