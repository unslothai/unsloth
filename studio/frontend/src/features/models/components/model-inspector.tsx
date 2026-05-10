// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn, formatCompact } from "@/lib/utils";
import { Streamdown } from "streamdown";
import {
  Calendar03Icon,
  CheckmarkCircle02Icon,
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
  PlayIcon,
  RamMemoryIcon,
  Share05Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { DatasetDownloadSection } from "./dataset-download-section";
import { ModelReadme } from "./model-readme";
import { OwnerAvatar } from "./owner-avatar";
import { CapabilityPill } from "./shared";
import { DownloadSection } from "./download-section";
import { ACCENT_RING, ACCENT_TEXT, pickAccent } from "../lib/accent";
import {
  type DatasetSizeInfo,
  fetchDatasetSize,
} from "../lib/dataset-size";
import { formatBytes, formatRelativeShort } from "../lib/format";
import { formatLocalUpdated, formatPipelineTag } from "../lib/view-models";
import type { SelectedModelView } from "../types";

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
        <span className="view-repo-link relative inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
          <HugeiconsIcon
            icon={Share05Icon}
            strokeWidth={1.75}
            className="pointer-events-none size-4"
          />
          <span aria-label="View repository">
            <Streamdown mode="static">{`[View repository](${url})`}</Streamdown>
          </span>
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        Open on Hugging Face
      </TooltipContent>
    </Tooltip>
  );
}

function CopyRepoButton({ repoId }: { repoId: string }) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    },
    [],
  );

  const handleCopy = async (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      await navigator.clipboard.writeText(repoId);
      setCopied(true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => setCopied(false), 1500);
    } catch {}
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

function LocalLoadPanel({
  isActive,
  isLoading,
  loadingPhase,
  sourceLabel,
  path,
  onLoad,
}: {
  isActive: boolean;
  isLoading: boolean;
  loadingPhase?: "downloading" | "starting";
  sourceLabel: string;
  path: string;
  onLoad: () => void;
}) {
  return (
    <div className="rounded-[14px] border border-border bg-muted/30 p-3">
      <div className="flex flex-col gap-2.5">
        <div className="space-y-0.5">
          <p className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Load
          </p>
          <p className="text-[12.5px] leading-5 text-muted-foreground">
            This model exists on the device via {sourceLabel}. Loading it does
            not trigger a new Hub download.
          </p>
        </div>
        <div className="rounded-[10px] bg-background/80 px-3 py-2 text-[11.5px] leading-5 text-muted-foreground break-all">
          {path}
        </div>
        <Button
          variant={isActive ? "secondary" : "dark"}
          className="h-9 rounded-[10px]"
          disabled={isActive || isLoading}
          onClick={onLoad}
        >
          {isLoading ? (
            <>
              <HugeiconsIcon
                icon={CpuIcon}
                strokeWidth={1.75}
                className="size-4 animate-pulse"
              />
              {loadingPhase === "downloading" ? "Preparing…" : "Loading…"}
            </>
          ) : isActive ? (
            <>
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                strokeWidth={2}
                className="size-4"
              />
              Loaded
            </>
          ) : (
            <>
              <HugeiconsIcon
                icon={PlayIcon}
                strokeWidth={1.75}
                className="size-4"
              />
              Load local model
            </>
          )}
        </Button>
      </div>
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
  onInventoryChange,
  isDataset = false,
}: {
  model: SelectedModelView | null;
  isDataset?: boolean;
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
  onInventoryChange?: () => void;
}) {
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
      <div className="flex h-[calc(100dvh-205px)] flex-col items-center justify-center gap-3 px-6 py-10 text-center">
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

  const accent = pickAccent(model.capabilities);

  const languages = (() => {
    const tags = model.tags ?? [];
    const langs: string[] = [];
    for (const t of tags) {
      if (t.startsWith("language:")) {
        const code = t.slice("language:".length);
        if (code && !langs.includes(code)) langs.push(code);
      }
    }
    return langs;
  })();

  return (
    <div className="flex h-[calc(100dvh-205px)] flex-col">
      <div className="shrink-0 px-6 pb-2 pt-4">
        <div className="flex items-center gap-3.5">
          <OwnerAvatar
            owner={model.owner}
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
            <p className="mt-0.5 min-w-0 truncate text-[14px] leading-[22px] text-muted-foreground">
              {model.owner}
            </p>
          </div>
        </div>

        {(isDataset || isActive || !isDataset || model.capabilities.length > 0) && (
          <div className="mt-4 flex flex-wrap items-center gap-1.5">
            {isDataset && (
              <span className="inline-flex shrink-0 items-center rounded-full border border-violet-500/40 bg-transparent px-2 py-0.5 text-[11.5px] font-medium text-violet-600 dark:text-violet-400">
                Dataset
              </span>
            )}
            {!isDataset && (
              <span className="inline-flex h-6 items-center gap-1.5 rounded-full bg-muted px-2.5 text-[11.5px] font-medium text-foreground">
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
        )}

      </div>

      {isDataset && model.hubRepoId && !model.isLocal && (
        <div className="shrink-0 px-6 pt-3">
          <DatasetDownloadSection
            repoId={model.hubRepoId}
            isDownloaded={model.isDownloaded}
            onChange={onInventoryChange}
          />
        </div>
      )}
      {!isDataset && (
        <div className="shrink-0 px-6 pt-3">
          {model.isLocal ? (
            <LocalLoadPanel
              isActive={isActive}
              isLoading={isLoadingThisModel}
              loadingPhase={loadingPhase}
              sourceLabel={model.sourceLabel}
              path={model.path ?? model.displayId}
              onLoad={onLoadLocal}
            />
          ) : (
            <DownloadSection
              repoId={model.id}
              isGguf={model.isGguf}
              isDownloaded={model.isDownloaded}
              isActive={isActive}
              activeQuant={isActive ? (activeGgufVariant ?? null) : null}
              isLoadingThisModel={isLoadingThisModel}
              loadingPhase={loadingPhase}
              gpuGb={gpuGb}
              systemRamGb={systemRamGb}
              onLoad={onLoad}
              onUseInChat={onUseInChat}
              onChange={onInventoryChange}
            />
          )}
        </div>
      )}

      <div className="shrink-0 px-6 pb-5 pt-5">
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
          {isDataset &&
            (datasetSize?.numBytesParquet ?? datasetSize?.numBytesOriginal) !=
              null && (
              <StatRow
                label="Size"
                value={formatBytes(
                  (datasetSize?.numBytesParquet ??
                    datasetSize?.numBytesOriginal) as number,
                )}
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
        className="min-h-0 flex-1 space-y-7 overflow-y-auto pl-6 pr-4 pt-5 pb-0 [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
        {!isDataset && vramInfo && !model.isGguf && (
          <div
            className={cn(
              "rounded-[12px] border px-3 py-2.5 text-[12px] leading-5",
              vramInfo.status === "exceeds"
                ? "border-red-500/20 bg-red-500/8 text-red-700 dark:text-red-300"
                : vramInfo.status === "tight"
                  ? "border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                  : "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
            )}
          >
            Estimated memory load is about {vramInfo.est} GB.{" "}
            {vramInfo.status === "exceeds"
              ? "This is likely above the current GPU budget."
              : vramInfo.status === "tight"
                ? "This should fit, but with limited headroom."
                : "This should fit comfortably on the current GPU."}
          </div>
        )}

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
