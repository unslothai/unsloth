// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn, formatCompact } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  Calendar03Icon,
  CheckmarkCircle02Icon,
  CpuIcon,
  CubeIcon,
  DownloadCircle02Icon,
  FavouriteIcon,
  LicenseIcon,
  PackageIcon,
  PlayIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { DatasetDownloadSection } from "./dataset-download-section";
import { ModelReadme } from "./model-readme";
import { OwnerAvatar } from "./owner-avatar";
import { CapabilityPill } from "./shared";
import { DownloadSection } from "./download-section";
import { formatRelativeShort } from "../lib/format";
import { formatLocalUpdated, formatPipelineTag } from "../lib/view-models";
import type { SelectedModelView } from "../types";

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
    <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
      <div className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <HugeiconsIcon
          icon={icon}
          strokeWidth={1.75}
          className="size-3 text-muted-foreground/70"
        />
        {label}
      </div>
      <div
        title={value}
        className="truncate text-[12px] font-medium tabular-nums text-foreground"
      >
        {value}
      </div>
    </div>
  );
}

function StatGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-[12px] border border-border/60 bg-muted/30">
      <div className="grid grid-cols-1 sm:grid-cols-2 sm:divide-x divide-border/50">
        {children}
      </div>
    </div>
  );
}

function StatColumn({ children }: { children: React.ReactNode }) {
  return <div className="divide-y divide-border/50">{children}</div>;
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
    <div className="rounded-[14px] border border-border/60 bg-muted/30 p-3">
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
  gpuLabel,
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
  gpuLabel: string;
  gpuGb?: number;
  systemRamGb?: number;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onLoadLocal: () => void;
  onUseInChat: () => void;
  onInventoryChange?: () => void;
}) {
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

  const updatedLabel = model.updatedAt
    ? formatRelativeShort(model.updatedAt)
    : formatLocalUpdated(model.localUpdatedAt);
  const taskLabel = formatPipelineTag(model.pipelineTag) ?? "General";
  const licenseLabel = model.license ?? "—";
  const paramsLabel = model.totalParams
    ? formatCompact(model.totalParams)
    : model.isGguf
      ? "Unavailable"
      : "—";

  return (
    <div className="flex h-[calc(100dvh-205px)] flex-col pb-3">
      <div className="shrink-0 border-b border-border/60 px-5 pb-3 pt-4">
        <div className="flex items-center gap-3">
          <OwnerAvatar
            owner={model.owner}
            className="size-10 rounded-[11px] text-[14px]"
          />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[17px] font-semibold leading-[22px] tracking-[-0.02em] text-foreground">
              {model.title}
            </h2>
            <p className="truncate text-[12.5px] leading-[18px] text-muted-foreground">
              {model.owner}
            </p>
          </div>
        </div>

        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {isDataset && (
            <span className="inline-flex h-5 items-center rounded-full border border-violet-500/40 bg-transparent px-2 text-[9.5px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-400">
              Dataset
            </span>
          )}
          {!isDataset && model.isGguf && (
            <span className="inline-flex h-5 items-center rounded-full border border-blue-500/40 bg-transparent px-2 text-[9.5px] font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
              GGUF
            </span>
          )}
          {!isDataset && model.isDownloaded && (
            <span className="inline-flex h-5 items-center gap-1 rounded-full border border-emerald-500/40 px-2 text-[9.5px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                strokeWidth={2.5}
                className="size-2.5"
              />
              On device
            </span>
          )}
          {isActive && (
            <span className="inline-flex h-5 items-center rounded-full border border-primary/40 px-2 text-[9.5px] font-semibold uppercase tracking-wider text-primary">
              Loaded
            </span>
          )}
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {model.hubRepoId && (
            <a
              href={`https://huggingface.co/${isDataset ? "datasets/" : ""}${model.hubRepoId}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-9 items-center gap-1.5 rounded-[10px] border border-border/60 bg-muted/40 px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-muted"
            >
              View repository
              <HugeiconsIcon
                icon={ArrowUpRight01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </a>
          )}
          {isActive && (
            <Button
              variant="dark"
              className="h-9 rounded-[10px]"
              onClick={onUseInChat}
            >
              Open in chat
            </Button>
          )}
        </div>
      </div>

      {isDataset && model.hubRepoId && !model.isLocal && (
        <div className="shrink-0 border-b border-border/60 px-5 py-4">
          <DatasetDownloadSection
            repoId={model.hubRepoId}
            isDownloaded={model.isDownloaded}
            onChange={onInventoryChange}
          />
        </div>
      )}
      {!isDataset && (
        <div className="shrink-0 border-b border-border/60 px-5 py-4">
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
              onChange={onInventoryChange}
            />
          )}
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {!isDataset && model.capabilities.length > 0 && (
          <div className="space-y-2">
            <p className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              Capabilities
            </p>
            <div className="flex flex-wrap gap-1.5">
              {model.capabilities.map((capability) => (
                <CapabilityPill key={capability.key} capability={capability} />
              ))}
            </div>
          </div>
        )}

        <StatGrid>
          <StatColumn>
            <StatRow
              label={isDataset ? "Type" : "Task"}
              value={isDataset ? "Dataset" : taskLabel}
              icon={CubeIcon}
            />
            <StatRow
              label="License"
              value={licenseLabel}
              icon={LicenseIcon}
            />
            <StatRow
              label="Downloads"
              value={
                model.downloads !== undefined
                  ? formatCompact(model.downloads)
                  : "—"
              }
              icon={DownloadCircle02Icon}
            />
            <StatRow
              label="Likes"
              value={
                model.likes !== undefined ? formatCompact(model.likes) : "—"
              }
              icon={FavouriteIcon}
            />
          </StatColumn>
          {!isDataset && (
            <StatColumn>
              <StatRow label="Parameters" value={paramsLabel} icon={CpuIcon} />
              <StatRow
                label="Memory"
                value={minMemory ?? "—"}
                icon={RamMemoryIcon}
              />
              <StatRow
                label="Updated"
                value={updatedLabel}
                icon={Calendar03Icon}
              />
              <StatRow label="GPU fit" value={gpuLabel} icon={PackageIcon} />
            </StatColumn>
          )}
          {isDataset && (
            <StatColumn>
              <StatRow
                label="Updated"
                value={updatedLabel}
                icon={Calendar03Icon}
              />
            </StatColumn>
          )}
        </StatGrid>

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
          <div className="space-y-2 border-t border-border/60 pt-4">
            <p className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {isDataset ? "Dataset card" : "Model card"}
            </p>
            <ModelReadme
              repoId={model.hubRepoId}
              kind={isDataset ? "dataset" : "model"}
            />
          </div>
        )}
      </div>
    </div>
  );
}
