// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVllmAvailable, useVllmPlatform } from "@/features/model-picker";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { classifyUnslothSupport } from "@/features/hub/hooks/use-hub-model-search";
import {
  formatRelativeLong,
  formatRelativeShort,
} from "@/features/hub/lib/format";
import {
  formatModelParamLabel,
  formatPipelineTag,
} from "@/features/hub/lib/view-models";
import { cn, formatCompact } from "@/lib/utils";
import {
  Download01Icon,
  FavouriteIcon,
  LayoutTwoColumnIcon,
  Refresh01Icon,
  ViewSidebarLeftIcon,
} from "@hugeicons/core-free-icons";
import {
  ChevronLeftIcon,
} from "lucide-react";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, memo, useMemo } from "react";
import type { DiscoverRow } from "../types";
import {
  DOWNLOADING_DOT_CLASS,
  buildRowStatusTooltip,
} from "./models-catalog-rows";
import { OwnerAvatar } from "./owner-avatar";
import { AccessGlyphs } from "./shared";

export const RESULT_CARD_HEIGHT_PX = 78;
export const RESULT_ROW_GAP_PX = 8;
export const RESULT_ROW_HEIGHT_PX = RESULT_CARD_HEIGHT_PX + RESULT_ROW_GAP_PX;
// Phone widths (below md) use a shorter card; ResultCard's max-md classes match it.
export const RESULT_CARD_MOBILE_HEIGHT_PX = 66;
export const RESULT_ROW_MOBILE_HEIGHT_PX =
  RESULT_CARD_MOBILE_HEIGHT_PX + RESULT_ROW_GAP_PX;
export const RESULT_GRID_HEIGHT_PX = 64;
export const RESULT_GRID_ROW_GAP_PX = 8;
export const RESULT_GRID_ROW_HEIGHT_PX =
  RESULT_GRID_HEIGHT_PX + RESULT_GRID_ROW_GAP_PX;
// Compact rows for the split-view master pane.
export const RESULT_SPLIT_HEIGHT_PX = 56;
export const RESULT_SPLIT_ROW_HEIGHT_PX =
  RESULT_SPLIT_HEIGHT_PX + RESULT_ROW_GAP_PX;

export type AllModelsView = "two" | "split";

function ViewToggleButton({
  active,
  label,
  icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: IconSvgElement;
  onClick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-pressed={active}
          aria-label={label}
          onClick={onClick}
          className={cn(
            "relative z-10 inline-flex size-8 cursor-pointer items-center justify-center rounded-full transition-colors",
            active
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

export type InventorySort = "recent" | "name" | "size" | "format";

export function HubListHeader({
  title,
  subtitle,
  count,
  view,
  onViewChange,
  onBack,
  onRefresh,
  isRefreshing,
  actions,
}: {
  title: string;
  subtitle?: string;
  count?: number;
  view?: AllModelsView;
  onViewChange?: (view: AllModelsView) => void;
  onBack?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  actions?: ReactNode;
}) {
  const accessibleLabel =
    count && count > 0 ? `${title} (${count.toLocaleString()})` : undefined;

  return (
    <div
      className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 pb-3"
      aria-label={accessibleLabel}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            aria-label="Back to feed"
            // Pull the button left so the inset chevron lines up with the
            // avatars below, just inside the row hover's left edge.
            className="hub-section-chevron -ml-3 inline-flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground"
          >
            <ChevronLeftIcon
              strokeWidth={2}
              className="size-4"
            />
          </button>
        )}
        <div className="min-w-0 space-y-0.5">
          {/* truncate keeps the heading on one line and clips a long search
              query with an ellipsis instead of overflowing the pills. */}
          <h2 className="truncate text-ui-18 font-semibold tracking-[-0.02em] text-foreground">
            {title}
          </h2>
          {subtitle && (
            <p className="text-ui-12p5 leading-tight text-muted-foreground">
              {subtitle}
            </p>
          )}
        </div>
        {onRefresh && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label="Refresh"
                onClick={onRefresh}
                // Tiny drop so the icon aligns with the heading text.
                className="inline-flex size-7 shrink-0 translate-y-px cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={Refresh01Icon}
                  strokeWidth={1.75}
                  className={cn("size-[calc(11px*var(--ui-space-scale,1))]", isRefreshing && "animate-spin")}
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="tooltip-compact">
              Refresh from Hugging Face
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      {(actions || onViewChange) && (
        <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-2">
          {actions}
          {onViewChange && (
            <div
              className="hub-tab-toggle relative inline-flex h-8 shrink-0 items-center rounded-full"
              aria-label="Results layout"
            >
              <span
                aria-hidden="true"
                className="hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 w-8 rounded-full transition-transform duration-200 ease-out"
                style={{
                  transform: `translateX(${view === "split" ? 100 : 0}%)`,
                }}
              />
              <ViewToggleButton
                active={view === "two"}
                label="Two columns"
                icon={LayoutTwoColumnIcon}
                onClick={() => onViewChange("two")}
              />
              <ViewToggleButton
                active={view === "split"}
                label="Split view"
                icon={ViewSidebarLeftIcon}
                onClick={() => onViewChange("split")}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const STATUS_DOT_CLASS = "inline-block size-[calc(6px*var(--ui-space-scale,1))] shrink-0 rounded-full";

function TitleMarkers({
  format,
  gated,
  isPrivate,
  partial,
  downloading = false,
  unsupported,
  onDevice,
}: {
  format?: "gguf" | "checkpoint" | null;
  gated?: false | "auto" | "manual";
  isPrivate?: boolean;
  partial: boolean;
  downloading?: boolean;
  unsupported: boolean;
  onDevice: boolean;
}) {
  return (
    <span className="flex shrink-0 items-center gap-1.5">
      {format === "gguf" && (
        <span
          role="img"
          aria-label="GGUF"
          className={cn(STATUS_DOT_CLASS, "bg-format-gguf")}
        />
      )}
      {format === "checkpoint" && (
        <span
          role="img"
          aria-label="Safetensors"
          className={cn(STATUS_DOT_CLASS, "bg-format-checkpoint")}
        />
      )}
      <AccessGlyphs gated={gated} isPrivate={isPrivate} tooltip={false} />
      {partial && !downloading && (
        <span
          role="img"
          aria-label="Partial download"
          className={cn(STATUS_DOT_CLASS, "bg-status-warning")}
        />
      )}
      {partial && downloading && (
        <span
          role="img"
          aria-label="Downloading"
          className={cn(STATUS_DOT_CLASS, DOWNLOADING_DOT_CLASS)}
        />
      )}
      {unsupported && (
        <span
          role="img"
          aria-label="May not be supported yet"
          className={cn(STATUS_DOT_CLASS, "bg-status-danger")}
        />
      )}
      {onDevice && (
        <span
          role="img"
          aria-label="On device"
          className={cn(STATUS_DOT_CLASS, "bg-status-success")}
        />
      )}
    </span>
  );
}

function VerifiedOwner({ owner }: { owner: string }) {
  return (
    <span className="flex min-w-0 items-center gap-1">
      <span className="truncate">{owner}</span>
      {owner.toLowerCase() === "unsloth" && (
        <span
          aria-label="Verified Unsloth"
          className="hub-verified-badge size-3.5 shrink-0 text-verified"
        />
      )}
    </span>
  );
}

function useResultRowModel(
  row: DiscoverRow,
  deviceType: string | null,
  isDataset: boolean,
) {
  const vllmAvailable = useVllmAvailable();
  const vllmPlatform = useVllmPlatform();
  const support = useMemo(
    () =>
      isDataset
        ? null
        : classifyUnslothSupport({
            modelId: row.id,
            pipelineTag: row.result.pipelineTag,
            tags: row.result.tags,
            libraryName: row.result.libraryName,
            deviceType,
            quantMethod: row.result.quantMethod,
            vllmAvailable,
            vllmPlatform,
          }),
    [isDataset, row.id, row.result, deviceType, vllmAvailable, vllmPlatform],
  );
  const sizeLabel = formatModelParamLabel(row.repo, row.result.totalParams);
  const taskLabel = isDataset
    ? null
    : formatPipelineTag(row.result.pipelineTag);
  const unsupported = support?.status === "unsupported" && !support?.supportedIn;
  return {
    support,
    unsupported,
    partial: row.isAvailableOnDevice && row.isPartialOnDevice,
    downloading:
      row.isAvailableOnDevice &&
      row.isPartialOnDevice &&
      row.isDownloadingOnDevice === true,
    onDevice: row.isAvailableOnDevice && !row.isPartialOnDevice,
    sizeLabel: sizeLabel !== "N/A" ? sizeLabel : null,
    taskLabel,
  };
}

export const ResultCard = memo(function ResultCard({
  row,
  deviceType,
  isDataset,
  showFormatDot = true,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  showFormatDot?: boolean;
  onSelect: (id: string) => void;
}) {
  const {
    support,
    unsupported,
    partial,
    downloading,
    onDevice,
    sizeLabel,
    taskLabel,
  } =
    useResultRowModel(row, deviceType, isDataset);
  const format =
    isDataset || !showFormatDot
      ? null
      : row.result.isGguf
        ? "gguf"
        : "checkpoint";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
    downloading,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });

  const textParts: Array<{ key: string; node: ReactNode }> = [];
  if (taskLabel) {
    textParts.push({
      key: "task",
      node: <span className="min-w-0 truncate">{taskLabel}</span>,
    });
  }
  if (sizeLabel) {
    textParts.push({
      key: "size",
      node: <span className="shrink-0">{sizeLabel}</span>,
    });
  }
  if (row.result.updatedAt) {
    textParts.push({
      key: "updated",
      node: (
        <span className="shrink-0">
          <span className="hidden @[26rem]:inline">Updated </span>
          {formatRelativeLong(row.result.updatedAt)}
        </span>
      ),
    });
  }

  const card = (
    <button
      type="button"
      aria-label={row.repo}
      onClick={() => onSelect(row.id)}
      className="hub-result-row hub-result-card group/row @container flex h-full w-full cursor-pointer items-center gap-3.5 px-4 text-left max-md:gap-3 max-md:px-3 outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
    >
      <OwnerAvatar
        owner={row.owner}
        repoName={row.repo}
        className="size-[calc(52px*var(--ui-space-scale,1))] shrink-0 rounded-[16px] text-ui-16 max-md:size-[calc(42px*var(--ui-space-scale,1))] max-md:rounded-[13px] max-md:text-ui-14 ring-1 ring-[rgb(0_0_0_/_calc(0.05*var(--contrast-edge-gain,1)))] dark:ring-[rgb(255_255_255_/_calc(0.1*var(--contrast-edge-gain,1)))]"
        remote={false}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-ui-15 font-semibold leading-ui-18 text-foreground max-md:text-ui-14">
            {row.repo}
          </span>
          <TitleMarkers
            format={format}
            gated={row.result.gated}
            isPrivate={row.result.private}
            partial={partial}
            downloading={downloading}
            unsupported={unsupported}
            onDevice={onDevice}
          />
        </div>
        <span className="flex min-w-0 items-center gap-1 text-ui-12p5 leading-ui-16 text-muted-foreground/80">
          <VerifiedOwner owner={row.owner} />
        </span>
        <div className="flex min-w-0 items-center gap-2 overflow-hidden text-ui-11p5 leading-ui-16 tabular-nums text-muted-foreground/65">
          {/* Narrow cards drop the task label, then the date. */}
          {textParts.map((part, index) => (
            <span
              key={part.key}
              className={
                part.key === "task"
                  ? "hidden @[26rem]:contents"
                  : part.key === "updated"
                    ? "hidden @[19rem]:contents"
                    : "contents"
              }
            >
              {index > 0 && (
                <span
                  aria-hidden="true"
                  className={cn(
                    "shrink-0 text-muted-foreground/35",
                    index === 1 &&
                      textParts[0].key === "task" &&
                      "hidden @[26rem]:inline",
                  )}
                >
                  •
                </span>
              )}
              {part.node}
            </span>
          ))}
          <span className="hub-meta-tag">
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-3 shrink-0"
            />
            {formatCompact(row.result.downloads)}
          </span>
          <span className="hub-meta-tag">
            <HugeiconsIcon
              icon={FavouriteIcon}
              strokeWidth={1.75}
              className="size-3 shrink-0"
            />
            {formatCompact(row.result.likes)}
          </span>
        </div>
      </div>
    </button>
  );

  if (!tip) {
    return card;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{card}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact max-w-[calc(260px*var(--ui-space-scale,1))]">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
});

// Compact master-pane row for split view: avatar + name/owner left, stats right,
// with a selected highlight for the model shown in the detail pane.
export const ResultSplitRow = memo(function ResultSplitRow({
  row,
  deviceType,
  isDataset,
  selected,
  showFormatDot = true,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  selected: boolean;
  showFormatDot?: boolean;
  onSelect: (id: string) => void;
}) {
  const { support, unsupported, partial, downloading, onDevice } =
    useResultRowModel(
    row,
    deviceType,
    isDataset,
  );
  const format =
    isDataset || !showFormatDot
      ? null
      : row.result.isGguf
        ? "gguf"
        : "checkpoint";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
    downloading,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  const node = (
    <button
      type="button"
      aria-label={row.repo}
      aria-current={selected || undefined}
      data-selected={selected || undefined}
      onClick={() => onSelect(row.id)}
      className="group/row flex h-full w-full cursor-pointer items-center gap-2.5 rounded-[12px] px-2.5 text-left outline-none transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] data-[selected]:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset dark:hover:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:data-[selected]:bg-accent"
    >
      <OwnerAvatar
        owner={row.owner}
        repoName={row.repo}
        className="size-8 shrink-0 rounded-[9px] text-ui-12"
        remote={false}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-ui-12p5 font-semibold leading-ui-16 text-foreground">
            {row.repo}
          </span>
          <TitleMarkers
            format={format}
            gated={row.result.gated}
            isPrivate={row.result.private}
            partial={partial}
            downloading={downloading}
            unsupported={unsupported}
            onDevice={onDevice}
          />
        </div>
        <span className="mt-0.5 flex min-w-0 items-center gap-1 text-ui-10p5 leading-ui-14 text-muted-foreground/80">
          <VerifiedOwner owner={row.owner} />
        </span>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5 text-ui-10p5 tabular-nums text-muted-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1">
            <HugeiconsIcon
              icon={FavouriteIcon}
              strokeWidth={1.75}
              className="size-[calc(11px*var(--ui-space-scale,1))] shrink-0"
            />
            {formatCompact(row.result.likes)}
          </span>
          <span className="inline-flex items-center gap-1">
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-[calc(11px*var(--ui-space-scale,1))] shrink-0"
            />
            {formatCompact(row.result.downloads)}
          </span>
        </div>
        {row.result.updatedAt && (
          <span className="text-muted-foreground/60">
            {formatRelativeShort(row.result.updatedAt)}
          </span>
        )}
      </div>
    </button>
  );
  if (!tip) {
    return node;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{node}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact max-w-[calc(260px*var(--ui-space-scale,1))]">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
});
