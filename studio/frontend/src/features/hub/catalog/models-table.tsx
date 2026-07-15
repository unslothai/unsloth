// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn, formatCompact } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  Copy01Icon,
  Download01Icon,
  FavouriteIcon,
  LayoutTwoColumnIcon,
  LeftToRightListBulletIcon,
  LinkSquare02Icon,
  MoreVerticalIcon,
  Refresh01Icon,
  ViewSidebarLeftIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { Fragment, type ReactNode, memo, useMemo } from "react";
import { confirmExternalLink } from "../stores/external-link-confirm";
import type { DiscoverRow } from "../types";
import { HubOptionMenu } from "./hub-option-menu";
import { buildRowStatusTooltip } from "./models-catalog-rows";
import { OwnerAvatar } from "./owner-avatar";
import { AccessGlyphs, CapabilityPill } from "./shared";

export const RESULT_CARD_HEIGHT_PX = 78;
export const RESULT_ROW_GAP_PX = 8;
export const RESULT_ROW_HEIGHT_PX = RESULT_CARD_HEIGHT_PX + RESULT_ROW_GAP_PX;
export const RESULT_GRID_HEIGHT_PX = 64;
export const RESULT_GRID_ROW_GAP_PX = 8;
export const RESULT_GRID_ROW_HEIGHT_PX =
  RESULT_GRID_HEIGHT_PX + RESULT_GRID_ROW_GAP_PX;
// Compact rows for the split-view master pane.
export const RESULT_SPLIT_HEIGHT_PX = 56;
export const RESULT_SPLIT_ROW_HEIGHT_PX =
  RESULT_SPLIT_HEIGHT_PX + RESULT_ROW_GAP_PX;

export type AllModelsView = "grid" | "two" | "split";

// Shared column widths so header and rows align: two flexible lead columns
// (Model, Capabilities) plus fixed metric columns that drop on narrow viewports.
const LIST_COLS = {
  model: "flex min-w-0 flex-[2.4] items-center gap-3",
  caps: "hidden min-w-0 flex-[1.7] items-center gap-1.5 md:flex",
  capsModel: "hidden w-[132px] shrink-0 items-center gap-1.5 md:flex",
  size: "hidden w-[60px] shrink-0 lg:block",
  updated: "hidden w-[82px] shrink-0 xl:block",
  downloads: "hidden w-[104px] shrink-0 items-center gap-1.5 sm:flex",
  likes: "hidden w-[76px] shrink-0 items-center gap-1.5 sm:flex",
  actions: "flex w-[64px] shrink-0 items-center justify-end gap-0.5",
} as const;

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

export type InventorySort = "recent" | "name" | "size";

const INVENTORY_SORTS: { value: InventorySort; label: string }[] = [
  { value: "recent", label: "Recent" },
  { value: "name", label: "Name" },
  { value: "size", label: "Size" },
];

// Sort picker as a compact dropdown pill so it sits beside the view-mode tabs.
export function InventorySortControl({
  value,
  onChange,
}: {
  value: InventorySort;
  onChange: (value: InventorySort) => void;
}) {
  return (
    <HubOptionMenu<InventorySort>
      value={value}
      options={INVENTORY_SORTS}
      onValueChange={onChange}
      ariaLabel="Sort downloads"
      align="end"
      className="h-8 text-[11.5px]"
    />
  );
}

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
      className="flex items-center justify-between gap-4 pb-3"
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
            <HugeiconsIcon
              icon={ArrowLeft01Icon}
              strokeWidth={2}
              className="size-4"
            />
          </button>
        )}
        <div className="min-w-0 space-y-0.5">
          <h2 className="text-[18px] font-semibold tracking-[-0.02em] text-foreground">
            {title}
          </h2>
          {subtitle && (
            <p className="text-[12.5px] leading-tight text-muted-foreground">
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
                  className={cn("size-[11px]", isRefreshing && "animate-spin")}
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
        <div className="flex shrink-0 items-center gap-2">
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
                  transform: `translateX(${
                    (view === "split" ? 1 : view === "grid" ? 2 : 0) * 100
                  }%)`,
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
              <ViewToggleButton
                active={view === "grid"}
                label="Compact"
                icon={LeftToRightListBulletIcon}
                onClick={() => onViewChange("grid")}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ResultListHeader({ isDataset }: { isDataset: boolean }) {
  return (
    <div className="flex w-full items-center gap-3 px-4 pb-2 text-[11px] font-medium text-muted-foreground/55">
      <span className={LIST_COLS.model}>{isDataset ? "Dataset" : "Model"}</span>
      <span className={isDataset ? LIST_COLS.caps : LIST_COLS.capsModel}>
        {isDataset ? "Details" : "Capabilities"}
      </span>
      <span className={LIST_COLS.size}>Size</span>
      <span className={LIST_COLS.updated}>Updated</span>
      <span className={LIST_COLS.downloads}>Downloads</span>
      <span className={LIST_COLS.likes}>Likes</span>
      <span className={LIST_COLS.actions} aria-hidden="true" />
    </div>
  );
}

const STATUS_DOT_CLASS = "inline-block size-[6px] shrink-0 rounded-full";

function TitleMarkers({
  format,
  gated,
  isPrivate,
  partial,
  unsupported,
  onDevice,
}: {
  format?: "gguf" | "checkpoint" | null;
  gated?: false | "auto" | "manual";
  isPrivate?: boolean;
  partial: boolean;
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
      {partial && (
        <span
          role="img"
          aria-label="Partial download"
          className={cn(STATUS_DOT_CLASS, "bg-status-warning")}
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

function StatItem({ icon, value }: { icon: IconSvgElement; value: string }) {
  return (
    <span className="flex shrink-0 items-center gap-1.5">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-3.5 shrink-0"
      />
      {value}
    </span>
  );
}

function CapabilitiesCell({
  row,
  taskLabel,
  onSelect,
}: {
  row: DiscoverRow;
  taskLabel: string | null;
  onSelect: (id: string) => void;
}) {
  const caps = row.capabilities;
  if (caps.length === 0 && !taskLabel) {
    return null;
  }
  const max = 3;
  const shown = caps.slice(0, max);
  const extra = caps.length - shown.length;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <div
          className="pointer-events-auto flex w-full min-w-0 cursor-pointer items-center gap-1.5"
          onClick={(event) => {
            event.stopPropagation();
            onSelect(row.id);
          }}
        >
          {shown.map((capability) => (
            <CapabilityPill
              key={capability.key}
              capability={capability}
              iconOnly={true}
              tooltip={false}
            />
          ))}
          {extra > 0 && <span className="hub-chip shrink-0">+{extra}</span>}
          {shown.length === 0 && taskLabel && (
            <span className="truncate text-[12px] text-muted-foreground/75">
              {taskLabel}
            </span>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" align="start" className="tooltip-compact">
        <div className="flex flex-col items-start gap-1">
          {taskLabel && (
            <span className="text-[11px] font-medium text-muted-foreground">
              {taskLabel}
            </span>
          )}
          {caps.map((capability) => (
            <CapabilityPill key={capability.key} capability={capability} />
          ))}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function RowActions({
  row,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
  const hfUrl = `https://huggingface.co/${isDataset ? "datasets/" : ""}${row.result.id}`;
  const actionClass =
    "pointer-events-auto inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:bg-foreground/[0.07] hover:text-foreground focus-visible:text-foreground data-[state=open]:bg-foreground/[0.07] data-[state=open]:text-foreground";
  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={`Get ${row.repo}`}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(row.id);
            }}
            className={actionClass}
          >
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-4"
            />
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" className="tooltip-compact">
          {isDataset ? "View dataset" : "View and download"}
        </TooltipContent>
      </Tooltip>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            aria-label={`More options for ${row.repo}`}
            onClick={(event) => event.stopPropagation()}
            className={actionClass}
          >
            <HugeiconsIcon
              icon={MoreVerticalIcon}
              strokeWidth={1.75}
              className="size-4"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-[184px]">
          <DropdownMenuItem
            onClick={async (event) => {
              event.stopPropagation();
              await copyToClipboard(row.result.id);
            }}
          >
            <HugeiconsIcon
              icon={Copy01Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            Copy ID
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              if (!confirmExternalLink(hfUrl)) {
                window.open(hfUrl, "_blank", "noopener,noreferrer");
              }
            }}
          >
            <HugeiconsIcon
              icon={LinkSquare02Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            Open on Hugging Face
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}

function useResultRowModel(
  row: DiscoverRow,
  deviceType: string | null,
  isDataset: boolean,
) {
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
          }),
    [isDataset, row.id, row.result, deviceType],
  );
  const sizeLabel = formatModelParamLabel(row.repo, row.result.totalParams);
  const taskLabel = isDataset ? null : formatPipelineTag(row.result.pipelineTag);
  const unsupported = support?.status === "unsupported";
  return {
    support,
    unsupported,
    partial: row.isAvailableOnDevice && row.isPartialOnDevice,
    onDevice: row.isAvailableOnDevice && !row.isPartialOnDevice,
    sizeLabel: sizeLabel !== "N/A" ? sizeLabel : null,
    taskLabel,
  };
}

export const ResultCard = memo(function ResultCard({
  row,
  deviceType,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
  const { support, unsupported, partial, onDevice, sizeLabel, taskLabel } =
    useResultRowModel(row, deviceType, isDataset);
  const format = isDataset ? null : row.result.isGguf ? "gguf" : "checkpoint";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
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
          Updated {formatRelativeLong(row.result.updatedAt)}
        </span>
      ),
    });
  }

  const card = (
    <button
      type="button"
      aria-label={row.repo}
      onClick={() => onSelect(row.id)}
      className="hub-result-row hub-result-card group/row flex h-full w-full cursor-pointer items-center gap-3.5 px-4 text-left outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
    >
      <OwnerAvatar
        owner={row.owner}
        repoName={row.repo}
        className="size-[52px] shrink-0 rounded-[16px] text-[16px] ring-1 ring-black/5 dark:ring-white/10"
        remote={false}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-[15px] font-semibold leading-[18px] text-foreground">
            {row.repo}
          </span>
          <TitleMarkers
            format={format}
            gated={row.result.gated}
            isPrivate={row.result.private}
            partial={partial}
            unsupported={unsupported}
            onDevice={onDevice}
          />
        </div>
        <span className="flex min-w-0 items-center gap-1 text-[12.5px] leading-[16px] text-muted-foreground/80">
          <VerifiedOwner owner={row.owner} />
        </span>
        <div className="flex min-w-0 items-center gap-2 overflow-hidden text-[11.5px] leading-[16px] tabular-nums text-muted-foreground/65">
          {textParts.map((part, index) => (
            <Fragment key={part.key}>
              {index > 0 && (
                <span
                  aria-hidden="true"
                  className="shrink-0 text-muted-foreground/35"
                >
                  •
                </span>
              )}
              {part.node}
            </Fragment>
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
      <TooltipContent side="top" className="tooltip-compact max-w-[260px]">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
});

export const ResultGridRow = memo(function ResultGridRow({
  row,
  deviceType,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
  const { support, unsupported, partial, onDevice, sizeLabel, taskLabel } =
    useResultRowModel(row, deviceType, isDataset);
  const sizeDisplay = isDataset ? null : sizeLabel;
  const format = isDataset ? null : row.result.isGguf ? "gguf" : "checkpoint";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  const overlayButton = (
    <button
      type="button"
      aria-label={row.repo}
      onClick={() => onSelect(row.id)}
      className="absolute inset-0 z-0 cursor-pointer rounded-[inherit] outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
    />
  );

  return (
    <div className="hub-result-row group/row relative flex h-full w-full items-center gap-3 px-4">
      {tip ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>{overlayButton}</TooltipTrigger>
          <TooltipContent
            side="top"
            align="start"
            className="tooltip-compact max-w-[280px]"
          >
            {tip}
          </TooltipContent>
        </Tooltip>
      ) : (
        overlayButton
      )}
      <div className="pointer-events-none relative z-[1] flex w-full items-center gap-3">
        <div className={LIST_COLS.model}>
          <OwnerAvatar
            owner={row.owner}
            repoName={row.repo}
            className="size-9 shrink-0 rounded-[12px] text-[13px] ring-1 ring-black/5 dark:ring-white/10"
            remote={false}
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-[13.5px] font-semibold leading-[17px] text-foreground">
                {row.repo}
              </span>
              <TitleMarkers
                format={format}
                gated={row.result.gated}
                isPrivate={row.result.private}
                partial={partial}
                unsupported={unsupported}
                onDevice={onDevice}
              />
            </div>
            <span className="mt-0.5 flex min-w-0 items-center gap-1 text-[11.5px] leading-[15px] text-muted-foreground/80">
              <VerifiedOwner owner={row.owner} />
            </span>
          </div>
        </div>
        <div className={isDataset ? LIST_COLS.caps : LIST_COLS.capsModel}>
          {isDataset ? (
            row.summary ? (
              <span className="truncate text-[12px] text-muted-foreground/75">
                {row.summary}
              </span>
            ) : null
          ) : (
            <CapabilitiesCell
              row={row}
              taskLabel={taskLabel}
              onSelect={onSelect}
            />
          )}
        </div>
        <div
          className={cn(
            LIST_COLS.size,
            "truncate text-[12px] tabular-nums text-muted-foreground",
          )}
        >
          {sizeDisplay ?? "—"}
        </div>
        <div
          className={cn(
            LIST_COLS.updated,
            "truncate text-[12px] tabular-nums text-muted-foreground",
          )}
        >
          {formatRelativeShort(row.result.updatedAt)}
        </div>
        <div
          className={cn(
            LIST_COLS.downloads,
            "text-[12px] tabular-nums text-muted-foreground",
          )}
        >
          <StatItem
            icon={Download01Icon}
            value={formatCompact(row.result.downloads)}
          />
        </div>
        <div
          className={cn(
            LIST_COLS.likes,
            "text-[12px] tabular-nums text-muted-foreground",
          )}
        >
          <StatItem
            icon={FavouriteIcon}
            value={formatCompact(row.result.likes)}
          />
        </div>
        <div className={LIST_COLS.actions}>
          <RowActions row={row} isDataset={isDataset} onSelect={onSelect} />
        </div>
      </div>
    </div>
  );
});

// Compact master-pane row for split view: avatar + name/owner left, stats right,
// with a selected highlight for the model shown in the detail pane.
export const ResultSplitRow = memo(function ResultSplitRow({
  row,
  deviceType,
  isDataset,
  selected,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const { support, unsupported, partial, onDevice } = useResultRowModel(
    row,
    deviceType,
    isDataset,
  );
  const format = isDataset ? null : row.result.isGguf ? "gguf" : "checkpoint";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
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
      className="group/row flex h-full w-full cursor-pointer items-center gap-2.5 rounded-[12px] px-2.5 text-left outline-none transition-colors hover:bg-foreground/[0.04] data-[selected]:bg-foreground/[0.07] focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset dark:hover:bg-white/[0.05] dark:data-[selected]:bg-white/[0.08]"
    >
      <OwnerAvatar
        owner={row.owner}
        repoName={row.repo}
        className="size-8 shrink-0 rounded-[9px] text-[12px]"
        remote={false}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-[12.5px] font-semibold leading-[16px] text-foreground">
            {row.repo}
          </span>
          <TitleMarkers
            format={format}
            gated={row.result.gated}
            isPrivate={row.result.private}
            partial={partial}
            unsupported={unsupported}
            onDevice={onDevice}
          />
        </div>
        <span className="mt-0.5 flex min-w-0 items-center gap-1 text-[10.5px] leading-[14px] text-muted-foreground/80">
          <VerifiedOwner owner={row.owner} />
        </span>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5 text-[10.5px] tabular-nums text-muted-foreground/70">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1">
            <HugeiconsIcon
              icon={FavouriteIcon}
              strokeWidth={1.75}
              className="size-[11px] shrink-0"
            />
            {formatCompact(row.result.likes)}
          </span>
          <span className="inline-flex items-center gap-1">
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-[11px] shrink-0"
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
      <TooltipContent side="top" className="tooltip-compact max-w-[260px]">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
});
