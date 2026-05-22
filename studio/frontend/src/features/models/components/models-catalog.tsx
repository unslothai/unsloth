// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn, formatCompact } from "@/lib/utils";
import { usePlatformStore } from "@/config/env";
import { classifyUnslothSupport } from "@/hooks";
import {
  CloudOffIcon,
  CubeIcon,
  Download01Icon,
  DownloadCircle02Icon,
  FavouriteIcon,
  FilterIcon,
  FolderSearchIcon,
  PackageIcon,
  RefreshIcon,
  WifiDisconnected02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  type RefObject,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ModelDeleteAction } from "@/components/assistant-ui/model-selector/model-delete-action";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  deleteCachedModel,
  invalidateGgufVariantsCache,
  listGgufVariants,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { deleteCachedDataset } from "@/features/training/api/datasets-api";
import { OwnerAvatar } from "./owner-avatar";
import { formatBytes, formatRelativeShort } from "../lib/format";
import { inventoryRowMatches } from "../lib/inventory-search";
import { formatLocalUpdated } from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  ModelsTab,
} from "../types";

/**
 * Cached GGUF size chip. Same look as a StatChip but lazily fetches the
 * per-variant breakdown when the user hovers/opens it. The list of files
 * appears in a `rich` tooltip so users can see exactly which quantizations
 * make up the total size shown on the chip.
 */
function CachedSizeChip({
  repoId,
  totalBytes,
  isGguf,
  isDataset = false,
}: {
  repoId: string;
  totalBytes: number;
  isGguf: boolean;
  isDataset?: boolean;
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const fetchedForRef = useRef<string | null>(null);
  const needsVariantFetch = isGguf && !isDataset;
  const hfToken = useHfTokenStore((s) => s.token) || undefined;

  useEffect(() => {
    setVariants(null);
    fetchedForRef.current = null;
  }, [repoId]);

  const ensureVariantsLoaded = useCallback(() => {
    if (!needsVariantFetch) return;
    if (fetchedForRef.current === repoId) return;
    fetchedForRef.current = repoId;
    listGgufVariants(repoId, hfToken)
      .then((res) => {
        if (fetchedForRef.current !== repoId) return;
        setVariants(res.variants.filter((v) => v.downloaded));
      })
      .catch(() => {
        if (fetchedForRef.current !== repoId) return;
        setVariants([]);
      });
  }, [needsVariantFetch, repoId, hfToken]);

  const trigger = (
    <span onClick={(e) => e.stopPropagation()}>
      <StatChip icon={PackageIcon} value={formatBytes(totalBytes)} />
    </span>
  );

  const rows: Array<{ filename: string; size_bytes: number }> =
    needsVariantFetch && variants && variants.length > 0
      ? variants
      : [{ filename: repoId, size_bytes: totalBytes }];

  return (
    <Tooltip
      onOpenChange={(open) => {
        if (open) ensureVariantsLoaded();
      }}
    >
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      <TooltipContent variant="default" side="top" sideOffset={4}>
        <ul className="flex flex-col gap-1">
          {rows.map((row) => (
            <li
              key={row.filename}
              className="flex items-center gap-3 tabular-nums"
            >
              <span className="min-w-0 truncate">{row.filename}</span>
              <span className="ml-auto">
                <StatChip
                  icon={PackageIcon}
                  value={formatBytes(row.size_bytes)}
                />
              </span>
            </li>
          ))}
        </ul>
      </TooltipContent>
    </Tooltip>
  );
}

function StatChip({
  icon,
  value,
}: {
  icon: IconSvgElement;
  value: string;
}) {
  return (
    <span className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[10px] font-medium leading-none tabular-nums text-muted-foreground/75">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-2.5 shrink-0"
      />
      {value}
    </span>
  );
}

function CatalogRow({
  selected,
  active,
  onClick,
  tooltip,
  children,
}: {
  selected: boolean;
  active?: boolean;
  tooltip?: ReactNode;
  onClick: () => void;
  children: ReactNode;
}) {
  const button = (
    <div
      role="button"
      tabIndex={0}
      data-selected={selected || undefined}
      data-active={active || undefined}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className="catalog-row group/row relative block w-full cursor-pointer select-none overflow-hidden rounded-[14px] pl-3 pr-2.5 py-2.5 text-left"
    >
      {children}
    </div>
  );
  if (!tooltip) return button;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent
        side="right"
        align="center"
        sideOffset={8}
        className="tooltip-compact max-w-xs"
      >
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function StatusDot({
  tone,
  label,
}: {
  tone: "warning" | "danger" | "success";
  label: string;
}) {
  const toneClass =
    tone === "warning"
      ? "bg-status-warning"
      : tone === "danger"
        ? "bg-status-danger"
        : "bg-status-success";
  return (
    <span
      role="img"
      aria-label={label}
      className={cn(
        "inline-block size-[5px] shrink-0 rounded-full",
        toneClass,
      )}
    />
  );
}

function TooltipLegendRow({
  toneClass,
  children,
}: {
  toneClass: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-1.5 leading-snug">
      <span
        aria-hidden="true"
        className={cn("mt-1 inline-block size-1.5 shrink-0 rounded-full", toneClass)}
      />
      <span className="min-w-0">{children}</span>
    </div>
  );
}

function buildRowStatusTooltip({
  isGguf,
  isAvailableOnDevice,
  partialRepoId,
  unsupported,
  unsupportedReason,
  resourceLabel = "model",
}: {
  isGguf?: boolean;
  isAvailableOnDevice?: boolean;
  partialRepoId?: string;
  unsupported?: boolean;
  unsupportedReason?: string | null;
  resourceLabel?: "model" | "dataset";
}): ReactNode {
  const lines: ReactNode[] = [];

  if (isGguf) {
    lines.push(
      <TooltipLegendRow key="gguf" toneClass="bg-format-gguf">
        GGUF format.
      </TooltipLegendRow>,
    );
  }

  if (partialRepoId) {
    lines.push(
      <TooltipLegendRow key="partial" toneClass="bg-status-warning">
        Partial download of{" "}
        <span className="font-medium">{partialRepoId}</span>. Click Resume to
        continue.
      </TooltipLegendRow>,
    );
  } else if (isAvailableOnDevice) {
    lines.push(
      <TooltipLegendRow key="success" toneClass="bg-status-success">
        On device. Ready to use locally.
      </TooltipLegendRow>,
    );
  }

  if (unsupported) {
    lines.push(
      <TooltipLegendRow key="danger" toneClass="bg-status-danger">
        <span className="block">This {resourceLabel} may not be supported yet.</span>
        {unsupportedReason && (
          <span className="mt-0.5 block text-white/75">
            {unsupportedReason}
          </span>
        )}
        <span className="mt-0.5 block text-white/75">
          Still downloadable to your Hugging Face cache, shared with every
          framework that reads it.
        </span>
      </TooltipLegendRow>,
    );
  }

  if (lines.length === 0) return null;
  return <div className="space-y-1.5">{lines}</div>;
}

function NetworkErrorState({
  online,
  message,
  onRetry,
}: {
  online: boolean;
  message: string;
  onRetry: () => void;
}) {
  const title = online ? "Couldn't reach Hugging Face" : "You're offline";
  const body = online
    ? "The discovery feed couldn't load. Check your connection or try again."
    : "Reconnect to the internet to browse models from Hugging Face.";
  const icon = online ? CloudOffIcon : WifiDisconnected02Icon;

  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-amber-500/10 text-amber-700 dark:text-amber-300">
        <HugeiconsIcon icon={icon} strokeWidth={1.6} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          {body}
        </p>
        <p className="text-[11px] text-muted-foreground/70">{message}</p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
      >
        <HugeiconsIcon
          icon={RefreshIcon}
          strokeWidth={1.75}
          className="size-3.5"
        />
        Try again
      </button>
    </div>
  );
}

function FilterStarvedState({
  scannedCount,
  hasActiveFilters,
  onKeepSearching,
  onClearFilters,
}: {
  scannedCount: number;
  hasActiveFilters: boolean;
  onKeepSearching: () => void;
  onClearFilters: () => void;
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={FilterIcon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          No matches yet
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          Scanned {scannedCount.toLocaleString()} results. None match the
          current filters. Loosen them, or keep searching for a deeper sweep.
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
        {hasActiveFilters && (
          <button
            type="button"
            onClick={onClearFilters}
            className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-foreground/[0.06] px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.1] dark:bg-white/[0.06] dark:hover:bg-white/[0.1]"
          >
            Clear filters
          </button>
        )}
        <button
          type="button"
          onClick={onKeepSearching}
          className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
        >
          <HugeiconsIcon
            icon={RefreshIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          Keep searching
        </button>
      </div>
    </div>
  );
}

function InventoryErrorState({
  isDataset,
  onRetry,
}: {
  isDataset: boolean;
  onRetry: () => void;
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-amber-500/10 text-amber-700 dark:text-amber-300">
        <HugeiconsIcon icon={CloudOffIcon} strokeWidth={1.6} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          Couldn't load your library
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          Something went wrong reading your downloaded{" "}
          {isDataset ? "datasets" : "models"}. Check that the backend is running
          and try again.
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
      >
        <HugeiconsIcon icon={RefreshIcon} strokeWidth={1.75} className="size-3.5" />
        Try again
      </button>
    </div>
  );
}

function EmptyState({
  title,
  body,
  icon = CubeIcon,
}: {
  title: string;
  body: string;
  icon?: IconSvgElement;
}) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={icon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}

function SkeletonRow() {
  return (
    <div className="flex items-center gap-3 px-3 py-2.5">
      <div className="size-8 shrink-0 animate-pulse rounded-[9px] bg-muted" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="h-[13px] w-1/2 animate-pulse rounded-full bg-muted" />
        <div className="h-[11px] w-3/4 animate-pulse rounded-full bg-muted/70" />
      </div>
    </div>
  );
}

function SkeletonList({ count = 6 }: { count?: number }) {
  return (
    <ul className="divide-y divide-border" aria-hidden="true">
      {Array.from({ length: count }).map((_, i) => (
        <li key={i}>
          <SkeletonRow />
        </li>
      ))}
    </ul>
  );
}

const DiscoverModelRow = memo(function DiscoverModelRow({
  row,
  selected,
  active,
  deviceType,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  selected: boolean;
  active: boolean;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
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
  const unsupported = support?.status === "unsupported";
  const handleClick = useCallback(
    () => onSelect(row.id),
    [onSelect, row.id],
  );
  const partialRepoId =
    row.isAvailableOnDevice && row.isPartialOnDevice
      ? row.result.id
      : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.result.isGguf,
    isAvailableOnDevice: row.isAvailableOnDevice,
    partialRepoId,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  return (
    <CatalogRow
      selected={selected}
      active={active}
      tooltip={tooltip}
      onClick={handleClick}
    >
      <div className="flex items-center gap-3">
        <OwnerAvatar
          owner={row.owner}
          repoName={row.repo}
          className="size-8 rounded-[9px]"
        />
        <div className="flex min-w-0 flex-1 flex-col gap-[3px]">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5 pr-2">
              <p className="truncate text-[12px] font-medium leading-[18px] tracking-[-0.005em] text-foreground">
                {row.repo}
              </p>
              {row.result.isGguf && (
                <span
                  role="img"
                  aria-label="GGUF"
                  className="inline-block size-[5px] shrink-0 rounded-full bg-format-gguf"
                />
              )}
              {unsupported && (
                <StatusDot tone="danger" label="May not be supported yet" />
              )}
              {row.isAvailableOnDevice && row.isPartialOnDevice && (
                <StatusDot tone="warning" label="Partial download" />
              )}
              {row.isAvailableOnDevice && !row.isPartialOnDevice && (
                <StatusDot tone="success" label="On device" />
              )}
            </div>
            <div className="flex shrink-0 items-center gap-2.5">
              <StatChip
                icon={FavouriteIcon}
                value={formatCompact(row.result.likes)}
              />
              <StatChip
                icon={Download01Icon}
                value={formatCompact(row.result.downloads)}
              />
            </div>
          </div>
          <div className="flex h-[16px] min-w-0 items-center justify-between gap-2 text-[11.5px] leading-[16px] text-muted-foreground/85">
            <span className="flex min-w-0 items-center gap-1">
              <span className="truncate">{row.owner}</span>
              {row.owner.toLowerCase() === "unsloth" && (
                <span
                  aria-label="Verified Unsloth"
                  className="verified-badge size-3.5 shrink-0 text-primary"
                />
              )}
            </span>
            <span className="shrink-0 text-[10.5px] tabular-nums">
              {formatRelativeShort(row.result.updatedAt)}
            </span>
          </div>
        </div>
      </div>
    </CatalogRow>
  );
});

function cachedRowActive(
  row: CachedInventoryRow,
  activeCheckpoint: string | null,
): boolean {
  return activeCheckpoint?.toLowerCase() === row.repoId.toLowerCase();
}

function localRowActive(
  row: LocalInventoryRow,
  activeCheckpoint: string | null,
): boolean {
  return activeCheckpoint?.toLowerCase() === row.id.toLowerCase();
}

const InventoryRow = memo(function InventoryRow({
  row,
  selected,
  activeCheckpoint,
  isDataset,
  dimmed,
  deviceType,
  onSelect,
  onChange,
}: {
  row: CachedInventoryRow | LocalInventoryRow;
  selected: boolean;
  activeCheckpoint: string | null;
  isDataset: boolean;
  dimmed: boolean;
  deviceType: string | null;
  onSelect: (id: string) => void;
  onChange?: () => void;
}) {
  const unsupported = useMemo(() => {
    if (isDataset) return false;
    if (row.kind === "cache") {
      return (
        classifyUnslothSupport({
          modelId: row.repoId,
          pipelineTag: row.pipelineTag,
          tags: row.tags,
          libraryName: row.libraryName,
          quantMethod: row.quantMethod,
          deviceType,
        }).status === "unsupported"
      );
    }
    return (
      !!row.repoId &&
      classifyUnslothSupport({ modelId: row.repoId, deviceType }).status ===
        "unsupported"
    );
  }, [isDataset, row, deviceType]);
  const handleClick = useCallback(() => onSelect(row.id), [onSelect, row.id]);
  const active =
    row.kind === "cache"
      ? cachedRowActive(row, activeCheckpoint)
      : localRowActive(row, activeCheckpoint);
  const title = row.kind === "cache" ? row.repo : row.title;

  const subLabel = row.owner;
  const trailing =
    row.kind === "local" && row.updatedAt
      ? formatLocalUpdated(row.updatedAt)
      : null;
  const cacheDeletableRepoId =
    row.kind === "cache"
      ? row.repoId
      : row.source === "hf_cache" && row.repoId
        ? row.repoId
        : null;
  const canDelete = cacheDeletableRepoId !== null;
  const partialRepoId = row.partial
    ? row.kind === "cache"
      ? row.repoId
      : (row.repoId ?? row.id)
    : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.isGguf,
    isAvailableOnDevice: !partialRepoId,
    partialRepoId,
    unsupported,
    resourceLabel: isDataset ? "dataset" : "model",
  });

  return (
    <CatalogRow
      selected={selected}
      active={active}
      tooltip={tooltip}
      onClick={handleClick}
    >
      <div
        className={cn(
          "group/inventory flex items-center gap-3 transition-opacity",
          dimmed && "opacity-25 hover:opacity-60",
        )}
      >
        <OwnerAvatar
          owner={row.owner}
          repoName={title}
          className="size-8 rounded-[9px]"
        />
        <div className="flex min-w-0 flex-1 flex-col gap-[3px]">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5 pr-2">
              <p className="truncate text-[12px] font-medium leading-[18px] tracking-[-0.005em] text-foreground">
                {title}
              </p>
              {row.isGguf && (
                <span
                  role="img"
                  aria-label="GGUF"
                  className="inline-block size-[5px] shrink-0 rounded-full bg-format-gguf"
                />
              )}
              {partialRepoId ? (
                <StatusDot tone="warning" label="Partial download" />
              ) : (
                <StatusDot tone="success" label="On device" />
              )}
              {unsupported && (
                <StatusDot tone="danger" label="May not be supported yet" />
              )}
            </div>
            <div className="flex shrink-0 items-center gap-1">
              {canDelete && cacheDeletableRepoId && (
                <ModelDeleteAction
                  ariaLabel={`Delete ${cacheDeletableRepoId}`}
                  title={
                    isDataset ? "Delete cached dataset?" : "Delete cached model?"
                  }
                  description={
                    <>
                      This will remove{" "}
                      <span className="font-medium text-foreground">
                        {cacheDeletableRepoId}
                      </span>{" "}
                      {isDataset
                        ? "and its downloaded files"
                        : row.isGguf
                          ? "and all of its downloaded quantizations"
                          : "and all of its downloaded files"}
                      {row.kind === "cache"
                        ? ` (${formatBytes(row.bytes)})`
                        : ""}{" "}
                      from disk. You can re-download it later.
                    </>
                  }
                  successMessage={`Deleted ${cacheDeletableRepoId}`}
                  buttonClassName="opacity-0 transition-opacity group-hover/inventory:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
                  iconClassName="size-3.5"
                  onConfirm={async () => {
                    if (isDataset) {
                      await deleteCachedDataset(cacheDeletableRepoId);
                    } else {
                      await deleteCachedModel(cacheDeletableRepoId);
                      invalidateGgufVariantsCache(cacheDeletableRepoId);
                    }
                  }}
                  onDeleted={onChange}
                />
              )}
              {row.kind === "cache" && (
                <CachedSizeChip
                  repoId={row.repoId}
                  totalBytes={row.bytes}
                  isGguf={row.isGguf}
                  isDataset={isDataset}
                />
              )}
            </div>
          </div>
          <div className="flex h-[16px] min-w-0 items-center justify-between gap-2 text-[11.5px] leading-[16px] text-muted-foreground/85">
            <span className="flex min-w-0 items-center gap-1">
              <span className="truncate">{subLabel}</span>
              {subLabel.toLowerCase() === "unsloth" && (
                <span
                  aria-label="Verified Unsloth"
                  className="verified-badge size-3.5 shrink-0 text-primary"
                />
              )}
            </span>
            {trailing && (
              <span className="shrink-0 text-[10.5px] tabular-nums">{trailing}</span>
            )}
          </div>
        </div>
      </div>
    </CatalogRow>
  );
});

// Approximate row height; measureElement corrects per-row variance after paint.
const ROW_ESTIMATE_PX = 56;

// Windowed list: only rows near the viewport mount, so avatars and tooltips
// render only for visible rows however many results stream in.
function VirtualRows<T>({
  items,
  scrollRef,
  getKey,
  renderRow,
}: {
  items: readonly T[];
  scrollRef: RefObject<HTMLDivElement | null>;
  getKey: (item: T, index: number) => string;
  renderRow: (item: T) => ReactNode;
}) {
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_ESTIMATE_PX,
    overscan: 10,
    getItemKey: (index) => getKey(items[index], index),
  });

  return (
    <ul
      style={{
        height: virtualizer.getTotalSize(),
        position: "relative",
        width: "100%",
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => (
        <li
          key={virtualRow.key}
          data-index={virtualRow.index}
          ref={virtualizer.measureElement}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            transform: `translateY(${virtualRow.start}px)`,
          }}
        >
          {renderRow(items[virtualRow.index])}
        </li>
      ))}
    </ul>
  );
}

type InventoryItem =
  | { variant: "cached"; row: CachedInventoryRow }
  | { variant: "local"; row: LocalInventoryRow };

export function ModelsCatalog({
  tab,
  discoverRows,
  cachedRows,
  localRows,
  selectedId,
  onSelect,
  isLoading,
  isLoadingMore,
  downloadedReady,
  inventoryError,
  query,
  scrollRef,
  sentinelRef,
  activeCheckpoint,
  searchError,
  online,
  isDataset,
  inventoryTokens,
  filterPaused,
  scannedCount,
  hasActiveFilters,
  onKeepSearching,
  onClearFilters,
  onRetry,
  onInventoryChange,
}: {
  tab: ModelsTab;
  discoverRows: DiscoverRow[];
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  isLoading: boolean;
  isLoadingMore: boolean;
  downloadedReady: boolean;
  inventoryError: boolean;
  query: string;
  scrollRef: RefObject<HTMLDivElement | null>;
  sentinelRef: RefObject<HTMLDivElement | null>;
  activeCheckpoint: string | null;
  searchError: string | null;
  online: boolean;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  filterPaused: boolean;
  scannedCount: number;
  hasActiveFilters: boolean;
  onKeepSearching: () => void;
  onClearFilters: () => void;
  onRetry: () => void;
  onInventoryChange?: (hint?: import("./download-section").InventoryHint) => void;
}) {
  const [scrolled, setScrolled] = useState(false);
  const deviceType = usePlatformStore((s) => s.deviceType);

  // One scroll container, so cached + local share a single virtual window.
  const inventoryItems = useMemo<InventoryItem[]>(
    () => [
      ...cachedRows.map((row) => ({ variant: "cached" as const, row })),
      ...localRows.map((row) => ({ variant: "local" as const, row })),
    ],
    [cachedRows, localRows],
  );
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setScrolled(el.scrollTop > 0);
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [scrollRef]);

  // Active fetches happen in short bursts (a few hundred ms) followed by a
  // ~500ms throttle gap before the next auto-paginated batch. Without a grace
  // window the loading bar would flicker on and off; we keep it lit for an
  // extra moment so streaming sessions feel continuous.
  //
  // React 18 also batches state updates across async boundaries, so a fast
  // fetchMore can flip isLoadingMore true→false in a single commit. We don't
  // rely on observing that flag alone. Any change to the loading flags or
  // the row count is treated as "activity" and re-lights the bar.
  const [streamingActive, setStreamingActive] = useState(false);
  useEffect(() => {
    if (tab !== "discover") {
      setStreamingActive(false);
      return;
    }
    setStreamingActive(true);
    const timer = window.setTimeout(() => setStreamingActive(false), 1500);
    return () => window.clearTimeout(timer);
  }, [tab, isLoading, isLoadingMore, discoverRows.length]);

  const showDiscoverLoading = tab === "discover" && streamingActive;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      <div
        aria-hidden="true"
        className={cn(
          "mx-5 shrink-0 border-t transition-colors",
          scrolled ? "border-border" : "border-transparent",
        )}
      />
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto pb-6 pl-5 pr-3 pt-0 [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
      {tab === "discover" ? (
        <>
          {searchError && discoverRows.length === 0 ? (
            <NetworkErrorState
              online={online}
              message={searchError}
              onRetry={onRetry}
            />
          ) : filterPaused && discoverRows.length === 0 ? (
            <FilterStarvedState
              scannedCount={scannedCount}
              hasActiveFilters={hasActiveFilters}
              onKeepSearching={onKeepSearching}
              onClearFilters={onClearFilters}
            />
          ) : discoverRows.length === 0 && isLoading ? (
            <SkeletonList />
          ) : discoverRows.length === 0 ? (
            <EmptyState
              icon={query.trim() ? FolderSearchIcon : CubeIcon}
              title={
                query.trim() ? "No matching models" : "No models available"
              }
              body={
                query.trim()
                  ? "Try a broader search or remove some filters."
                  : "The current filters are excluding every result."
              }
            />
          ) : (
            <VirtualRows
              items={discoverRows}
              scrollRef={scrollRef}
              getKey={(row) => row.id}
              renderRow={(row) => (
                <DiscoverModelRow
                  row={row}
                  selected={selectedId === row.id}
                  active={
                    activeCheckpoint?.toLowerCase() === row.id.toLowerCase()
                  }
                  deviceType={deviceType}
                  isDataset={isDataset}
                  onSelect={onSelect}
                />
              )}
            />
          )}

          <div ref={sentinelRef} className="h-px" />
        </>
      ) : !downloadedReady ? (
        <div className="flex min-h-[240px] items-center justify-center gap-3 text-[13px] text-muted-foreground">
          <Spinner className="size-4" />
          Loading local inventory…
        </div>
      ) : inventoryError &&
        cachedRows.length === 0 &&
        localRows.length === 0 ? (
        <InventoryErrorState
          isDataset={isDataset}
          onRetry={() => onInventoryChange?.()}
        />
      ) : cachedRows.length === 0 && localRows.length === 0 ? (
        <EmptyState
          icon={query.trim() ? FolderSearchIcon : DownloadCircle02Icon}
          title={
            query.trim() ? "No matches on device" : "Nothing on device yet"
          }
          body={
            query.trim()
              ? "Clear the search or try a different query. No cached or local model matches it."
              : "Downloaded repositories and indexed local folders will appear here."
          }
        />
      ) : (
        <VirtualRows
          items={inventoryItems}
          scrollRef={scrollRef}
          getKey={(item) => `${item.variant}-${item.row.id}`}
          renderRow={(item) => (
            <InventoryRow
              row={item.row}
              selected={selectedId === item.row.id}
              activeCheckpoint={activeCheckpoint}
              isDataset={isDataset}
              dimmed={!inventoryRowMatches(item.row, inventoryTokens)}
              deviceType={deviceType}
              onSelect={onSelect}
              onChange={onInventoryChange}
            />
          )}
        />
      )}
      </div>
      <div
        role="status"
        aria-live="polite"
        aria-label={showDiscoverLoading ? "Loading models" : undefined}
        className={cn(
          "shrink-0 transition-opacity duration-150",
          showDiscoverLoading ? "opacity-100" : "opacity-0",
        )}
      >
        <div className="hub-loading-bar" />
      </div>
    </div>
  );
}
