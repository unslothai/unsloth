// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelDeleteAction } from "@/components/assistant-ui/model-selector/model-delete-action";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type GgufVariantDetail,
  deleteCachedModel,
  deleteCachedDataset,
  formatLocalUpdated,
  invalidateGgufVariantsCache,
  listGgufVariants,
  useGgufVariantsCacheVersion,
} from "@/features/inventory";
import { classifyUnslothSupport } from "@/hooks";
import { formatBytes, formatRelativeShort } from "@/lib/format";
import { modelIdsMatch } from "@/lib/model-identity";
import { cn, formatCompact } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  Download01Icon,
  FavouriteIcon,
  PackageIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ReactNode,
  type RefObject,
  memo,
  useCallback,
  useMemo,
  useRef,
  useState,
} from "react";
import { notifyInventoryEntryDeleted } from "../delete-notifications";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../types";
import { OwnerAvatar } from "./owner-avatar";

function CachedSizeChip({
  repoId,
  totalBytes,
  isGguf,
  isDataset = false,
  cachePath,
}: {
  repoId: string;
  totalBytes: number;
  isGguf: boolean;
  isDataset?: boolean;
  cachePath?: string | null;
}) {
  const hfToken = useHfTokenStore((s) => s.token) || undefined;
  const variantsVersion = useGgufVariantsCacheVersion(repoId);
  const fetchKey = `${repoId}::${cachePath ?? ""}::${variantsVersion}`;
  const [variantState, setVariantState] = useState<{
    key: string;
    status: "idle" | "loading" | "loaded" | "error";
    variants: GgufVariantDetail[];
  }>(() => ({
    key: fetchKey,
    status: "idle",
    variants: [],
  }));
  const fetchedForRef = useRef<string | null>(null);
  const needsVariantFetch = isGguf && !isDataset;
  const currentVariantState =
    variantState.key === fetchKey
      ? variantState
      : { key: fetchKey, status: "idle" as const, variants: [] };

  const ensureVariantsLoaded = useCallback(() => {
    if (!needsVariantFetch) return;
    if (fetchedForRef.current === fetchKey) return;
    fetchedForRef.current = fetchKey;
    setVariantState({ key: fetchKey, status: "loading", variants: [] });
    listGgufVariants(repoId, hfToken, {
      preferLocalCache: true,
      localPath: cachePath ?? null,
    })
      .then((res) => {
        if (fetchedForRef.current !== fetchKey) return;
        setVariantState({
          key: fetchKey,
          status: "loaded",
          variants: res.variants.filter((v) => v.downloaded),
        });
      })
      .catch(() => {
        if (fetchedForRef.current !== fetchKey) return;
        fetchedForRef.current = null;
        setVariantState({ key: fetchKey, status: "error", variants: [] });
      });
  }, [needsVariantFetch, fetchKey, repoId, hfToken, cachePath]);

  const trigger = (
    <span onClick={(e) => e.stopPropagation()}>
      <StatChip icon={PackageIcon} value={formatBytes(totalBytes)} />
    </span>
  );

  const rows: Array<{ filename: string; size_bytes: number }> | null =
    !needsVariantFetch
      ? [{ filename: repoId, size_bytes: totalBytes }]
      : currentVariantState.status === "loaded" &&
          currentVariantState.variants.length > 0
        ? currentVariantState.variants
        : null;
  const variantMessage =
    currentVariantState.status === "loading"
      ? "Loading downloaded variants..."
      : currentVariantState.status === "error"
        ? "Couldn't load downloaded variants. Open again to retry."
        : currentVariantState.status === "loaded"
          ? "No downloaded GGUF variants found."
          : "Open to load downloaded variants.";

  return (
    <Tooltip
      onOpenChange={(open) => {
        if (open) ensureVariantsLoaded();
      }}
    >
      <TooltipTrigger asChild={true}>{trigger}</TooltipTrigger>
      <TooltipContent variant="default" side="top" sideOffset={4}>
        {rows ? (
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
        ) : (
          <span className="block max-w-52 text-[11px] leading-4 text-muted-foreground">
            {variantMessage}
          </span>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

export function StatChip({
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
      <TooltipTrigger asChild={true}>{button}</TooltipTrigger>
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
      className={cn("inline-block size-[5px] shrink-0 rounded-full", toneClass)}
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
        className={cn(
          "mt-1 inline-block size-1.5 shrink-0 rounded-full",
          toneClass,
        )}
      />
      <span className="min-w-0">{children}</span>
    </div>
  );
}

function buildRowStatusTooltip({
  isGguf,
  isAdapter,
  isAvailableOnDevice,
  partialRepoId,
  unsupported,
  unsupportedReason,
  resourceLabel = "model",
}: {
  isGguf?: boolean;
  isAdapter?: boolean;
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
  if (isAdapter) {
    lines.push(
      <TooltipLegendRow key="adapter" toneClass="bg-format-adapter">
        Adapter format.
      </TooltipLegendRow>,
    );
  }

  if (partialRepoId) {
    lines.push(
      <TooltipLegendRow key="partial" toneClass="bg-status-warning">
        Partial download of <span className="font-medium">{partialRepoId}</span>
        . Click Resume to continue.
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
        <span className="block">
          This {resourceLabel} may not be supported yet.
        </span>
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

export const DiscoverModelRow = memo(function DiscoverModelRow({
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
  const handleClick = useCallback(() => onSelect(row.id), [onSelect, row.id]);
  const partialRepoId =
    row.isAvailableOnDevice && row.isPartialOnDevice
      ? row.result.id
      : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.result.isGguf,
    isAdapter: false,
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
  activeGgufVariant: string | null,
): boolean {
  if (!modelIdsMatch(activeCheckpoint, row.loadId)) return false;
  if (row.modelFormat === "gguf") {
    return row.capabilities.requiresVariant ? activeGgufVariant !== null : true;
  }
  return activeGgufVariant === null;
}

function localRowActive(
  row: LocalInventoryRow,
  activeCheckpoint: string | null,
  activeGgufVariant: string | null,
): boolean {
  if (!modelIdsMatch(activeCheckpoint, row.loadId)) return false;
  if (row.modelFormat === "gguf") {
    return row.capabilities.requiresVariant ? activeGgufVariant !== null : true;
  }
  return activeGgufVariant === null;
}

export const InventoryRow = memo(function InventoryRow({
  row,
  selected,
  activeCheckpoint,
  activeGgufVariant,
  isDataset,
  dimmed,
  deviceType,
  onSelect,
  onChange,
}: {
  row: CachedInventoryRow | LocalInventoryRow;
  selected: boolean;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  isDataset: boolean;
  dimmed: boolean;
  deviceType: string | null;
  onSelect: (id: string) => void;
  onChange?: () => void;
}) {
  const rowModelId =
    row.kind === "cache"
      ? row.repoId
      : (row.repoId ?? row.baseModelHubId ?? row.baseModel ?? row.loadId);
  const rowTagsSignature = row.tags?.join("\u0001") ?? "";
  const unsupported = useMemo(() => {
    if (isDataset) return false;
    return (
      classifyUnslothSupport({
        modelId: rowModelId,
        pipelineTag: row.pipelineTag,
        tags: rowTagsSignature ? rowTagsSignature.split("\u0001") : undefined,
        libraryName: row.libraryName,
        quantMethod: row.quantMethod,
        deviceType,
      }).status === "unsupported"
    );
  }, [
    isDataset,
    rowModelId,
    row.pipelineTag,
    rowTagsSignature,
    row.libraryName,
    row.quantMethod,
    deviceType,
  ]);
  const handleClick = useCallback(() => onSelect(row.id), [onSelect, row.id]);
  const active =
    row.kind === "cache"
      ? cachedRowActive(row, activeCheckpoint, activeGgufVariant)
      : localRowActive(row, activeCheckpoint, activeGgufVariant);
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
      : (row.repoId ?? row.loadId)
    : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.isGguf,
    isAdapter: row.modelFormat === "adapter",
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
              {row.modelFormat === "adapter" && (
                <span
                  role="img"
                  aria-label="Adapter"
                  className="inline-block size-[5px] shrink-0 rounded-full bg-format-adapter"
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
                    isDataset
                      ? "Delete cached dataset?"
                      : "Delete cached model?"
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
                  buttonClassName="opacity-0 transition-opacity group-hover/inventory:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 [@media(pointer:coarse)]:opacity-100"
                  iconClassName="size-3.5"
                  onConfirm={async () => {
                    if (isDataset) {
                      await deleteCachedDataset(cacheDeletableRepoId);
                      notifyInventoryEntryDeleted({
                        kind: "dataset",
                        id: cacheDeletableRepoId,
                      });
                    } else {
                      await deleteCachedModel(cacheDeletableRepoId);
                      invalidateGgufVariantsCache(cacheDeletableRepoId);
                      notifyInventoryEntryDeleted({
                        kind: "model",
                        id: cacheDeletableRepoId,
                      });
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
                  cachePath={row.cachePath}
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
              <span className="shrink-0 text-[10.5px] tabular-nums">
                {trailing}
              </span>
            )}
          </div>
        </div>
      </div>
    </CatalogRow>
  );
});

const ROW_HEIGHT_PX = 57;

export function VirtualRows<T>({
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
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT_PX,
    overscan: 10,
    getItemKey: (index) => getKey(items[index], index),
  });

  return (
    <ul
      style={{
        height: virtualizer.getTotalSize(),
        position: "relative",
        width: "100%",
        overflowAnchor: "none",
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => (
        <li
          key={virtualRow.key}
          data-index={virtualRow.index}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            transform: `translateY(${virtualRow.start}px)`,
            // Lock every row to the same height as the virtualizer's estimate.
            // No measureElement ref: dynamic measurement on each row mount
            // triggers virtualizer state churn (re-renders + occasional offset
            // recompute) that the user perceives as a jump while new rows
            // arrive. With a fixed height that matches estimateSize exactly,
            // appending below the viewport can never shift visible items.
            height: `${ROW_HEIGHT_PX}px`,
            contain: "layout paint",
          }}
        >
          {renderRow(items[virtualRow.index])}
        </li>
      ))}
    </ul>
  );
}
