// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { pinKey, usePinnedModelsStore } from "@/features/model-picker";
import {
  CubeIcon,
  DownloadCircle02Icon,
  FolderSearchIcon,
  PinIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { RefObject } from "react";
import { useMemo } from "react";
import {
  inventoryRowMatches,
  scoreInventoryRow,
} from "../lib/inventory-search";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../types";
import {
  DiscoverFetchMoreFooter,
  DiscoverFetchMoreState,
  EmptyState,
  InventoryErrorState,
  NetworkErrorState,
  SkeletonList,
} from "./catalog-states";
import { InventoryRow, VirtualRows } from "./models-catalog-rows";
import {
  type AllModelsView,
  type InventorySort,
  RESULT_CARD_HEIGHT_PX,
  RESULT_GRID_HEIGHT_PX,
  RESULT_GRID_ROW_HEIGHT_PX,
  RESULT_ROW_HEIGHT_PX,
  RESULT_SPLIT_HEIGHT_PX,
  RESULT_SPLIT_ROW_HEIGHT_PX,
  ResultCard,
  ResultGridRow,
  ResultSplitRow,
} from "./models-table";

type InventoryItem =
  | { variant: "cached"; row: CachedInventoryRow }
  | { variant: "local"; row: LocalInventoryRow };

function inventoryItemTitle(item: InventoryItem): string {
  return item.variant === "cached" ? item.row.repo : item.row.title;
}

function inventoryItemSize(item: InventoryItem): number {
  return item.variant === "cached" ? item.row.bytes : 0;
}

export function InventoryWarningRow({
  isDataset,
  onRetry,
}: {
  isDataset: boolean;
  onRetry: () => void;
}) {
  return (
    <div className="mx-5 mt-2 rounded-[8px] border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[12.5px] text-muted-foreground">
      <div className="flex items-center justify-between gap-3">
        <span>
          Some on-device sources couldn't be scanned. Showing available{" "}
          {isDataset ? "datasets" : "models"}.
        </span>
        <button
          type="button"
          className="shrink-0 text-[12px] font-medium text-foreground transition-colors hover:text-primary"
          onClick={onRetry}
        >
          Retry
        </button>
      </div>
    </div>
  );
}

export function DiscoverList({
  discoverRows,
  onSelect,
  isLoading,
  query,
  scrollElement,
  scrollMargin = 0,
  suppressEmptyState = false,
  sentinelRef,
  searchError,
  online,
  isDataset,
  deviceType,
  scannedCount,
  isLoadingMore,
  hasMore,
  hasActiveFilters,
  onFetchMore,
  onClearFilters,
  onRetry,
  onSwitchDevice,
  view,
  selectedId,
}: {
  discoverRows: DiscoverRow[];
  onSelect: (id: string) => void;
  selectedId?: string | null;
  isLoading: boolean;
  query: string;
  scrollElement: HTMLDivElement | null;
  scrollMargin?: number;
  suppressEmptyState?: boolean;
  sentinelRef: RefObject<HTMLDivElement | null>;
  searchError: string | null;
  online: boolean;
  isDataset: boolean;
  deviceType: string | null;
  scannedCount: number;
  isLoadingMore: boolean;
  hasMore: boolean;
  hasActiveFilters: boolean;
  onFetchMore: () => void;
  onClearFilters: () => void;
  onRetry: () => void;
  onSwitchDevice?: () => void;
  view: AllModelsView;
}) {
  // "two" = two cards per row; "grid" = compact table rows; "split" = one card
  // per row in the master pane alongside an inline detail view.
  const isSplit = view === "split";
  const isCardLike = view === "two" || view === "split";
  const rowHeight = isSplit
    ? RESULT_SPLIT_ROW_HEIGHT_PX
    : isCardLike
      ? RESULT_ROW_HEIGHT_PX
      : RESULT_GRID_ROW_HEIGHT_PX;
  const cellHeight = isSplit
    ? RESULT_SPLIT_HEIGHT_PX
    : isCardLike
      ? RESULT_CARD_HEIGHT_PX
      : RESULT_GRID_HEIGHT_PX;
  const columns = view === "two" ? 2 : 1;

  return (
    <>
      {online ? (
        discoverRows.length > 0 ? (
          <>
            <VirtualRows
              items={discoverRows}
              scrollElement={scrollElement}
              scrollMargin={scrollMargin}
              columns={columns}
              rowHeight={rowHeight}
              cellHeight={cellHeight}
              getKey={(row) => row.id}
              renderRow={(row) =>
                view === "split" ? (
                  <ResultSplitRow
                    row={row}
                    deviceType={deviceType}
                    isDataset={isDataset}
                    selected={row.id === selectedId}
                    onSelect={onSelect}
                  />
                ) : isCardLike ? (
                  <ResultCard
                    row={row}
                    deviceType={deviceType}
                    isDataset={isDataset}
                    onSelect={onSelect}
                  />
                ) : (
                  <ResultGridRow
                    row={row}
                    deviceType={deviceType}
                    isDataset={isDataset}
                    onSelect={onSelect}
                  />
                )
              }
            />
            {hasMore && (
              <DiscoverFetchMoreFooter
                hasActiveFilters={hasActiveFilters}
                isLoadingMore={isLoadingMore}
                onFetchMore={onFetchMore}
              />
            )}
          </>
        ) : suppressEmptyState ? null : searchError ? (
          <NetworkErrorState
            online={online}
            message={searchError}
            onRetry={onRetry}
            resourceLabel={isDataset ? "datasets" : "models"}
          />
        ) : hasMore ? (
          <DiscoverFetchMoreState
            scannedCount={scannedCount}
            hasActiveFilters={hasActiveFilters}
            isLoadingMore={isLoadingMore}
            onFetchMore={onFetchMore}
            onClearFilters={onClearFilters}
          />
        ) : isLoading ? (
          <SkeletonList />
        ) : (
          <EmptyState
            icon={query.trim() ? FolderSearchIcon : CubeIcon}
            title={
              query.trim()
                ? `No matching ${isDataset ? "datasets" : "models"}`
                : `No ${isDataset ? "datasets" : "models"} available`
            }
            body={
              query.trim()
                ? "Try a broader search or remove some filters."
                : "The current filters are excluding every result."
            }
          />
        )
      ) : suppressEmptyState ? null : (
        <NetworkErrorState
          online={online}
          message="Discovery is unavailable while offline."
          onRetry={onRetry}
          onSwitchDevice={onSwitchDevice}
          resourceLabel={isDataset ? "datasets" : "models"}
        />
      )}

      <div ref={sentinelRef} className="h-px" />
    </>
  );
}

export function DownloadedList({
  cachedRows,
  localRows,
  selectedId,
  onSelect,
  downloadedReady,
  inventoryError,
  query,
  scrollElement,
  columns = 1,
  activeCheckpoint,
  activeGgufVariant,
  isDataset,
  inventoryTokens,
  deviceType,
  compact = false,
  sort,
  onInventoryChange,
}: {
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  downloadedReady: boolean;
  inventoryError: boolean;
  query: string;
  scrollElement: HTMLDivElement | null;
  columns?: number;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  deviceType: string | null;
  /** Narrow split master pane: render compact inventory rows. */
  compact?: boolean;
  sort: InventorySort;
  onInventoryChange?: () => void;
}) {
  // Pinned repos surface first regardless of the active sort; the chosen sort
  // still orders rows within the pinned and unpinned groups.
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const inventoryItems = useMemo<InventoryItem[]>(() => {
    const merged: InventoryItem[] = [
      ...cachedRows.map((row) => ({ variant: "cached" as const, row })),
      ...localRows.map((row) => ({ variant: "local" as const, row })),
    ];
    // Pinned rows order by pin recency (newest pin first), not the active
    // sort, so "Pin to top" puts the row exactly where the user expects.
    const pinIndex = new Map(pinnedIds.map((key, index) => [key, index]));
    const pinRank = (item: InventoryItem) =>
      (item.row.repoId ? pinIndex.get(pinKey(item.row.repoId)) : undefined) ??
      Number.MAX_SAFE_INTEGER;
    if (inventoryTokens.length > 0) {
      return merged
        .map((item, index) => ({
          item,
          index,
          score: scoreInventoryRow(item.row, inventoryTokens),
        }))
        .sort(
          (a, b) =>
            pinRank(a.item) - pinRank(b.item) ||
            b.score - a.score ||
            a.index - b.index,
        )
        .map((entry) => entry.item);
    }
    if (sort === "recent") {
      return merged
        .map((item, index) => ({ item, index }))
        .sort((a, b) => pinRank(a.item) - pinRank(b.item) || a.index - b.index)
        .map((entry) => entry.item);
    }
    return merged
      .map((item, index) => ({ item, index }))
      .sort(
        (a, b) =>
          pinRank(a.item) - pinRank(b.item) ||
          (sort === "name"
            ? inventoryItemTitle(a.item).localeCompare(
                inventoryItemTitle(b.item),
              ) || a.index - b.index
            : inventoryItemSize(b.item) - inventoryItemSize(a.item) ||
              a.index - b.index),
      )
      .map((entry) => entry.item);
  }, [cachedRows, localRows, inventoryTokens, sort, pinnedIds]);
  const hasInventoryRows = cachedRows.length > 0 || localRows.length > 0;
  // Pinned repos get their own labelled section so it's clear why they lead
  // the list; inventoryItems already sorts them first, so this is a prefix.
  const pinnedCount = useMemo(
    () =>
      inventoryItems.filter(
        (item) => item.row.repoId && pinnedSet.has(pinKey(item.row.repoId)),
      ).length,
    [inventoryItems, pinnedSet],
  );
  const pinnedItems = inventoryItems.slice(0, pinnedCount);
  const unpinnedItems = inventoryItems.slice(pinnedCount);
  const rowHeightPx = compact
    ? RESULT_SPLIT_ROW_HEIGHT_PX
    : RESULT_GRID_ROW_HEIGHT_PX;
  const cellHeightPx = compact ? RESULT_SPLIT_HEIGHT_PX : RESULT_GRID_HEIGHT_PX;
  const renderInventoryRow = (item: InventoryItem) => (
    <InventoryRow
      row={item.row}
      selected={selectedId === item.row.id}
      activeCheckpoint={activeCheckpoint}
      activeGgufVariant={activeGgufVariant}
      isDataset={isDataset}
      dimmed={!inventoryRowMatches(item.row, inventoryTokens)}
      deviceType={deviceType}
      compact={compact}
      onSelect={onSelect}
      onChange={onInventoryChange}
    />
  );

  if (!downloadedReady && !hasInventoryRows) {
    return (
      <div className="flex min-h-[240px] items-center justify-center gap-3 text-[13px] text-muted-foreground">
        <Spinner className="size-4" />
        Loading local inventory...
      </div>
    );
  }

  if (inventoryError && cachedRows.length === 0 && localRows.length === 0) {
    return (
      <InventoryErrorState
        isDataset={isDataset}
        onRetry={() => onInventoryChange?.()}
      />
    );
  }

  if (cachedRows.length === 0 && localRows.length === 0) {
    return (
      <EmptyState
        icon={query.trim() ? FolderSearchIcon : DownloadCircle02Icon}
        title={query.trim() ? "No matches on device" : "Nothing on device yet"}
        body={
          query.trim()
            ? `Clear the search or try a different query. No cached or local ${isDataset ? "dataset" : "model"} matches it.`
            : isDataset
              ? "Downloaded datasets, recipe outputs, and uploaded files will appear here."
              : "Downloaded repositories and indexed local folders will appear here."
        }
      />
    );
  }

  return (
    <>
      {pinnedItems.length > 0 && (
        <>
          <div className="flex items-center gap-1.5 px-1 pb-2 pt-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            <HugeiconsIcon
              icon={PinIcon}
              strokeWidth={1.75}
              className="size-3.5"
            />
            Pinned
          </div>
          {/* Pinned rows are few, so render them as a plain grid matching the
              virtualized list's lane count and row spacing. */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${Math.max(1, columns)}, minmax(0, 1fr))`,
              columnGap: 12,
              rowGap: rowHeightPx - cellHeightPx,
              paddingBottom: rowHeightPx - cellHeightPx,
            }}
          >
            {pinnedItems.map((item) => (
              <div
                key={`${item.variant}-${item.row.id}`}
                className="min-w-0"
                style={{ height: cellHeightPx }}
              >
                {renderInventoryRow(item)}
              </div>
            ))}
          </div>
          {unpinnedItems.length > 0 && (
            <div className="px-1 pb-2 pt-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              All {isDataset ? "datasets" : "models"}
            </div>
          )}
        </>
      )}
      <VirtualRows
        items={unpinnedItems}
        scrollElement={scrollElement}
        columns={columns}
        rowHeight={rowHeightPx}
        cellHeight={cellHeightPx}
        getKey={(item) => `${item.variant}-${item.row.id}`}
        renderRow={renderInventoryRow}
      />
    </>
  );
}
