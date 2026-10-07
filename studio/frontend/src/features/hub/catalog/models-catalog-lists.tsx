// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  makePinRank,
  pinKey,
  usePinnedModelsStore,
} from "@/features/model-picker";
import {
  CubeIcon,
  Download01Icon,
  PinIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Ref } from "react";
import type { HubFailure } from "@/features/hub/lib/network";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
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
  compareInventoryItemsByRecent,
  type InventoryItem,
  inventoryItemSize,
  inventoryItemTitle,
} from "./inventory-sort";
import {
  DiscoverFetchMoreFooter,
  DiscoverFetchMoreState,
  EmptyState,
  InventoryErrorState,
  NetworkErrorState,
  SkeletonList,
} from "./catalog-states";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import {
  CATALOG_COLUMN_GAP_PX,
  InventoryRow,
  VirtualRows,
} from "./models-catalog-rows";
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

export function InventoryWarningRow({
  isDataset,
  onRetry,
}: {
  isDataset: boolean;
  onRetry: () => void;
}) {
  return (
    <div className="mx-5 mt-2 rounded-[8px] border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-ui-12p5 text-muted-foreground">
      <div className="flex items-center justify-between gap-3">
        <span>
          Some on-device sources couldn't be scanned. Showing available{" "}
          {isDataset ? "datasets" : "models"}.
        </span>
        <button
          type="button"
          className="shrink-0 text-ui-12 font-medium text-foreground transition-colors hover:text-primary"
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
  searchFailure,
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
  showFormatDots = true,
}: {
  discoverRows: DiscoverRow[];
  onSelect: (id: string) => void;
  selectedId?: string | null;
  isLoading: boolean;
  query: string;
  scrollElement: HTMLDivElement | null;
  scrollMargin?: number;
  suppressEmptyState?: boolean;
  sentinelRef: Ref<HTMLDivElement>;
  searchError: string | null;
  searchFailure?: HubFailure | null;
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
  showFormatDots?: boolean;
}) {
  // "two" = two cards per row; "grid" = compact table rows; "split" = one card per row.
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
      {online || discoverRows.length > 0 ? (
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
                    showFormatDot={showFormatDots}
                    onSelect={onSelect}
                  />
                ) : isCardLike ? (
                  <ResultCard
                    row={row}
                    deviceType={deviceType}
                    isDataset={isDataset}
                    showFormatDot={showFormatDots}
                    onSelect={onSelect}
                  />
                ) : (
                  <ResultGridRow
                    row={row}
                    deviceType={deviceType}
                    isDataset={isDataset}
                    showFormatDot={showFormatDots}
                    onSelect={onSelect}
                  />
                )
              }
            />
            {/* searchFailure, not `online`: that is the backoff TTL, which
                lapses on a timer, so the notice and its Retry vanished before
                anything had proved recovery. The cause clears on success. It
                covers the avatar and card case too, which marks the same origin
                without the listing ever failing. */}
            {(hasMore || searchError || searchFailure) && (
              <DiscoverFetchMoreFooter
                hasActiveFilters={hasActiveFilters}
                isLoadingMore={isLoadingMore}
                onFetchMore={onFetchMore}
                // searchFailure too: infinite scroll is gated on reachability, so the button would be inert.
                failed={Boolean(searchError || searchFailure)}
                failureText={searchFailure?.message ?? searchError ?? ""}
                onRetry={onRetry}
              />
            )}
          </>
        ) : suppressEmptyState ? null : searchError ? (
          <NetworkErrorState
            online={online}
            message={searchError}
            failure={searchFailure}
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
            icon={query.trim() ? Search01Icon : CubeIcon}
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
          // The raw SDK error includes the request URL (with the query), so prefer the classified one.
          message={searchFailure ? "" : (searchError ?? "")}
          failure={searchFailure}
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
  typeFilterActive = false,
  onClearFilters,
  scrollElement,
  columns = 1,
  isDataset,
  inventoryTokens,
  deviceType,
  compact = false,
  sort,
  onInventoryChange,
  showFormatDots = true,
}: {
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  downloadedReady: boolean;
  inventoryError: boolean;
  query: string;
  typeFilterActive?: boolean;
  onClearFilters?: () => void;
  scrollElement: HTMLDivElement | null;
  columns?: number;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  deviceType: string | null;
  compact?: boolean;
  sort: InventorySort;
  onInventoryChange?: () => void;
  showFormatDots?: boolean;
}) {
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const movePinned = usePinnedModelsStore((s) => s.movePinned);
  const beginPinnedDrag = usePinnedModelsStore((s) => s.beginPinnedDrag);
  const endPinnedDrag = usePinnedModelsStore((s) => s.endPinnedDrag);
  // Ref, not state: dragenter can fire before a dragstart re-render commits.
  const dragPinKeyRef = useRef<string | null>(null);
  // Dim by dragged cell, not pin key: one repo in two formats shares a pin key.
  const [dragRowKey, setDragRowKey] = useState<string | null>(null);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const inventoryItems = useMemo<InventoryItem[]>(() => {
    const merged: InventoryItem[] = [
      ...cachedRows.map((row) => ({ variant: "cached" as const, row })),
      ...localRows.map((row) => ({ variant: "local" as const, row })),
    ];
    // Pinned rows order by pin recency, not the active sort.
    const rank = makePinRank(pinnedIds);
    const pinRank = (item: InventoryItem) =>
      item.row.repoId ? rank(pinKey(item.row.repoId)) : Number.MAX_SAFE_INTEGER;
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
        .sort(
          (a, b) =>
            pinRank(a.item) - pinRank(b.item) ||
            compareInventoryItemsByRecent(a.item, b.item) ||
            a.index - b.index,
        )
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
  const pinnedCount = useMemo(
    () =>
      inventoryItems.filter(
        (item) => item.row.repoId && pinnedSet.has(pinKey(item.row.repoId)),
      ).length,
    [inventoryItems, pinnedSet],
  );
  const pinnedItems = inventoryItems.slice(0, pinnedCount);
  const unpinnedItems = inventoryItems.slice(pinnedCount);
  const [virtualRowsWrapper, setVirtualRowsWrapper] =
    useState<HTMLDivElement | null>(null);
  const [scrollMargin, setScrollMargin] = useState(0);
  useLayoutEffect(() => {
    if (!virtualRowsWrapper || !scrollElement) return;
    const measure = () => {
      const margin = Math.max(
        0,
        Math.round(
          virtualRowsWrapper.getBoundingClientRect().top -
            scrollElement.getBoundingClientRect().top +
            scrollElement.scrollTop,
        ),
      );
      setScrollMargin((current) => (current === margin ? current : margin));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(virtualRowsWrapper.parentElement ?? scrollElement);
    return () => observer.disconnect();
  }, [virtualRowsWrapper, scrollElement]);
  const rowHeightPx = compact
    ? RESULT_SPLIT_ROW_HEIGHT_PX
    : RESULT_GRID_ROW_HEIGHT_PX;
  const cellHeightPx = compact ? RESULT_SPLIT_HEIGHT_PX : RESULT_GRID_HEIGHT_PX;
  // The pinned grid is laid out by hand, so scale it to match VirtualRows.
  const pinnedScale = useUiSpaceScale();
  const pinnedRowHeightPx = Math.round(rowHeightPx * pinnedScale);
  const pinnedCellHeightPx = Math.round(cellHeightPx * pinnedScale);
  const pinnedColumnGapPx = Math.round(CATALOG_COLUMN_GAP_PX * pinnedScale);
  const renderInventoryRow = (item: InventoryItem) => (
    <InventoryRow
      row={item.row}
      selected={selectedId === item.row.id}
      isDataset={isDataset}
      dimmed={!inventoryRowMatches(item.row, inventoryTokens)}
      deviceType={deviceType}
      compact={compact}
      showFormatDot={showFormatDots}
      onSelect={onSelect}
      onChange={onInventoryChange}
    />
  );

  if (!downloadedReady && !hasInventoryRows) {
    return (
      <div className="flex min-h-[calc(240px*var(--ui-space-scale,1))] items-center justify-center gap-3 text-ui-13 text-muted-foreground">
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
    if (!query.trim() && typeFilterActive) {
      return (
        <EmptyState
          icon={Search01Icon}
          title="No matching models on device"
          body="No downloaded or local model matches the selected type filter."
          action={
            onClearFilters && (
              <button
                type="button"
                onClick={onClearFilters}
                className="inline-flex h-8 items-center gap-1.5 rounded-full bg-transparent px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] dark:hover:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]"
              >
                Show all types
              </button>
            )
          }
        />
      );
    }
    return (
      <EmptyState
        icon={query.trim() ? Search01Icon : Download01Icon}
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
          <div className="flex items-center gap-1.5 px-1 pb-2 pt-3 text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">
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
              columnGap: pinnedColumnGapPx,
              rowGap: pinnedRowHeightPx - pinnedCellHeightPx,
              paddingBottom: pinnedRowHeightPx - pinnedCellHeightPx,
            }}
          >
            {pinnedItems.map((item) => {
              const rowKey = `${item.variant}-${item.row.id}`;
              // Only offer drag for keys actually in the pinned list (pins may be `repoId::quant`).
              // Datasets are excluded: pin keys carry no repo type, so a drag could reorder model pins.
              const itemPinKey =
                !isDataset &&
                item.row.repoId &&
                pinnedSet.has(pinKey(item.row.repoId))
                  ? pinKey(item.row.repoId)
                  : null;
              return (
                <div
                  key={rowKey}
                  className="min-w-0"
                  style={{
                    height: pinnedCellHeightPx,
                    opacity: dragRowKey === rowKey ? 0.4 : undefined,
                  }}
                  draggable={itemPinKey != null}
                  onDragStart={(event) => {
                    if (!itemPinKey) return;
                    event.dataTransfer.effectAllowed = "move";
                    // Firefox will not start a drag without data.
                    event.dataTransfer.setData("text/plain", itemPinKey);
                    dragPinKeyRef.current = itemPinKey;
                    setDragRowKey(rowKey);
                    // Reordering happens live on dragenter; this snapshot is the rollback for a cancelled drag.
                    beginPinnedDrag();
                  }}
                  onDragEnd={() => {
                    dragPinKeyRef.current = null;
                    setDragRowKey(null);
                    // Escape or release outside a cell reaches dragend without a drop; after a drop this is a no-op.
                    endPinnedDrag(false);
                  }}
                  onDragOver={(event) => {
                    if (dragPinKeyRef.current) event.preventDefault();
                  }}
                  onDragEnter={() => {
                    const dragKey = dragPinKeyRef.current;
                    if (dragKey && itemPinKey && dragKey !== itemPinKey) {
                      movePinned(dragKey, itemPinKey);
                    }
                  }}
                  onDrop={(event) => {
                    event.preventDefault();
                    dragPinKeyRef.current = null;
                    setDragRowKey(null);
                    endPinnedDrag(true);
                  }}
                >
                  {renderInventoryRow(item)}
                </div>
              );
            })}
          </div>
          {unpinnedItems.length > 0 && (
            <div className="px-1 pb-2 pt-2 text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">
              All {isDataset ? "datasets" : "models"}
            </div>
          )}
        </>
      )}
      <div ref={setVirtualRowsWrapper}>
        <VirtualRows
          items={unpinnedItems}
          scrollElement={scrollElement}
          scrollMargin={scrollMargin}
          columns={columns}
          rowHeight={rowHeightPx}
          cellHeight={cellHeightPx}
          getKey={(item) => `${item.variant}-${item.row.id}`}
          renderRow={renderInventoryRow}
        />
      </div>
    </>
  );
}
