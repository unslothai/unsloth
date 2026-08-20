


import { Spinner } from "@/components/ui/spinner";
import {
  makePinRank,
  pinKey,
  usePinnedModelsStore,
} from "@/features/model-picker";
import {
  CubeIcon,
  DownloadCircle02Icon,
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
      {/* Keep fetched results on screen when the Hub becomes unreachable. */}
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
                // searchFailure too: the footer is retained over an outage the
                // listing never saw, and useHubInfiniteScroll is gated on
                // reachability, so the button was visible and inert meanwhile.
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
          // The classified failure supplies the wording; the raw SDK error
          // appends the request URL, which carries the query.
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
  activeCheckpoint,
  activeGgufVariant,
  isDataset,
  inventoryTokens,
  deviceType,
  compact = false,
  sort,
  onInventoryChange,
  onOpenModelSettings,
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
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  deviceType: string | null;
  /** Narrow split master pane: render compact inventory rows. */
  compact?: boolean;
  sort: InventorySort;
  onInventoryChange?: () => void;
  onOpenModelSettings?: (row: CachedInventoryRow | LocalInventoryRow) => void;
}) {
  // Pinned repos surface first regardless of the active sort, which still orders within groups.
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const movePinned = usePinnedModelsStore((s) => s.movePinned);
  const beginPinnedDrag = usePinnedModelsStore((s) => s.beginPinnedDrag);
  const endPinnedDrag = usePinnedModelsStore((s) => s.endPinnedDrag);
  // Ref, not state: dragenter can fire before a dragstart re-render commits.
  const dragPinKeyRef = useRef<string | null>(null);
  // Dimming keys off the dragged CELL, not off its pin key: one repo cached in
  // two formats yields two rows sharing a single pin key, and dimming both
  // would report the untouched twin as the thing being dragged.
  const [dragRowKey, setDragRowKey] = useState<string | null>(null);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const inventoryItems = useMemo<InventoryItem[]>(() => {
    const merged: InventoryItem[] = [
      ...cachedRows.map((row) => ({ variant: "cached" as const, row })),
      ...localRows.map((row) => ({ variant: "local" as const, row })),
    ];
    // Pinned rows order by pin recency, not the active sort, so "Pin to top" lands where expected.
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
  // Pinned repos get their own labelled section; inventoryItems already sorts them first.
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
      onOpenSettings={onOpenModelSettings}
    />
  );

  if (!downloadedReady && !hasInventoryRows) {
    return (
      <div className="flex min-h-[240px] items-center justify-center gap-3 text-ui-13 text-muted-foreground">
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
                className="inline-flex h-8 items-center gap-1.5 rounded-full bg-transparent px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
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
        icon={query.trim() ? Search01Icon : DownloadCircle02Icon}
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
              columnGap: 12,
              rowGap: rowHeightPx - cellHeightPx,
              paddingBottom: rowHeightPx - cellHeightPx,
            }}
          >
            {pinnedItems.map((item) => {
              const rowKey = `${item.variant}-${item.row.id}`;
              // movePinned can only move a key that is in the pinned list, and
              // pins also exist as `repoId::quant` (written by the GGUF quant
              // menus). Deriving the key without checking membership would let
              // a row advertise a drag that every movePinned call silently
              // found nothing to do. pinnedCount selects this slice on the same
              // predicate today, so the check holds the two in lockstep rather
              // than trusting them to stay identical. Datasets are excluded
              // outright: pin keys carry no repo type, so a dataset whose
              // repoId also names a pinned model reaches this grid, and the row
              // menu offers datasets no pin action, so a drag here must not
              // reorder the user's model pins from the dataset list.
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
                    height: cellHeightPx,
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
                    // Reordering happens live on dragenter; this snapshot is
                    // what a cancelled drag rolls back to.
                    beginPinnedDrag();
                  }}
                  onDragEnd={() => {
                    dragPinKeyRef.current = null;
                    setDragRowKey(null);
                    // Escape, or a release outside any cell, reaches dragend
                    // without a drop. A drop already committed and cleared the
                    // session, so this call is then a no-op.
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
