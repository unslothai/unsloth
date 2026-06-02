// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import {
  CubeIcon,
  DownloadCircle02Icon,
  FolderSearchIcon,
} from "@hugeicons/core-free-icons";
import type { RefObject } from "react";
import { useMemo } from "react";
import { inventoryRowMatches, scoreInventoryRow } from "../lib/inventory-search";
import type {
  CachedInventoryRow,
  DiscoverRow,
  InventoryHint,
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
import {
  DiscoverModelRow,
  InventoryRow,
  VirtualRows,
} from "./models-catalog-rows";

type InventoryItem =
  | { variant: "cached"; row: CachedInventoryRow }
  | { variant: "local"; row: LocalInventoryRow };

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
  selectedId,
  onSelect,
  isLoading,
  query,
  scrollElement,
  sentinelRef,
  activeCheckpoint,
  searchError,
  online,
  isDataset,
  deviceType,
  scannedCount,
  isLoadingMore,
  hasMore,
  manualFetchAvailable,
  hasActiveFilters,
  onFetchMore,
  onClearFilters,
  onRetry,
  onSwitchDevice,
}: {
  discoverRows: DiscoverRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  isLoading: boolean;
  query: string;
  scrollElement: HTMLDivElement | null;
  sentinelRef: RefObject<HTMLDivElement | null>;
  activeCheckpoint: string | null;
  searchError: string | null;
  online: boolean;
  isDataset: boolean;
  deviceType: string | null;
  scannedCount: number;
  isLoadingMore: boolean;
  hasMore: boolean;
  manualFetchAvailable: boolean;
  hasActiveFilters: boolean;
  onFetchMore: () => void;
  onClearFilters: () => void;
  onRetry: () => void;
  onSwitchDevice?: () => void;
}) {
  return (
    <>
      {online ? (
        searchError && discoverRows.length === 0 ? (
          <NetworkErrorState
            online={online}
            message={searchError}
            onRetry={onRetry}
            resourceLabel={isDataset ? "datasets" : "models"}
          />
        ) : hasMore && discoverRows.length === 0 ? (
          <DiscoverFetchMoreState
            scannedCount={scannedCount}
            hasActiveFilters={hasActiveFilters}
            isLoadingMore={isLoadingMore}
            onFetchMore={onFetchMore}
            onClearFilters={onClearFilters}
          />
        ) : discoverRows.length === 0 && isLoading ? (
          <SkeletonList />
        ) : discoverRows.length === 0 ? (
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
        ) : (
          <>
            <VirtualRows
              items={discoverRows}
              scrollElement={scrollElement}
              getKey={(row) => row.id}
              renderRow={(row) => (
                <DiscoverModelRow
                  row={row}
                  selected={selectedId === row.id}
                  active={modelIdsMatch(activeCheckpoint, row.id)}
                  deviceType={deviceType}
                  isDataset={isDataset}
                  onSelect={onSelect}
                />
              )}
            />
            {hasMore && (
              <DiscoverFetchMoreFooter
                scannedCount={scannedCount}
                manualFetchAvailable={manualFetchAvailable}
                hasActiveFilters={hasActiveFilters}
                isLoadingMore={isLoadingMore}
                onFetchMore={onFetchMore}
              />
            )}
          </>
        )
      ) : (
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
  activeCheckpoint,
  activeGgufVariant,
  isDataset,
  inventoryTokens,
  deviceType,
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
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  deviceType: string | null;
  onInventoryChange?: (hint?: InventoryHint) => void;
}) {
  const inventoryItems = useMemo<InventoryItem[]>(() => {
    const merged: InventoryItem[] = [
      ...cachedRows.map((row) => ({ variant: "cached" as const, row })),
      ...localRows.map((row) => ({ variant: "local" as const, row })),
    ];
    if (inventoryTokens.length === 0) return merged;
    return merged
      .map((item, index) => ({
        item,
        index,
        score: scoreInventoryRow(item.row, inventoryTokens),
      }))
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .map((entry) => entry.item);
  }, [cachedRows, localRows, inventoryTokens]);
  const hasInventoryRows = cachedRows.length > 0 || localRows.length > 0;

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
    <VirtualRows
      items={inventoryItems}
      scrollElement={scrollElement}
      getKey={(item) => `${item.variant}-${item.row.id}`}
      renderRow={(item) => (
        <InventoryRow
          row={item.row}
          selected={selectedId === item.row.id}
          activeCheckpoint={activeCheckpoint}
          activeGgufVariant={activeGgufVariant}
          isDataset={isDataset}
          dimmed={!inventoryRowMatches(item.row, inventoryTokens)}
          deviceType={deviceType}
          onSelect={onSelect}
          onChange={onInventoryChange}
        />
      )}
    />
  );
}
