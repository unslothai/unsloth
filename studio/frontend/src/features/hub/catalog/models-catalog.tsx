// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import {
  type RefObject,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  ModelsTab,
} from "../types";
import {
  DiscoverList,
  DownloadedList,
  InventoryWarningRow,
} from "./models-catalog-lists";
import { CATALOG_ROW_HEIGHT_PX } from "./models-catalog-rows";

export interface ModelsCatalogState {
  tab: ModelsTab;
  discoverRows: DiscoverRow[];
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  selectedId: string | null;
  isLoading: boolean;
  downloadedReady: boolean;
  inventoryError: boolean;
  inventoryWarning: boolean;
  query: string;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  searchError: string | null;
  online: boolean;
  isDataset: boolean;
  inventoryTokens: readonly string[];
  scannedCount: number;
  loadingIntentCount: number;
  hasMore: boolean;
  manualFetchAvailable: boolean;
  hasActiveFilters: boolean;
}

export interface ModelsCatalogPagination {
  scrollRef: RefObject<HTMLDivElement | null>;
  sentinelRef: RefObject<HTMLDivElement | null>;
  isLoadingMore: boolean;
}

export interface ModelsCatalogHandlers {
  onSelect: (id: string) => void;
  onFetchMore: () => void;
  onClearFilters: () => void;
  onRetry: () => void;
  onInventoryChange?: () => void;
  onSwitchDevice?: () => void;
}

function assignRef<T>(ref: RefObject<T | null>, value: T | null) {
  (ref as { current: T | null }).current = value;
}

export const ModelsCatalog = memo(function ModelsCatalog({
  state,
  pagination,
  handlers,
}: {
  state: ModelsCatalogState;
  pagination: ModelsCatalogPagination;
  handlers: ModelsCatalogHandlers;
}) {
  const {
    tab,
    discoverRows,
    cachedRows,
    localRows,
    selectedId,
    isLoading,
    downloadedReady,
    inventoryError,
    inventoryWarning,
    query,
    activeCheckpoint,
    activeGgufVariant,
    searchError,
    online,
    isDataset,
    inventoryTokens,
    scannedCount,
    loadingIntentCount,
    hasMore,
    manualFetchAvailable,
    hasActiveFilters,
  } = state;
  const { scrollRef, sentinelRef, isLoadingMore } = pagination;
  const {
    onSelect,
    onFetchMore,
    onClearFilters,
    onRetry,
    onInventoryChange,
    onSwitchDevice,
  } = handlers;
  const [scrolled, setScrolled] = useState(false);
  const [streamingActive, setStreamingActive] = useState(false);
  const previousScannedCountRef = useRef(scannedCount);
  const previousLoadingIntentRef = useRef(loadingIntentCount);
  const discoverScrollRef = useRef<HTMLDivElement>(null);
  const downloadedScrollRef = useRef<HTMLDivElement>(null);
  const [discoverScrollEl, setDiscoverScrollEl] = useState<HTMLDivElement | null>(
    null,
  );
  const [downloadedScrollEl, setDownloadedScrollEl] =
    useState<HTMLDivElement | null>(null);
  const activeTabRef = useRef(tab);
  // Per-tab scroll positions. Some browsers drop the hidden pane's scrollTop, so
  // mirror it live via the scroll listener and restore on tab entry.
  const savedScrollTopsRef = useRef<Record<ModelsTab, number>>({
    discover: 0,
    downloaded: 0,
  });
  const savedScrollHeightsRef = useRef<Record<ModelsTab, number>>({
    discover: 0,
    downloaded: 0,
  });
  const deviceType = usePlatformStore((s) => s.deviceType);

  const setDiscoverScrollNode = useCallback((node: HTMLDivElement | null) => {
    discoverScrollRef.current = node;
    setDiscoverScrollEl(node);
  }, []);

  const setDownloadedScrollNode = useCallback((node: HTMLDivElement | null) => {
    downloadedScrollRef.current = node;
    setDownloadedScrollEl(node);
  }, []);

  // Restore the incoming pane's scrollTop and rebind scrollRef. Don't read scrollTop
  // off the outgoing pane: overflow toggling can clamp it to 0 before this runs, so
  // trust the live listener mirror instead and avoid that race.
  useLayoutEffect(() => {
    let restoreFrame: number | null = null;
    const previousTab = activeTabRef.current;
    if (previousTab !== tab) {
      const previousEl =
        previousTab === "discover"
          ? discoverScrollRef.current
          : downloadedScrollRef.current;
      if (previousEl) {
        savedScrollTopsRef.current[previousTab] = previousEl.scrollTop;
        savedScrollHeightsRef.current[previousTab] = previousEl.scrollHeight;
      }
    }
    const nextEl =
      tab === "discover"
        ? discoverScrollRef.current
        : downloadedScrollRef.current;
    if (nextEl) {
      const saved = savedScrollTopsRef.current[tab];
      if (nextEl.scrollTop !== saved) {
        nextEl.scrollTop = saved;
      }
      savedScrollHeightsRef.current[tab] = nextEl.scrollHeight;
      restoreFrame = window.requestAnimationFrame(() => {
        if (activeTabRef.current === tab && nextEl.scrollTop !== saved) {
          nextEl.scrollTop = saved;
        }
      });
    }
    activeTabRef.current = tab;
    assignRef(scrollRef, nextEl);
    const nextScrolled = (nextEl?.scrollTop ?? 0) > 0;
    setScrolled((current) => (current === nextScrolled ? current : nextScrolled));
    return () => {
      if (restoreFrame !== null) {
        window.cancelAnimationFrame(restoreFrame);
      }
    };
  }, [scrollRef, tab]);

  useEffect(() => {
    const discoverEl = discoverScrollRef.current;
    const downloadedEl = downloadedScrollRef.current;
    // Mirror scrollTop ONLY from the active pane; the inactive pane can fire
    // spurious scroll events that would overwrite the saved position with 0.
    const onDiscoverScroll = () => {
      if (!discoverEl || activeTabRef.current !== "discover") {
        return;
      }
      savedScrollTopsRef.current.discover = discoverEl.scrollTop;
      savedScrollHeightsRef.current.discover = discoverEl.scrollHeight;
      const nextScrolled = discoverEl.scrollTop > 0;
      setScrolled((current) =>
        current === nextScrolled ? current : nextScrolled,
      );
    };
    const onDownloadedScroll = () => {
      if (!downloadedEl || activeTabRef.current !== "downloaded") {
        return;
      }
      savedScrollTopsRef.current.downloaded = downloadedEl.scrollTop;
      savedScrollHeightsRef.current.downloaded = downloadedEl.scrollHeight;
      const nextScrolled = downloadedEl.scrollTop > 0;
      setScrolled((current) =>
        current === nextScrolled ? current : nextScrolled,
      );
    };
    discoverEl?.addEventListener("scroll", onDiscoverScroll, { passive: true });
    downloadedEl?.addEventListener("scroll", onDownloadedScroll, {
      passive: true,
    });
    return () => {
      discoverEl?.removeEventListener("scroll", onDiscoverScroll);
      downloadedEl?.removeEventListener("scroll", onDownloadedScroll);
    };
  }, []);

  useEffect(
    () => () => {
      if (
        scrollRef.current === discoverScrollRef.current ||
        scrollRef.current === downloadedScrollRef.current
      ) {
        assignRef(scrollRef, null);
      }
    },
    [scrollRef],
  );

  useEffect(() => {
    // Only the Discover tab drives the streaming loading bar.
    if (tab !== "discover") {
      previousScannedCountRef.current = scannedCount;
      previousLoadingIntentRef.current = loadingIntentCount;
      setStreamingActive((current) => (current ? false : current));
      return;
    }
    const previousCount = previousScannedCountRef.current;
    const previousLoadingIntent = previousLoadingIntentRef.current;
    previousScannedCountRef.current = scannedCount;
    previousLoadingIntentRef.current = loadingIntentCount;
    const rowsScanned = scannedCount > previousCount;
    const fetchRequested = loadingIntentCount > previousLoadingIntent;
    let nextActive = false;
    let linger = false;
    if (isLoading || isLoadingMore) {
      nextActive = true;
    } else if (rowsScanned || fetchRequested) {
      nextActive = true;
      linger = true;
    }
    setStreamingActive((current) => (current === nextActive ? current : nextActive));
    const timer = linger
      ? window.setTimeout(() => setStreamingActive(false), 1400)
      : null;
    return () => {
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [tab, isLoading, isLoadingMore, scannedCount, loadingIntentCount]);

  const showDiscoverLoading = tab === "discover" && streamingActive;
  // overflow-y stays `auto` on BOTH panes: toggling to `hidden` would clamp the
  // inactive pane's scrollTop to 0 and corrupt our mirror. Visibility +
  // pointer-events-none hides it while preserving native scroll state.
  const scrollPaneClassName =
    "absolute inset-0 min-h-0 overflow-y-auto pb-6 pl-4 pr-3 pt-0 [overflow-anchor:none] [scrollbar-gutter:stable] [scrollbar-width:thin]";
  const discoverActive = tab === "discover";
  const downloadedActive = tab === "downloaded";
  const discoverInactiveHeight = Math.max(
    savedScrollHeightsRef.current.discover,
    discoverRows.length * CATALOG_ROW_HEIGHT_PX,
  );
  const downloadedInactiveHeight = Math.max(
    savedScrollHeightsRef.current.downloaded,
    (cachedRows.length + localRows.length) * CATALOG_ROW_HEIGHT_PX,
  );

  return (
    <div className="relative flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <div
        aria-hidden="true"
        className={cn(
          "mx-5 shrink-0 border-t transition-colors",
          scrolled ? "border-border" : "border-transparent",
        )}
      />
      {/* Banner sits outside the scroll container so toggling it never shifts virtualized rows. */}
      {downloadedActive && inventoryWarning && (
        <InventoryWarningRow
          isDataset={isDataset}
          onRetry={() => onInventoryChange?.()}
        />
      )}
      <div className="relative min-h-0 flex-1">
        <div
          ref={setDiscoverScrollNode}
          data-hub-scroll="true"
          aria-hidden={!discoverActive}
          tabIndex={discoverActive ? undefined : -1}
          className={cn(
            scrollPaneClassName,
            discoverActive ? "visible" : "pointer-events-none invisible",
          )}
        >
          {discoverActive ? (
            <DiscoverList
              discoverRows={discoverRows}
              selectedId={selectedId}
              onSelect={onSelect}
              isLoading={isLoading}
              query={query}
              scrollElement={discoverScrollEl}
              sentinelRef={sentinelRef}
              activeCheckpoint={activeCheckpoint}
              searchError={searchError}
              online={online}
              isDataset={isDataset}
              deviceType={deviceType}
              scannedCount={scannedCount}
              isLoadingMore={isLoadingMore}
              hasMore={hasMore}
              manualFetchAvailable={manualFetchAvailable}
              hasActiveFilters={hasActiveFilters}
              onFetchMore={onFetchMore}
              onClearFilters={onClearFilters}
              onRetry={onRetry}
              onSwitchDevice={onSwitchDevice}
            />
          ) : (
            <div
              aria-hidden="true"
              style={{ height: discoverInactiveHeight }}
            />
          )}
        </div>
        <div
          ref={setDownloadedScrollNode}
          data-hub-scroll="true"
          aria-hidden={!downloadedActive}
          tabIndex={downloadedActive ? undefined : -1}
          className={cn(
            scrollPaneClassName,
            downloadedActive ? "visible" : "pointer-events-none invisible",
          )}
        >
          {downloadedActive ? (
            <DownloadedList
              cachedRows={cachedRows}
              localRows={localRows}
              selectedId={selectedId}
              onSelect={onSelect}
              downloadedReady={downloadedReady}
              inventoryError={inventoryError}
              query={query}
              scrollElement={downloadedScrollEl}
              activeCheckpoint={activeCheckpoint}
              activeGgufVariant={activeGgufVariant}
              isDataset={isDataset}
              inventoryTokens={inventoryTokens}
              deviceType={deviceType}
              onInventoryChange={onInventoryChange}
            />
          ) : (
            <div
              aria-hidden="true"
              style={{ height: downloadedInactiveHeight }}
            />
          )}
        </div>
      </div>
      <output
        aria-live="polite"
        aria-label={showDiscoverLoading ? "Loading models" : undefined}
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 z-10 h-[3px] transition-opacity duration-150",
          showDiscoverLoading ? "opacity-100" : "opacity-0",
        )}
      >
        <div
          className="hub-loading-bar"
          data-active={showDiscoverLoading || undefined}
        />
      </output>
    </div>
  );
});
