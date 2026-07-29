// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import {
  type ReactNode,
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
import {
  type AllModelsView,
  type InventorySort,
  RESULT_GRID_ROW_HEIGHT_PX,
  RESULT_ROW_HEIGHT_PX,
  RESULT_SPLIT_ROW_HEIGHT_PX,
} from "./models-table";

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
  typeFilterActive: boolean;
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
  header,
  downloadedHeader,
  suppressEmptyState = false,
  resetScrollKey,
  discoverView,
  inventorySort,
}: {
  state: ModelsCatalogState;
  pagination: ModelsCatalogPagination;
  handlers: ModelsCatalogHandlers;
  header?: ReactNode;
  downloadedHeader?: ReactNode;
  suppressEmptyState?: boolean;
  resetScrollKey?: string;
  discoverView: AllModelsView;
  inventorySort: InventorySort;
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
    hasActiveFilters,
    typeFilterActive,
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
  const [discoverHeaderHeight, setDiscoverHeaderHeight] = useState(0);
  const headerRef = useRef<HTMLDivElement>(null);
  const lastHeaderHeightRef = useRef(0);
  const previousResetScrollKeyRef = useRef(resetScrollKey);
  const previousScannedCountRef = useRef(scannedCount);
  const previousLoadingIntentRef = useRef(loadingIntentCount);
  const discoverScrollRef = useRef<HTMLDivElement>(null);
  const downloadedScrollRef = useRef<HTMLDivElement>(null);
  const [discoverScrollEl, setDiscoverScrollEl] =
    useState<HTMLDivElement | null>(null);
  const [downloadedScrollEl, setDownloadedScrollEl] =
    useState<HTMLDivElement | null>(null);
  const activeTabRef = useRef(tab);
  // Per-tab scroll positions: some browsers drop the hidden pane's scrollTop,
  // so mirror it live via the scroll listener and restore on tab entry.
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

  // Restore the incoming pane's scrollTop and rebind scrollRef. Don't read
  // scrollTop off the outgoing pane: overflow toggling can clamp it to 0 before
  // this runs, so trust the live listener mirror instead and avoid that race.
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
    setScrolled((current) =>
      current === nextScrolled ? current : nextScrolled,
    );
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

  useLayoutEffect(() => {
    if (resetScrollKey === previousResetScrollKeyRef.current) {
      return;
    }
    previousResetScrollKeyRef.current = resetScrollKey;
    const el = discoverScrollRef.current;
    if (el) {
      el.scrollTop = 0;
    }
    savedScrollTopsRef.current.discover = 0;
  }, [resetScrollKey]);

  useLayoutEffect(() => {
    if (tab !== "discover") {
      return;
    }
    const node = headerRef.current;
    if (!node) {
      lastHeaderHeightRef.current = 0;
      setDiscoverHeaderHeight((current) => (current === 0 ? current : 0));
      return;
    }
    let frame: number | null = null;
    const measure = () => {
      frame = null;
      const next = node.offsetHeight;
      if (next !== lastHeaderHeightRef.current) {
        lastHeaderHeightRef.current = next;
        setDiscoverHeaderHeight(next);
      }
    };
    const schedule = () => {
      if (frame !== null) {
        return;
      }
      frame = window.requestAnimationFrame(measure);
    };
    measure();
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(schedule);
      observer.observe(node);
    }
    return () => {
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
      }
      observer?.disconnect();
    };
  }, [tab, header]);

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
    setStreamingActive((current) =>
      current === nextActive ? current : nextActive,
    );
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
  // overflow-y stays `auto` on BOTH panes: toggling to `hidden` clamps the
  // inactive pane's scrollTop to 0 and corrupts the mirror; visibility +
  // pointer-events-none hides it while preserving native scroll state.
  // Non-split reserves an equal `both-edges` gutter so the centered 1100px
  // column stays symmetric and aligned with the top bar; split mode pins a
  // narrow master left, so it reserves only the right (divider) gutter.
  const scrollPaneClassName =
    "absolute inset-0 min-h-0 overflow-x-hidden overflow-y-auto pb-6 pt-0 [overflow-anchor:none] [scrollbar-width:thin] " +
    (discoverView === "split"
      ? "[scrollbar-gutter:stable]"
      : "[scrollbar-gutter:stable_both-edges]");
  // Split mode keeps the top bar's left padding to align the list header but
  // tightens the right padding so compact rows run wider toward the divider.
  const splitView = discoverView === "split";
  const discoverColumnClassName = splitView
    ? "mx-auto w-full max-w-[1100px] pl-5 pr-2 sm:pl-8"
    : "mx-auto w-full max-w-[1100px] px-5 sm:px-8";
  const downloadedColumnClassName = splitView
    ? "mx-auto w-full max-w-[1100px] pl-5 pr-2 sm:pl-8"
    : "mx-auto w-full max-w-[1100px] px-5 sm:px-8";
  const discoverActive = tab === "discover";
  const downloadedActive = tab === "downloaded";
  const discoverInactiveHeight = Math.max(
    savedScrollHeightsRef.current.discover,
    (discoverView === "two"
      ? Math.ceil(discoverRows.length / 2)
      : discoverRows.length) *
      (discoverView === "grid"
        ? RESULT_GRID_ROW_HEIGHT_PX
        : discoverView === "split"
          ? RESULT_SPLIT_ROW_HEIGHT_PX
          : RESULT_ROW_HEIGHT_PX) +
      lastHeaderHeightRef.current,
  );
  const downloadedInactiveHeight = Math.max(
    savedScrollHeightsRef.current.downloaded,
    (cachedRows.length + localRows.length) * RESULT_GRID_ROW_HEIGHT_PX,
  );

  return (
    <div className="relative flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      {/* Banner sits outside the scroll container so toggling it never shifts virtualized rows. */}
      {downloadedActive && inventoryWarning && (
        <InventoryWarningRow
          isDataset={isDataset}
          onRetry={() => onInventoryChange?.()}
        />
      )}
      <div className="relative min-h-0 flex-1">
        <div
          aria-hidden="true"
          data-scrolled={scrolled || undefined}
          className="hub-scroll-fade pointer-events-none absolute inset-x-0 top-0 z-10 h-7"
        />
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
            <div className={discoverColumnClassName}>
              {header ? <div ref={headerRef}>{header}</div> : null}
              <DiscoverList
                discoverRows={discoverRows}
                onSelect={onSelect}
                isLoading={isLoading}
                query={query}
                scrollElement={discoverScrollEl}
                scrollMargin={discoverHeaderHeight}
                suppressEmptyState={suppressEmptyState}
                sentinelRef={sentinelRef}
                searchError={searchError}
                online={online}
                isDataset={isDataset}
                deviceType={deviceType}
                scannedCount={scannedCount}
                isLoadingMore={isLoadingMore}
                hasMore={hasMore}
                hasActiveFilters={hasActiveFilters}
                onFetchMore={onFetchMore}
                onClearFilters={onClearFilters}
                onRetry={onRetry}
                onSwitchDevice={onSwitchDevice}
                view={discoverView}
                selectedId={selectedId}
              />
            </div>
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
            <div className={downloadedColumnClassName}>
              {downloadedHeader ? (
                <div className="flex flex-col gap-3 pt-6">
                  {downloadedHeader}
                </div>
              ) : null}
              <DownloadedList
                cachedRows={cachedRows}
                localRows={localRows}
                selectedId={selectedId}
                onSelect={onSelect}
                downloadedReady={downloadedReady}
                inventoryError={inventoryError}
                query={query}
                typeFilterActive={typeFilterActive}
                onClearFilters={onClearFilters}
                scrollElement={downloadedScrollEl}
                activeCheckpoint={activeCheckpoint}
                activeGgufVariant={activeGgufVariant}
                isDataset={isDataset}
                inventoryTokens={inventoryTokens}
                deviceType={deviceType}
                compact={splitView}
                columns={discoverView === "two" ? 2 : 1}
                sort={inventorySort}
                onInventoryChange={onInventoryChange}
              />
            </div>
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
