// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * IntersectionObserver sentinel for infinite scroll, plus a ResizeObserver
 * fallback that auto-fetches while the scroll container doesn't yet overflow.
 * Fallback firings are coalesced to one `scrollHeight` read (forced layout) per
 * frame; concurrency is gated at the data-source layer.
 *
 * `signal` (typically `results.length`) is a dep so the fit check re-runs after
 * a fetch even when the page filter rejected every new row and the DOM didn't
 * change. `DEFAULT_MAX_AUTO_FILL_FETCHES` caps a runaway sweep of the full
 * listing; callers with a manual continuation UI can lower it.
 */
const DEFAULT_MAX_AUTO_FILL_FETCHES = 40;

export interface InfiniteScrollOptions {
  enabled?: boolean;
  onFetchIntent?: () => void;
  resultCount?: number;
  resetKey?: string | number | boolean | null;
  maxAutoFillFetches?: number;
  manualFetchAfterAutoFill?: boolean;
  isFetching?: boolean;
}

function hasScrollableOverflow(root: HTMLElement): boolean {
  return root.scrollHeight > root.clientHeight + 4;
}

export function useHubInfiniteScroll(
  fetchMore: () => boolean | void,
  signal: number,
  options: InfiniteScrollOptions = {},
) {
  const enabled = options.enabled ?? true;
  const onFetchIntent = options.onFetchIntent;
  const resultCount = options.resultCount ?? signal;
  const resetKey = options.resetKey ?? null;
  const maxAutoFillFetches =
    options.maxAutoFillFetches ?? DEFAULT_MAX_AUTO_FILL_FETCHES;
  const manualFetchAfterAutoFill = options.manualFetchAfterAutoFill ?? false;
  const isFetching = options.isFetching ?? false;

  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const fetchMoreRef = useRef(fetchMore);
  const onFetchIntentRef = useRef(onFetchIntent);
  const enabledRef = useRef(enabled);
  const isFetchingRef = useRef(isFetching);
  useEffect(() => {
    fetchMoreRef.current = fetchMore;
  }, [fetchMore]);
  useEffect(() => {
    onFetchIntentRef.current = onFetchIntent;
  }, [onFetchIntent]);
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);
  useEffect(() => {
    isFetchingRef.current = isFetching;
  }, [isFetching]);

  const autoFireCountRef = useRef(0);
  const prevSignalRef = useRef(signal);
  const prevResultCountRef = useRef(resultCount);
  const resetKeyRef = useRef(resetKey);
  const wasEnabledRef = useRef(false);
  const manualFetchAvailableRef = useRef(false);
  const manualStateTimerRef = useRef<ReturnType<
    typeof globalThis.setTimeout
  > | null>(null);
  const [manualFetchAvailable, setManualFetchAvailableState] = useState(false);

  const setManualFetchAvailable = useCallback((next: boolean) => {
    manualFetchAvailableRef.current = next;
    if (manualStateTimerRef.current !== null) {
      globalThis.clearTimeout(manualStateTimerRef.current);
    }
    manualStateTimerRef.current = globalThis.setTimeout(() => {
      manualStateTimerRef.current = null;
      setManualFetchAvailableState((current) =>
        current === next ? current : next,
      );
    }, 0);
  }, []);

  const requestFetchMore = useCallback(() => {
    const accepted = fetchMoreRef.current() !== false;
    if (accepted) {
      onFetchIntentRef.current?.();
    }
    return accepted;
  }, []);

  const fetchMoreManually = useCallback(() => {
    if (!enabledRef.current || isFetchingRef.current) return;
    if (requestFetchMore()) {
      setManualFetchAvailable(false);
    }
  }, [requestFetchMore, setManualFetchAvailable]);

  useEffect(
    () => () => {
      if (manualStateTimerRef.current !== null) {
        globalThis.clearTimeout(manualStateTimerRef.current);
      }
    },
    [],
  );

  // Fires when the stable sentinel scrolls into view. Omits `signal` on purpose:
  // rebuilding per batch could drop an intersection. Refills fall to the auto-fire effect.
  useEffect(() => {
    if (!enabled) return;
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const root = scrollRef.current;
            if (
              !root ||
              isFetchingRef.current ||
              manualFetchAvailableRef.current ||
              !hasScrollableOverflow(root)
            ) {
              continue;
            }
            if (autoFireCountRef.current >= maxAutoFillFetches) {
              setManualFetchAvailable(manualFetchAfterAutoFill);
              continue;
            }
            if (requestFetchMore()) {
              autoFireCountRef.current += 1;
            }
          }
        }
      },
      {
        threshold: 0,
        root: scrollRef.current,
        rootMargin: "200px 0px",
      },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [
    enabled,
    manualFetchAfterAutoFill,
    maxAutoFillFetches,
    requestFetchMore,
    setManualFetchAvailable,
  ]);

  // Auto-fire fallback: keep requesting batches while the container doesn't yet
  // overflow (initial empty state or aggressive filters). Driven only off
  // `enabled`/`signal` and a ResizeObserver on the scroll root, so it wakes on
  // listing-shape changes rather than thrashing the main thread every frame
  // (the prior childList/subtree observer was the dominant Hub lag source).
  useEffect(() => {
    if (!enabled) {
      wasEnabledRef.current = false;
      setManualFetchAvailable(false);
      return;
    }
    // Fresh enable or a shrinking list clears the backstop so loading can refill the viewport.
    if (
      !wasEnabledRef.current ||
      signal < prevSignalRef.current ||
      resultCount < prevResultCountRef.current ||
      resetKey !== resetKeyRef.current
    ) {
      autoFireCountRef.current = 0;
      setManualFetchAvailable(false);
    } else if (resultCount > prevResultCountRef.current) {
      autoFireCountRef.current = 0;
      setManualFetchAvailable(false);
    }
    wasEnabledRef.current = true;
    prevSignalRef.current = signal;
    prevResultCountRef.current = resultCount;
    resetKeyRef.current = resetKey;

    const root = scrollRef.current;
    if (!root) return;

    const tryFire = () => {
      const sentinel = sentinelRef.current;
      if (!sentinel?.isConnected) return;
      if (manualFetchAvailableRef.current) return;
      // Once content overflows, the IntersectionObserver takes over - stop polling.
      if (hasScrollableOverflow(root)) {
        setManualFetchAvailable(false);
        return;
      }
      if (isFetchingRef.current) return;
      if (autoFireCountRef.current >= maxAutoFillFetches) {
        setManualFetchAvailable(manualFetchAfterAutoFill);
        return;
      }
      if (requestFetchMore()) {
        setManualFetchAvailable(false);
        autoFireCountRef.current += 1;
      }
    };

    let frame: number | null = null;
    const schedule = () => {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        tryFire();
      });
    };

    schedule();

    const ro = new ResizeObserver(schedule);
    ro.observe(root);

    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      ro.disconnect();
    };
  }, [
    enabled,
    isFetching,
    manualFetchAfterAutoFill,
    maxAutoFillFetches,
    resetKey,
    requestFetchMore,
    resultCount,
    setManualFetchAvailable,
    signal,
  ]);

  return {
    scrollRef,
    sentinelRef,
    manualFetchAvailable,
    fetchMoreManually,
  };
}
