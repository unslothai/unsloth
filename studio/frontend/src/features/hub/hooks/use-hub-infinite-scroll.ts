// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Wires an IntersectionObserver-driven sentinel for infinite scroll, with a
 * ResizeObserver fallback that auto-fetches whenever the scroll container's
 * content doesn't yet overflow.
 *
 * Both fallback observers can fire many times per scroll burst; we coalesce
 * them into one read per animation frame so we only hit `scrollHeight`
 * (a forced-layout property) once per frame. The downstream `fetchMore` is
 * additionally rate-limited at the data-source layer, which is where the
 * real concurrency hazard (parallel pulls on the same iterator) lives.
 *
 * The `signal` parameter is an arbitrary value the caller updates whenever
 * the data layer has produced new state — typically `results.length`. Passing
 * it as a dep guarantees we re-evaluate the fit check after a fetch returns,
 * even when the page-level filter rejected every new row (in which case the
 * DOM doesn't change and the MutationObserver wouldn't fire on its own).
 *
 * The auto-fire fallback is also bounded by `DEFAULT_MAX_AUTO_FILL_FETCHES`: a
 * hard backstop on no-overflow fetches that caps a runaway sweep of the full
 * remote listing. Callers with an explicit manual continuation UI can lower the
 * cap and expose `manualFetchAvailable` once the automatic fill stops.
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
  enabledOrOptions: boolean | InfiniteScrollOptions = true,
  legacyOnFetchIntent?: () => void,
) {
  const options =
    typeof enabledOrOptions === "boolean"
      ? { enabled: enabledOrOptions, onFetchIntent: legacyOnFetchIntent }
      : enabledOrOptions;
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

  // IntersectionObserver: fires when the sentinel scrolls into view. The
  // sentinel is stable, so this deliberately omits `signal` — rebuilding per
  // batch could drop an intersection. Refills fall to the auto-fire effect.
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

  // Auto-fire fallback: keep requesting batches while the scroll container
  // doesn't overflow yet (initial empty state or aggressive filters). The
  // primary `IntersectionObserver` above handles "user scrolled near the
  // sentinel" perfectly once the list overflows, so the fallback only needs
  // to wake up when the *shape* of the listing changes — not on every frame.
  //
  // The previous implementation observed `childList:true, subtree:true` on
  // the scroll root and re-fired on every scroll event. With a virtualized
  // list that mounts/unmounts rows constantly while scrolling, that was a
  // per-frame `scrollHeight/scrollTop/clientHeight` read storm (three forced
  // layouts every frame), and it was the dominant lag source on the Hub.
  //
  // We now drive the fallback purely off:
  //   - `enabled` / `signal` (caller signals new data or fresh enable)
  //   - a ResizeObserver on the scroll root (window resize / layout change)
  // This is enough to cover the no-overflow case without thrashing the main
  // thread during normal scrolling.
  useEffect(() => {
    if (!enabled) {
      wasEnabledRef.current = false;
      setManualFetchAvailable(false);
      return;
    }
    // A fresh enable (new search, or the user resuming after a paused sweep)
    // or a shrinking list (results reset) clears the backstop so legitimate
    // loading can refill the viewport.
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
      // Once the content actually overflows the viewport, the
      // IntersectionObserver takes over — no need to keep polling.
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
