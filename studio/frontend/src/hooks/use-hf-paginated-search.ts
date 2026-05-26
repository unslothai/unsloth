// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

interface HfPaginatedState<T> {
  results: T[];
  scannedCount: number;
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  error: string | null;
}

const INITIAL: HfPaginatedState<never> = {
  results: [],
  scannedCount: 0,
  isLoading: false,
  isLoadingMore: false,
  hasMore: false,
  error: null,
};
const BATCH = 20;
const MAX_RAW_ITEMS_PER_BATCH = BATCH * 4;
type BusyKind = "initial" | "more";

/**
 * Minimum gap between consecutive fetchMore() calls. Two observers (intersection
 * + mutation/resize) can fire in the same tick during a scroll burst, and React
 * commits the in-flight flag asynchronously, so this caps the worst case at one
 * network request per window.
 *
 * Pairs with a trailing-edge schedule: a call blocked by this gate doesn't
 * vanish — it queues one fire at the end of the window so filters that
 * starve the visible list (e.g. user picks a GGUF-only filter and every
 * incoming raw row is non-GGUF) keep paginating instead of dead-locking.
 */
const MIN_FETCH_INTERVAL_MS = 1000;

// Preserved results older than this are refetched on re-enable so the feed
// can't lag behind the Hub. Reset by every successful pull, so only idle time
// counts. Mirrors the modelInfo TTL in hf-cache.ts.
const STALE_AFTER_MS = 5 * 60 * 1000;

async function pullBatch<T>(
  iter: AsyncGenerator<unknown>,
  mapItem: (raw: unknown) => T | null,
  size: number,
) {
  const items: T[] = [];
  let scanned = 0;
  while (items.length < size && scanned < MAX_RAW_ITEMS_PER_BATCH) {
    const result = await iter.next();
    if (result.done) {
      return { items, done: true, scanned };
    }
    scanned += 1;
    const mapped = mapItem(result.value);
    if (mapped !== null) {
      items.push(mapped);
    }
  }
  return { items, done: false, scanned };
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function isDocumentHidden(): boolean {
  return typeof document !== "undefined" && document.hidden;
}

export function useHfPaginatedSearch<T>(
  createIter: (signal: AbortSignal) => AsyncGenerator<unknown>,
  mapItem: (raw: unknown) => T | null,
  options?: { enabled?: boolean },
): HfPaginatedState<T> & { fetchMore: () => boolean; retry: () => void } {
  const enabled = options?.enabled ?? true;
  const [state, setState] = useState<HfPaginatedState<T>>(
    INITIAL as HfPaginatedState<T>,
  );
  const [retryNonce, setRetryNonce] = useState(0);
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const iterRef = useRef<AsyncGenerator<unknown> | null>(null);
  const versionRef = useRef(0);
  // Aborts the live iterator's in-flight page fetches. Replaced (prior one
  // aborted) when a new query supersedes the feed, so an abandoned listing
  // stops fetching and priming the cache instead of running to completion.
  const abortRef = useRef<AbortController | null>(null);

  // Identity of the query we last fetched. A fetch (re)starts only when one of
  // these changes, never just because `enabled` toggled — that's what keeps tab
  // switches instant instead of refetching the whole feed each time.
  const loadedFactoryRef = useRef<typeof createIter | null>(null);
  const loadedMapItemRef = useRef<typeof mapItem | null>(null);
  const loadedNonceRef = useRef(-1);
  const loadedAtRef = useRef(0);

  // Synchronous in-flight guard. Set before any setState so back-to-back
  // fetchMore() calls cannot both pass the gate while React batches the
  // isLoadingMore commit. Cleared in finally() of the matching pull.
  const busyRef = useRef(false);
  const busyKindRef = useRef<BusyKind | null>(null);
  // Wall-clock timestamp of the last accepted request (initial or fetchMore).
  // Together with the trailing-edge timer below, enforces at most one accepted
  // request per MIN_FETCH_INTERVAL_MS regardless of how many times observers
  // call fetchMore in the meantime.
  const lastFireAtRef = useRef(0);
  // Pending trailing-edge fire. A fetchMore() call blocked by the time gate
  // schedules one of these (idempotent — only one can be queued at a time)
  // so we don't lose the request when no later DOM event would re-trigger us.
  const trailingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // One request that arrived while a "load more" pull was already in flight.
  // Scroll observers can fire repeatedly at the bottom, but only one follow-up
  // page should wait behind the active pull.
  const queuedAfterBusyRef = useRef(false);
  const queuedWhileHiddenRef = useRef(false);

  const cancelTrailing = useCallback(() => {
    if (trailingTimerRef.current !== null) {
      clearTimeout(trailingTimerRef.current);
      trailingTimerRef.current = null;
    }
  }, []);

  const clearDeferredFetch = useCallback(() => {
    cancelTrailing();
    queuedAfterBusyRef.current = false;
    queuedWhileHiddenRef.current = false;
  }, [cancelTrailing]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      clearDeferredFetch();
    },
    [clearDeferredFetch],
  );

  useEffect(() => {
    if (!enabled) {
      clearDeferredFetch();
      if (busyRef.current) {
        versionRef.current += 1;
        abortRef.current?.abort();
        abortRef.current = null;
        iterRef.current = null;
        loadedAtRef.current = 0;
        busyRef.current = false;
        busyKindRef.current = null;
      }
      setState((prev) =>
        prev.isLoading || prev.isLoadingMore || prev.error
          ? {
              ...prev,
              isLoading: false,
              isLoadingMore: false,
              error: null,
            }
          : prev,
      );
      return;
    }

    // Same query, still fresh: reuse what we have. Stale results fall through
    // and refetch so the feed can't lag behind the Hub.
    const sameQuery =
      loadedFactoryRef.current === createIter &&
      loadedMapItemRef.current === mapItem &&
      loadedNonceRef.current === retryNonce;
    const fresh = Date.now() - loadedAtRef.current < STALE_AFTER_MS;
    if (sameQuery && fresh && iterRef.current !== null) {
      return;
    }
    loadedFactoryRef.current = createIter;
    loadedMapItemRef.current = mapItem;
    loadedNonceRef.current = retryNonce;

    const v = ++versionRef.current;
    iterRef.current = null;
    busyRef.current = true;
    busyKindRef.current = "initial";
    clearDeferredFetch();
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    lastFireAtRef.current = Date.now();
    setState({
      ...(INITIAL as HfPaginatedState<T>),
      isLoading: true,
    });

    const iter = createIter(controller.signal);
    iterRef.current = iter;

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done, scanned }) => {
        if (versionRef.current !== v) return;
        loadedAtRef.current = Date.now();
        setState({
          results: items,
          scannedCount: scanned,
          isLoading: false,
          isLoadingMore: false,
          hasMore: !done,
          error: null,
        });
      })
      .catch((err) => {
        if (versionRef.current !== v || isAbortError(err)) return;
        setState({
          results: [],
          scannedCount: 0,
          isLoading: false,
          isLoadingMore: false,
          hasMore: false,
          error: err instanceof Error ? err.message : "Search failed",
        });
      })
      .finally(() => {
        if (versionRef.current === v) {
          busyRef.current = false;
          busyKindRef.current = null;
        }
      });

    return () => {
      clearDeferredFetch();
    };
  }, [createIter, mapItem, enabled, retryNonce, clearDeferredFetch]);

  const retry = useCallback(() => {
    setRetryNonce((n) => n + 1);
  }, []);

  const fetchMore = useCallback(function fetchMoreInner(): boolean {
    if (!enabled) {
      queuedAfterBusyRef.current = false;
      queuedWhileHiddenRef.current = false;
      return false;
    }
    // Synchronous in-flight gate. Runs before any setState so concurrent
    // fires from sibling observers all see the same truth and only one
    // proceeds — this closes the race window React's batched commit opens.
    if (busyRef.current) {
      if (busyKindRef.current === "more" && stateRef.current.isLoadingMore) {
        if (queuedAfterBusyRef.current) return false;
        queuedAfterBusyRef.current = true;
        return true;
      }
      return false;
    }

    const iter = iterRef.current;
    const { hasMore } = stateRef.current;
    if (!iter || !hasMore) {
      queuedAfterBusyRef.current = false;
      queuedWhileHiddenRef.current = false;
      return false;
    }

    if (isDocumentHidden()) {
      queuedAfterBusyRef.current = false;
      if (queuedWhileHiddenRef.current) return false;
      queuedWhileHiddenRef.current = true;
      return true;
    }

    const now = Date.now();
    const elapsed = now - lastFireAtRef.current;

    if (elapsed < MIN_FETCH_INTERVAL_MS) {
      // Trailing-edge schedule. Multiple calls during the window collapse to
      // one timer; clearing happens implicitly when fetchMore proceeds (and
      // explicitly on iterator reset / unmount).
      if (trailingTimerRef.current === null) {
        trailingTimerRef.current = setTimeout(
          () => {
            trailingTimerRef.current = null;
            fetchMoreInner();
          },
          MIN_FETCH_INTERVAL_MS - elapsed,
        );
        return true;
      }
      return false;
    }

    cancelTrailing();
    queuedAfterBusyRef.current = false;
    queuedWhileHiddenRef.current = false;
    busyRef.current = true;
    busyKindRef.current = "more";
    lastFireAtRef.current = now;

    const v = versionRef.current;
    let shouldScheduleFollowUp = false;
    setState((prev) => ({ ...prev, isLoadingMore: true }));

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done, scanned }) => {
        if (versionRef.current !== v) return;
        shouldScheduleFollowUp = !done && queuedAfterBusyRef.current;
        loadedAtRef.current = Date.now();
        setState((prev) => ({
          ...prev,
          results: [...prev.results, ...items],
          scannedCount: prev.scannedCount + scanned,
          isLoadingMore: false,
          hasMore: !done,
          // Clear any error left by a prior failed page now that one succeeded.
          error: null,
        }));
      })
      .catch((err) => {
        if (versionRef.current !== v || isAbortError(err)) return;
        shouldScheduleFollowUp = false;
        setState((prev) => ({
          ...prev,
          isLoadingMore: false,
          // Keep the accumulated results and hasMore=true: the same iterator is
          // still valid, so the next fetchMore() resumes the page that failed
          // without discarding the list. (retry() instead restarts from page 1.)
          error: err instanceof Error ? err.message : "Failed to load more",
        }));
      })
      .finally(() => {
        if (versionRef.current === v) {
          busyRef.current = false;
          busyKindRef.current = null;
          if (
            shouldScheduleFollowUp &&
            trailingTimerRef.current === null
          ) {
            queuedAfterBusyRef.current = false;
            const elapsed = Date.now() - lastFireAtRef.current;
            trailingTimerRef.current = setTimeout(
              () => {
                trailingTimerRef.current = null;
                fetchMoreInner();
              },
              Math.max(0, MIN_FETCH_INTERVAL_MS - elapsed),
            );
          }
        }
      });
    return true;
  }, [enabled, mapItem, cancelTrailing]);

  useEffect(() => {
    if (!enabled || typeof document === "undefined") return;
    const handleVisibilityChange = () => {
      if (document.hidden || !queuedWhileHiddenRef.current) return;
      queuedWhileHiddenRef.current = false;
      fetchMore();
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [enabled, fetchMore]);

  return { ...state, fetchMore, retry };
}
