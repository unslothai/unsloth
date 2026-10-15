// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { sanitizeHubErrorMessage } from "../lib/network";

interface HfPaginatedState<T> {
  results: T[];
  scannedCount: number;
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  error: string | null;
}

interface InternalPaginatedState<T> extends HfPaginatedState<T> {
  queryKey: object | null;
}

const INITIAL: HfPaginatedState<never> = {
  results: [],
  scannedCount: 0,
  isLoading: false,
  isLoadingMore: false,
  hasMore: false,
  error: null,
};
// listModels returns up to 500 per fetch, so a larger batch just walks the in-memory page.
const BATCH = 48;
const MAX_RAW_ITEMS_PER_BATCH = BATCH * 4;
type BusyKind = "initial" | "more";

/**
 * Min gap between fetchMore() calls, since sibling observers can fire in one tick. A blocked
 * call queues a trailing-edge fire so starved filters keep paginating.
 */
const MIN_FETCH_INTERVAL_MS = 350;

// Mirrors the modelInfo TTL in hf-cache.ts.
const STALE_AFTER_MS = 5 * 60 * 1000;

export async function pullBatch<T>(
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
    // A throw from mapItem means skip; letting it out kills the generator on the same row every restart.
    let mapped: T | null = null;
    try {
      mapped = mapItem(result.value);
    } catch {
      continue;
    }
    if (mapped !== null) {
      items.push(mapped);
    }
  }
  return { items, done: false, scanned };
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

// The SDK message includes the request URL, which carries the user's search query.
function hubErrorText(err: unknown, fallback: string): string {
  return err instanceof Error
    ? sanitizeHubErrorMessage(err.message)
    : fallback;
}

function isDocumentHidden(): boolean {
  return typeof document !== "undefined" && document.hidden;
}

export function useHubPaginatedSearch<T>(
  createIter: (signal: AbortSignal) => AsyncGenerator<unknown>,
  mapItem: (raw: unknown) => T | null,
  options?: { enabled?: boolean },
): HfPaginatedState<T> & {
  fetchMore: () => boolean;
  retry: () => void;
  needsRestart: () => boolean;
} {
  const enabled = options?.enabled ?? true;
  const [retryNonce, setRetryNonce] = useState(0);
  // A thrown async generator is closed, so a failed page can only be restarted.
  const iterDeadRef = useRef(false);
  const queryKey = useMemo(
    () => ({ createIter, mapItem, retryNonce }),
    [createIter, mapItem, retryNonce],
  );
  const [state, setState] = useState<InternalPaginatedState<T>>({
    ...(INITIAL as HfPaginatedState<T>),
    queryKey: null,
  });
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const iterRef = useRef<AsyncGenerator<unknown> | null>(null);
  const versionRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  // Restart only when the query identity changes, never on `enabled`, so tab switches stay instant.
  const loadedFactoryRef = useRef<typeof createIter | null>(null);
  const loadedMapItemRef = useRef<typeof mapItem | null>(null);
  const loadedNonceRef = useRef(-1);
  const loadedAtRef = useRef(0);

  // Synchronous guard set before any setState so back-to-back calls cannot both pass.
  const busyRef = useRef(false);
  const busyKindRef = useRef<BusyKind | null>(null);
  const busyTokenRef = useRef(0);
  const lastFireAtRef = useRef(0);
  // Trailing-edge fire so a time-gated request is not lost when no later event re-triggers.
  const trailingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
        busyTokenRef.current += 1;
        abortRef.current?.abort();
        abortRef.current = null;
        iterRef.current = null;
        loadedAtRef.current = 0;
        busyRef.current = false;
        busyKindRef.current = null;
      }
      // Keep `error` so disabling the feed does not erase why the last attempt failed.
      setState((prev) =>
        prev.isLoading || prev.isLoadingMore
          ? {
              ...prev,
              isLoading: false,
              isLoadingMore: false,
            }
          : prev,
      );
      return;
    }

    // Same query, still fresh: reuse what we have. Stale results refetch.
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
    const busyToken = ++busyTokenRef.current;
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
      queryKey,
    });

    const iter = createIter(controller.signal);
    iterRef.current = iter;
    iterDeadRef.current = false;

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
          queryKey,
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
          error: hubErrorText(err, "Search failed"),
          queryKey,
        });
      })
      .finally(() => {
        if (busyTokenRef.current === busyToken) {
          busyRef.current = false;
          busyKindRef.current = null;
        }
      });

    return () => {
      clearDeferredFetch();
    };
  }, [
    createIter,
    mapItem,
    enabled,
    retryNonce,
    queryKey,
    clearDeferredFetch,
  ]);

  const needsRestart = useCallback(() => iterDeadRef.current, []);

  const retry = useCallback(() => {
    setRetryNonce((n) => n + 1);
  }, []);

  const fetchMore = useCallback(function fetchMoreInner(): boolean {
    if (!enabled) {
      queuedAfterBusyRef.current = false;
      queuedWhileHiddenRef.current = false;
      return false;
    }
    if (busyRef.current) {
      if (busyKindRef.current === "more" && stateRef.current.isLoadingMore) {
        if (queuedAfterBusyRef.current) return false;
        queuedAfterBusyRef.current = true;
        return true;
      }
      return false;
    }

    // A thrown generator would return done and clear hasMore, swallowing the error; only restart resumes.
    if (iterDeadRef.current) {
      queuedAfterBusyRef.current = false;
      queuedWhileHiddenRef.current = false;
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
    const busyToken = ++busyTokenRef.current;
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
          error: null,
        }));
      })
      .catch((err) => {
        if (versionRef.current !== v || isAbortError(err)) return;
        shouldScheduleFollowUp = false;
        // Keep rows and hasMore so the footer survives; continuing needs a restart.
        iterDeadRef.current = true;
        setState((prev) => ({
          ...prev,
          isLoadingMore: false,
          error: hubErrorText(err, "Failed to load more"),
        }));
      })
      .finally(() => {
        if (busyTokenRef.current === busyToken) {
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

  const visibleState: InternalPaginatedState<T> =
    state.queryKey === queryKey
      ? state
      : {
          ...(INITIAL as HfPaginatedState<T>),
          isLoading: enabled,
          queryKey,
        };
  return {
    results: visibleState.results,
    scannedCount: visibleState.scannedCount,
    isLoading: visibleState.isLoading,
    isLoadingMore: visibleState.isLoadingMore,
    hasMore: visibleState.hasMore,
    error: visibleState.error,
    fetchMore,
    retry,
    needsRestart,
  };
}
