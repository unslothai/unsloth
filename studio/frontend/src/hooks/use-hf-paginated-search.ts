// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

interface HfPaginatedState<T> {
  results: T[];
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  error: string | null;
}

const INITIAL: HfPaginatedState<never> = {
  results: [],
  isLoading: false,
  isLoadingMore: false,
  hasMore: false,
  error: null,
};
const BATCH = 20;

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
const MIN_FETCH_INTERVAL_MS = 500;

async function pullBatch<T>(
  iter: AsyncGenerator<unknown>,
  mapItem: (raw: unknown) => T | null,
  size: number,
) {
  const items: T[] = [];
  while (items.length < size) {
    const result = await iter.next();
    if (result.done) {
      return { items, done: true };
    }
    const mapped = mapItem(result.value);
    if (mapped !== null) {
      items.push(mapped);
    }
  }
  return { items, done: false };
}

export function useHfPaginatedSearch<T>(
  createIter: () => AsyncGenerator<unknown>,
  mapItem: (raw: unknown) => T | null,
  options?: { enabled?: boolean },
): HfPaginatedState<T> & { fetchMore: () => void; retry: () => void } {
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

  // Synchronous in-flight guard. Set before any setState so back-to-back
  // fetchMore() calls cannot both pass the gate while React batches the
  // isLoadingMore commit. Cleared in finally() of the matching pull.
  const busyRef = useRef(false);
  // Wall-clock timestamp of the last accepted request (initial or fetchMore).
  // Together with the trailing-edge timer below, enforces at most one accepted
  // request per MIN_FETCH_INTERVAL_MS regardless of how many times observers
  // call fetchMore in the meantime.
  const lastFireAtRef = useRef(0);
  // Pending trailing-edge fire. A fetchMore() call blocked by the time gate
  // schedules one of these (idempotent — only one can be queued at a time)
  // so we don't lose the request when no later DOM event would re-trigger us.
  const trailingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelTrailing = useCallback(() => {
    if (trailingTimerRef.current !== null) {
      clearTimeout(trailingTimerRef.current);
      trailingTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    const v = ++versionRef.current;
    iterRef.current = null;
    busyRef.current = false;
    cancelTrailing();

    if (!enabled) {
      setState(INITIAL as HfPaginatedState<T>);
      return;
    }

    busyRef.current = true;
    lastFireAtRef.current = Date.now();
    setState({
      ...(INITIAL as HfPaginatedState<T>),
      isLoading: true,
    });

    const iter = createIter();
    iterRef.current = iter;

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done }) => {
        if (versionRef.current !== v) return;
        setState({
          results: items,
          isLoading: false,
          isLoadingMore: false,
          hasMore: !done,
          error: null,
        });
      })
      .catch((err) => {
        if (versionRef.current !== v) return;
        setState({
          results: [],
          isLoading: false,
          isLoadingMore: false,
          hasMore: false,
          error: err instanceof Error ? err.message : "Search failed",
        });
      })
      .finally(() => {
        if (versionRef.current === v) {
          busyRef.current = false;
        }
      });

    return () => {
      cancelTrailing();
    };
  }, [createIter, mapItem, enabled, retryNonce, cancelTrailing]);

  const retry = useCallback(() => {
    setRetryNonce((n) => n + 1);
  }, []);

  const fetchMore = useCallback(() => {
    // Synchronous in-flight gate. Runs before any setState so concurrent
    // fires from sibling observers all see the same truth and only one
    // proceeds — this closes the race window React's batched commit opens.
    if (busyRef.current) return;

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
            fetchMore();
          },
          MIN_FETCH_INTERVAL_MS - elapsed,
        );
      }
      return;
    }

    const iter = iterRef.current;
    const { hasMore } = stateRef.current;
    if (!iter || !hasMore) return;

    cancelTrailing();
    busyRef.current = true;
    lastFireAtRef.current = now;

    const v = versionRef.current;
    setState((prev) => ({ ...prev, isLoadingMore: true }));

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done }) => {
        if (versionRef.current !== v) return;
        setState((prev) => ({
          ...prev,
          results: [...prev.results, ...items],
          isLoadingMore: false,
          hasMore: !done,
        }));
      })
      .catch((err) => {
        if (versionRef.current !== v) return;
        setState((prev) => ({
          ...prev,
          isLoadingMore: false,
          // Keep hasMore=true so the user can recover via retry() once the
          // transient error (rate limit, network blip) clears.
          error: err instanceof Error ? err.message : "Failed to load more",
        }));
      })
      .finally(() => {
        if (versionRef.current === v) {
          busyRef.current = false;
        }
      });
  }, [mapItem, cancelTrailing]);

  return { ...state, fetchMore, retry };
}
