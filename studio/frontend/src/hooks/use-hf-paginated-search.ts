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
): HfPaginatedState<T> & { fetchMore: () => void } {
  const enabled = options?.enabled ?? true;
  const [state, setState] = useState<HfPaginatedState<T>>(
    INITIAL as HfPaginatedState<T>,
  );
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const iterRef = useRef<AsyncGenerator<unknown> | null>(null);
  const versionRef = useRef(0);

  useEffect(() => {
    const v = ++versionRef.current;
    iterRef.current = null;

    if (!enabled) {
      setState(INITIAL as HfPaginatedState<T>);
      return;
    }

    setState({
      ...(INITIAL as HfPaginatedState<T>),
      isLoading: true,
    });

    const iter = createIter();
    iterRef.current = iter;

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done }) => {
        if (versionRef.current !== v) {
          return;
        }
        setState({
          results: items,
          isLoading: false,
          isLoadingMore: false,
          hasMore: !done,
          error: null,
        });
      })
      .catch((err) => {
        if (versionRef.current !== v) {
          return;
        }
        setState({
          results: [],
          isLoading: false,
          isLoadingMore: false,
          hasMore: false,
          error: err instanceof Error ? err.message : "Search failed",
        });
      });
  }, [createIter, mapItem, enabled]);

  const fetchMore = useCallback(() => {
    const iter = iterRef.current;
    const { isLoading, isLoadingMore, hasMore } = stateRef.current;
    if (!iter || isLoading || isLoadingMore || !hasMore) {
      return;
    }

    const v = versionRef.current;
    setState((prev) => ({ ...prev, isLoadingMore: true }));

    pullBatch(iter, mapItem, BATCH)
      .then(({ items, done }) => {
        if (versionRef.current !== v) {
          return;
        }
        setState((prev) => ({
          ...prev,
          results: [...prev.results, ...items],
          isLoadingMore: false,
          hasMore: !done,
        }));
      })
      .catch(() => {
        if (versionRef.current !== v) {
          return;
        }
        setState((prev) => ({ ...prev, isLoadingMore: false, hasMore: false }));
      });
  }, [mapItem]);

  return { ...state, fetchMore };
}
