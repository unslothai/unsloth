// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";

export function useInfiniteScroll(
  fetchMore: () => void,
  itemCount: number,
  enabled = true,
) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const fetchMoreRef = useRef(fetchMore);

  useEffect(() => {
    fetchMoreRef.current = fetchMore;
  }, [fetchMore]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const sentinel = sentinelRef.current;
    const root = scrollRef.current;
    if (!sentinel) {
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          fetchMoreRef.current();
        }
      },
      { threshold: 0, root, rootMargin: "200px 0px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [enabled]);

  useEffect(() => {
    if (!enabled || itemCount === 0) {
      return;
    }
    const root = scrollRef.current;
    const sentinel = sentinelRef.current;
    if (!root || !sentinel?.isConnected) {
      return;
    }
    if (root.scrollHeight <= root.clientHeight + 4) {
      fetchMoreRef.current();
    }
  }, [itemCount, enabled]);

  return { scrollRef, sentinelRef };
}
