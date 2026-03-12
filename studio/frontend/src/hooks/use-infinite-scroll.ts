// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";

export function useInfiniteScroll(fetchMore: () => void, _itemCount: number) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) {
      return;
    }
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          fetchMore();
        }
      },
      { threshold: 0, root: scrollRef.current },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [fetchMore]);

  return { scrollRef, sentinelRef };
}
