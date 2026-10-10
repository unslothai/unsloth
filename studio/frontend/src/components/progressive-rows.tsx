// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo, type ReactNode, useEffect, useRef, useState } from "react";

/** mounts the next page this far below the sidebar's visible edge. */
const PRELOAD_PX = 800;

/** preserves mounted rows when the list grows because `render` remains stable. */
const Row = memo(function Row<T>({
  item,
  render,
}: { item: T; render: (item: T) => ReactNode }) {
  return render(item);
}) as <T>(props: { item: T; render: (item: T) => ReactNode }) => ReactNode;

/** pages mounted rows to bound Radix keydown listeners while callers retain the full list state. */
export function ProgressiveRows<T extends { id: string }>({
  items,
  pageSize,
  renderItem,
  end,
}: {
  items: readonly T[];
  pageSize: number;
  renderItem: (item: T) => ReactNode;
  end?: ReactNode;
}) {
  const [limit, setLimit] = useState(pageSize);
  const sentinelRef = useRef<HTMLLIElement>(null);
  const hasMore = items.length > limit;
  // re-observe because IntersectionObserver does not repeat while the sentinel stays intersecting
  // biome-ignore lint/correctness/useExhaustiveDependencies: `limit` is the re-observe trigger
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!hasMore || !sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setLimit((count) => count + pageSize);
        }
      },
      {
        // use the sidebar scroller because the viewport sees its clipped sentinel only when visible
        root: sentinel.closest<HTMLElement>("[data-sidebar='content']"),
        rootMargin: `0px 0px ${PRELOAD_PX}px 0px`,
      },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, limit, pageSize]);
  return (
    <>
      {items.slice(0, limit).map((item) => (
        <Row key={item.id} item={item} render={renderItem} />
      ))}
      {hasMore ? <li ref={sentinelRef} aria-hidden className="h-px" /> : end}
    </>
  );
}
