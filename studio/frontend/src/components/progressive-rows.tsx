// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo, type ReactNode, useEffect, useRef, useState } from "react";

/** How far below the sidebar's visible edge the next page mounts. */
const PRELOAD_PX = 800;

/** Lets a new page skip the rows already mounted: growing the list leaves `render` as it was. */
const Row = memo(function Row<T>({
  item,
  render,
}: { item: T; render: (item: T) => ReactNode }) {
  return render(item);
}) as <T>(props: { item: T; render: (item: T) => ReactNode }) => ReactNode;

/**
 * Mounts a long sidebar list a page at a time: each mounted chat row's Radix menus add a
 * `document` keydown listener, so every keystroke in the app cost time in proportion to the rows.
 * Only the DOM is paged; callers keep the full list for selection, ranges and drag order. `end`
 * renders only once every row is mounted, since it claims to be the list's end. The count lives
 * here so a new page re-renders this list alone.
 */
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
  // Re-observes after each page: an observer reports a change of intersection, not a steady
  // one, so a sentinel still in range after a page would never ask for the next.
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
        // The sidebar scrolls itself: against the viewport the sentinel is clipped until shown.
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
