// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo, type ReactNode, useEffect, useRef, useState } from "react";

/** How far below the sidebar's visible edge the next page is mounted, so a scroll at normal
 *  speed never reaches the end of what is mounted. */
const PRELOAD_PX = 800;

/** Lets a new page skip the rows already mounted: growing the list leaves `render` as it was. */
const Row = memo(function Row<T>({
  item,
  render,
}: { item: T; render: (item: T) => ReactNode }) {
  return render(item);
}) as <T>(props: { item: T; render: (item: T) => ReactNode }) => ReactNode;

/**
 * Mounts a long sidebar list a page at a time. Every mounted row costs work on every render and
 * on every keystroke anywhere in the app (each row's Radix menus keep a `document` listener), so
 * a history of thousands of chats made the whole UI slow. Rows past the first page mount when the
 * sentinel under the last one nears the sidebar's visible edge.
 *
 * Only the DOM is paged: callers keep passing the full list everywhere else, so selection,
 * shift-click ranges, keyboard navigation and drag order still see every row. `end` renders only
 * once every row is mounted, since anything drawn after the rows claims to be the list's end.
 * The count lives here, not in the caller, so a new page re-renders this list alone.
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
        // The sidebar scrolls itself, and a margin only stretches the root's own box: against
        // the viewport the sentinel would stay clipped by the sidebar until it was on screen.
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
