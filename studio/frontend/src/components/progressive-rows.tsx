// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ReactNode,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

/** mounts the next page this far below the sidebar's visible edge. */
const PRELOAD_PX = 800;

// Scrollers the user has moved. A server page loads only after that, so a tall window never reads
// page after page at startup just because the sentinel happens to be in view.
const armedScrollers = new WeakSet<Element>();
const ARMING_EVENTS = ["scroll", "wheel", "touchmove"] as const;

/** preserves mounted rows when the list grows because `render` remains stable. */
const Row = memo(function Row<T>({
  item,
  render,
}: { item: T; render: (item: T) => ReactNode }) {
  return render(item);
}) as <T>(props: { item: T; render: (item: T) => ReactNode }) => ReactNode;

/** pages mounted rows to bound Radix keydown listeners while callers retain the loaded list. When
 *  the caller holds only part of the list (`hasMore`), reaching the end asks it for the next page
 *  through `onReachEnd`, once the user has scrolled the sidebar. */
export function ProgressiveRows<T extends { id: string }>({
  items,
  pageSize,
  renderItem,
  end,
  hasMore = false,
  onReachEnd,
}: {
  items: readonly T[];
  pageSize: number;
  renderItem: (item: T) => ReactNode;
  end?: ReactNode;
  /** More rows exist than `items` holds. */
  hasMore?: boolean;
  /** Fetches the next page; the sentinel waits for it before asking again. */
  onReachEnd?: () => Promise<void> | void;
}) {
  const [limit, setLimit] = useState(pageSize);
  const sentinelRef = useRef<HTMLLIElement>(null);
  const intersectingRef = useRef(false);
  const fetchingRef = useRef(false);
  const mountedMore = items.length > limit;
  const fetchMore = hasMore && Boolean(onReachEnd);
  const showSentinel = mountedMore || fetchMore;

  const advance = useCallback(
    (root: Element | null) => {
      if (items.length > limit) {
        setLimit((count) => count + pageSize);
        return;
      }
      if (!fetchMore || fetchingRef.current) return;
      if (!root || !armedScrollers.has(root)) return;
      fetchingRef.current = true;
      setLimit((count) => Math.max(count, items.length) + pageSize);
      void Promise.resolve(onReachEnd?.()).finally(() => {
        fetchingRef.current = false;
      });
    },
    [items.length, limit, pageSize, fetchMore, onReachEnd],
  );

  // re-observe because IntersectionObserver does not repeat while the sentinel stays intersecting
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!showSentinel || !sentinel) return;
    // use the sidebar scroller because the viewport sees its clipped sentinel only when visible
    const root = sentinel.closest<HTMLElement>("[data-sidebar='content']");
    const observer = new IntersectionObserver(
      (entries) => {
        intersectingRef.current = entries.some((entry) => entry.isIntersecting);
        if (intersectingRef.current) advance(root);
      },
      { root, rootMargin: `0px 0px ${PRELOAD_PX}px 0px` },
    );
    observer.observe(sentinel);
    // the first scroll arms the scroller and loads a sentinel that was already in view
    const onArm = () => {
      if (!root) return;
      armedScrollers.add(root);
      if (intersectingRef.current) advance(root);
    };
    const armTarget = fetchMore ? root : null;
    for (const name of ARMING_EVENTS) {
      armTarget?.addEventListener(name, onArm, { passive: true });
    }
    return () => {
      observer.disconnect();
      for (const name of ARMING_EVENTS)
        armTarget?.removeEventListener(name, onArm);
    };
  }, [showSentinel, fetchMore, advance]);
  return (
    <>
      {items.slice(0, limit).map((item) => (
        <Row key={item.id} item={item} render={renderItem} />
      ))}
      {showSentinel ? (
        <li ref={sentinelRef} aria-hidden className="h-px" />
      ) : (
        end
      )}
    </>
  );
}
