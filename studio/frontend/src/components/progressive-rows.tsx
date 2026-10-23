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
/** frees a fetch that drew no rows (an error or the end), past the sidebar's render batching. */
const FETCH_SETTLE_MS = 250;
/** a scroll this soon after a page is drawn is the same gesture's own scroll, not a new one. */
const GESTURE_TAIL_MS = 200;

// one server page per user scroll, so a tall window or the preload margin never drains pages
const ARMING_EVENTS = ["scroll", "wheel", "touchstart", "touchmove"] as const;

/** preserves mounted rows when the list grows because `render` remains stable. */
const Row = memo(function Row<T>({
  item,
  render,
}: { item: T; render: (item: T) => ReactNode }) {
  return render(item);
}) as <T>(props: { item: T; render: (item: T) => ReactNode }) => ReactNode;

/** pages mounted rows to bound Radix keydown listeners; with `hasMore`, the end asks `onReachEnd` for more. */
export function ProgressiveRows<T extends { id: string }>({
  items,
  pageSize,
  renderItem,
  end,
  hasMore = false,
  onReachEnd,
  mountAll = false,
}: {
  items: readonly T[];
  pageSize: number;
  renderItem: (item: T) => ReactNode;
  end?: ReactNode;
  hasMore?: boolean;
  onReachEnd?: () => Promise<void> | void;
  /** Mount every row it is given and only page the server: for lists whose rows already arrive a
   *  page at a time, so the rows navigation and selection see are the rows on screen. */
  mountAll?: boolean;
}) {
  const [limit, setLimit] = useState(mountAll ? Number.POSITIVE_INFINITY : pageSize);
  const sentinelRef = useRef<HTMLLIElement>(null);
  const intersectingRef = useRef(false);
  const fetchingRef = useRef(false);
  const armedRef = useRef(false);
  const fetchFromRef = useRef(0);
  const itemCountRef = useRef(items.length);
  const drawnAtRef = useRef(0);
  const mountedMore = items.length > limit;
  const fetchMore = hasMore && Boolean(onReachEnd);
  const showSentinel = mountedMore || fetchMore;

  const advance = useCallback(
    (root: Element | null) => {
      if (items.length > limit) {
        setLimit((count) => count + pageSize);
        return;
      }
      if (!fetchMore || fetchingRef.current || !root || !armedRef.current) return;
      armedRef.current = false;
      fetchingRef.current = true;
      fetchFromRef.current = items.length;
      setLimit((count) => Math.max(count, items.length) + pageSize);
      void Promise.resolve(onReachEnd?.()).finally(() => {
        setTimeout(() => {
          if (itemCountRef.current <= fetchFromRef.current) fetchingRef.current = false;
        }, FETCH_SETTLE_MS);
      });
    },
    [items.length, limit, pageSize, fetchMore, onReachEnd],
  );

  // a scroll between the answer and the rows being drawn sees the old sentinel and would fetch again
  useEffect(() => {
    itemCountRef.current = items.length;
    if (fetchingRef.current && items.length > fetchFromRef.current) {
      fetchingRef.current = false;
      drawnAtRef.current = performance.now();
    }
  }, [items.length]);

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
    let lastTop = root?.scrollTop ?? 0;
    let touchY: number | null = null;
    const onArm = (event: Event) => {
      if (event.type === "touchstart") {
        touchY = (event as TouchEvent).touches[0]?.clientY ?? null;
        return;
      }
      // only a scroll down asks for more: a click's focus nudge or a scroll up does not
      if (event.type === "scroll") {
        const top = root?.scrollTop ?? 0;
        const down = top > lastTop;
        lastTop = top;
        if (!down) return;
      } else if (event.type === "wheel" && (event as WheelEvent).deltaY <= 0) {
        return;
      } else if (event.type === "touchmove") {
        // a finger moving up scrolls the list down; a swipe up the list or sideways does not
        const y = (event as TouchEvent).touches[0]?.clientY;
        const down = touchY !== null && y !== undefined && y < touchY;
        if (y !== undefined) touchY = y;
        if (!down) return;
      }
      if (fetchingRef.current || performance.now() - drawnAtRef.current < GESTURE_TAIL_MS) return;
      armedRef.current = true;
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
