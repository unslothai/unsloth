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

// one server page per user scroll, so a tall window or the preload margin never drains pages
const ARMING_EVENTS = ["scroll", "wheel", "touchmove"] as const;

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
}: {
  items: readonly T[];
  pageSize: number;
  renderItem: (item: T) => ReactNode;
  end?: ReactNode;
  hasMore?: boolean;
  onReachEnd?: () => Promise<void> | void;
}) {
  const [limit, setLimit] = useState(pageSize);
  const sentinelRef = useRef<HTMLLIElement>(null);
  const intersectingRef = useRef(false);
  const fetchingRef = useRef(false);
  const armedRef = useRef(false);
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
    const onArm = () => {
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
