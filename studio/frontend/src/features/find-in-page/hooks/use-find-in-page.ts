// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// DOM writes are limited to highlights so the observer cannot retrigger itself.

import { completeProgressiveMounts } from "@/components/assistant-ui/progressive-messages";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  cancelRevealPasses,
  clearHighlights,
  indexReaches,
  mutatesSearchableText,
  paintHighlights,
  paintWindow,
  rangeForMatch,
  rangeTop,
  resolveFindScope,
  resolvePortalSurfaces,
  revealRangeWhenPainted,
  scrollViewportTop,
  selectRangeFallback,
  supportsHighlightApi,
  viewportOffset,
} from "../lib/find-dom.ts";
import {
  EMPTY_TEXT_INDEX,
  FIND_SKIP_ATTRIBUTE,
  type FindElementLike,
  type FindMatch,
  type FindTextIndex,
  MAX_MATCHES,
  buildTextIndex,
  dropProbeFurthestFrom,
  findMatches,
  renumbersMatches,
} from "../lib/find-text-index.ts";

/**
 * Shortest gap between two rebuilds while the document is changing.
 *
 * A throttle rather than a debounce: a reply streams for as long as it takes to write, and a
 * debounce would leave the count frozen and the new text unfindable for that whole time. This
 * bounds the cost instead, at one flatten per interval however fast the tokens arrive.
 */
export const REINDEX_INTERVAL_MS = 300;

export interface FindResults {
  count: number;
  active: number;
  /** True when the cap cut something off, so `count` is a floor. */
  capped: boolean;
  truncated: boolean;
  next: () => void;
  previous: () => void;
}

/** First match at or below the viewport top; binary search since rects follow doc order. */
function firstMatchFromViewport(
  index: FindTextIndex,
  matches: FindMatch[],
): number {
  if (matches.length === 0) return -1;
  let lo = 0;
  let hi = matches.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const range = rangeForMatch(index, matches[mid]);
    if (!range) return 0;
    const top = rangeTop(range);
    if (top === null) return 0;
    if (top >= scrollViewportTop(range)) {
      found = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return found === -1 ? 0 : found;
}

function ordinalOfStart(matches: FindMatch[], start: number): number {
  let lo = 0;
  let hi = matches.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const at = matches[mid].start;
    if (at === start) return mid;
    if (at < start) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

export function useFindInPage(
  query: string,
  queryPending = false,
): FindResults {
  const [results, setResults] = useState<{
    count: number;
    active: number;
    capped: boolean;
    truncated: boolean;
  }>({ count: 0, active: -1, capped: false, truncated: false });

  const scopeRef = useRef<Element | null>(null);
  const indexRef = useRef<FindTextIndex>(EMPTY_TEXT_INDEX);
  const matchesRef = useRef<FindMatch[]>([]);
  /** The active occurrence's offset; ordinals are not stable once the capped window slides. */
  const activeStartRef = useRef<number | null>(null);
  const cappedRef = useRef(false);
  const activeRef = useRef(-1);
  const queryRef = useRef(query);
  const staleRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const queryPendingRef = useRef(queryPending);
  queryPendingRef.current = queryPending;

  /** The only place highlights and React state are written. */
  const apply = useCallback((reveal: boolean) => {
    if (queryPendingRef.current) return;
    const index = indexRef.current;
    const matches = matchesRef.current;
    const count = matches.length;
    let active = activeRef.current;
    if (count === 0) active = -1;
    else if (active < 0) active = 0;
    else if (active >= count) active = count - 1;
    activeRef.current = active;

    const activeRange =
      active >= 0 ? rangeForMatch(index, matches[active]) : null;

    if (supportsHighlightApi()) {
      const window_ = paintWindow(count, Math.max(active, 0));
      const ranges: Range[] = [];
      for (let i = window_.from; i < window_.to; i += 1) {
        const range =
          i === active ? activeRange : rangeForMatch(index, matches[i]);
        if (range) ranges.push(range);
      }
      paintHighlights(ranges, activeRange);
    } else {
      selectRangeFallback(activeRange);
    }

    if (reveal && activeRange) revealRangeWhenPainted(activeRange);

    activeStartRef.current = active >= 0 ? matches[active].start : null;

    const capped = cappedRef.current;
    setResults((previous) =>
      previous.count === count &&
      previous.active === active &&
      previous.capped === capped &&
      previous.truncated === index.truncated
        ? previous
        : { count, active, capped, truncated: index.truncated },
    );
  }, []);

  const search = useCallback(
    (reveal: boolean, fromViewport: boolean) => {
      const index = indexRef.current;
      // Search one past the cap to tell a floor from the total. The anchor is a thunk so layout
      // is read only when the cap is hit.
      let anchoredAt: number | null = null;
      const matches = findMatches(
        index,
        queryRef.current,
        MAX_MATCHES + 1,
        () => {
          anchoredAt = viewportOffset(index);
          return anchoredAt;
        },
      );
      cappedRef.current = matches.length > MAX_MATCHES;
      if (cappedRef.current) dropProbeFurthestFrom(matches, anchoredAt);
      // Read before `apply` writes it.
      const wasAt = activeStartRef.current;
      matchesRef.current = matches;
      if (fromViewport) {
        activeRef.current = firstMatchFromViewport(index, matches);
      } else {
        // Match by offset, valid only because the document merely grew at the tail.
        const at = wasAt === null ? -1 : ordinalOfStart(matches, wasAt);
        activeRef.current =
          at === -1 ? firstMatchFromViewport(index, matches) : at;
      }
      apply(reveal);
    },
    [apply],
  );

  /** Rebuild the index; true when matches were renumbered (anything but a tail append),
   *  which means re-anchoring to the viewport. */
  const reindex = useCallback((): boolean => {
    staleRef.current = false;
    const before = indexRef.current;
    const scope = scopeRef.current;
    indexRef.current = scope
      ? buildTextIndex(
          scope as unknown as FindElementLike,
          // Resolved every time: popovers open and close under an open bar.
          resolvePortalSurfaces(scope) as unknown as FindElementLike[],
        )
      : EMPTY_TEXT_INDEX;
    return renumbersMatches(before, indexRef.current, activeStartRef.current);
  }, []);

  useEffect(() => {
    const scope = resolveFindScope();
    scopeRef.current = scope;
    reindex();
    // A fresh open always starts from the reader, whatever the index says.
    search(false, true);

    // The thread mounts its tail first, so complete progressive mounts for this scope only.
    let live = true;
    void completeProgressiveMounts((viewport) =>
      indexReaches(scope, viewport),
    ).then(() => {
      if (!live) return;
      // Only re-anchor if completion actually renumbered, so a mid-run Enter is kept.
      search(false, reindex());
    });

    // No reveal: something moved under the reader, they did not ask to go anywhere.
    const invalidate = () => {
      staleRef.current = true;
      if (queryRef.current.length === 0) return;
      if (timerRef.current !== null) return;
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        if (!staleRef.current) return;
        search(false, reindex());
      }, REINDEX_INTERVAL_MS);
    };

    // Breakpoints hide columns without DOM mutations, so resize must invalidate too.
    window.addEventListener("resize", invalidate);

    // Container queries change without a window resize, so observe the scope's size.
    let sized: ResizeObserver | null = null;
    if (scope && typeof ResizeObserver !== "undefined") {
      // The initial observe callback reports the existing size; skip it.
      let measured = false;
      sized = new ResizeObserver(() => {
        if (!measured) {
          measured = true;
          return;
        }
        invalidate();
      });
      sized.observe(scope);
    }

    // Observe body, not scope: portaled surfaces are siblings of the shell.
    const watched = scope?.ownerDocument?.body ?? scope;
    let observer: MutationObserver | null = null;
    if (watched && typeof MutationObserver !== "undefined") {
      observer = new MutationObserver((records) => {
        // The bar lives inside the scope, so ignore its own counter re-renders.
        if (!records.some(mutatesSearchableText)) return;
        invalidate();
      });
      observer.observe(watched, {
        childList: true,
        subtree: true,
        characterData: true,
        // Workspace switches flip `inert`; filtered since `class` changes on every hover.
        attributes: true,
        // `open` toggles <details>; `data-state` marks closing popovers, menus and accordions.
        attributeFilter: [
          "inert",
          "hidden",
          "aria-hidden",
          "open",
          "data-state",
          FIND_SKIP_ATTRIBUTE,
        ],
      });
    }

    return () => {
      live = false;
      cancelRevealPasses();
      window.removeEventListener("resize", invalidate);
      sized?.disconnect();
      observer?.disconnect();
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      clearHighlights();
      if (!supportsHighlightApi()) selectRangeFallback(null);
      indexRef.current = EMPTY_TEXT_INDEX;
      matchesRef.current = [];
    };
  }, [reindex, search]);

  // Repaint immediately on input; the settled search is coalesced so typing never stalls.
  useLayoutEffect(() => {
    if (queryPending) {
      cancelRevealPasses();
      clearHighlights();
      if (!supportsHighlightApi()) selectRangeFallback(null);
      return;
    }
    if (queryRef.current === query) apply(false);
  }, [apply, query, queryPending]);

  useEffect(() => {
    queryRef.current = query;
    if (staleRef.current) reindex();
    search(true, true);
  }, [query, reindex, search]);

  const step = useCallback(
    (delta: number) => {
      const count = matchesRef.current.length;
      if (count === 0) return;
      activeRef.current = (activeRef.current + delta + count) % count;
      apply(true);
    },
    [apply],
  );

  const next = useCallback(() => step(1), [step]);
  const previous = useCallback(() => step(-1), [step]);

  return { ...results, next, previous };
}
