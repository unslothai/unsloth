// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The find engine. Mounted only while the bar is open, so a Studio nobody is searching runs none of
// this: no observer, no index, no ranges.
//
// Everything expensive lives in refs and is rebuilt only when the document changes. A keystroke
// runs `indexOf` over a string and creates at most `MAX_PAINTED_RANGES` ranges. Nothing here writes
// to the DOM, which is why the observer below cannot retrigger itself.

import { completeProgressiveMounts } from "@/components/assistant-ui/progressive-messages";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearHighlights,
  paintHighlights,
  paintWindow,
  rangeForMatch,
  resolveFindScope,
  rangeTop,
  scrollRangeIntoView,
  scrollViewportTop,
  selectRangeFallback,
  supportsHighlightApi,
} from "../lib/find-dom.ts";
import {
  EMPTY_TEXT_INDEX,
  FIND_SKIP_ATTRIBUTE,
  type FindElementLike,
  type FindMatch,
  type FindTextIndex,
  buildTextIndex,
  findMatches,
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
  /** Matches for the current query, capped at `MAX_MATCHES`. */
  count: number;
  /** Zero-based position in `count`, or -1 when nothing matches. */
  active: number;
  /** True when the document was too large to flatten in full. */
  truncated: boolean;
  next: () => void;
  previous: () => void;
}

/**
 * The first match at or below the top of the viewport, so a fresh query starts where the reader is
 * rather than at the top of the conversation.
 *
 * Binary search, not a scan: matches come out in document order, so their rects are ordered too.
 * Absolutely positioned content can break that; being one match out is the whole cost.
 */
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
  // Everything is above the fold: wrap to the first match.
  return found === -1 ? 0 : found;
}

/**
 * True when a mutation touched text the index covers, rather than the bar's own chrome. `some()`
 * short-circuits on the first qualifying record, so this costs a `closest` call only on the batches
 * the bar itself produced.
 */
function mutatesSearchableText(record: MutationRecord): boolean {
  const target = record.target;
  const element =
    target.nodeType === 1
      ? (target as Element)
      : (target.parentElement ?? null);
  if (!element) return true;
  return element.closest(`[${FIND_SKIP_ATTRIBUTE}]`) === null;
}

export function useFindInPage(query: string): FindResults {
  const [results, setResults] = useState<{
    count: number;
    active: number;
    truncated: boolean;
  }>({ count: 0, active: -1, truncated: false });

  const scopeRef = useRef<Element | null>(null);
  const indexRef = useRef<FindTextIndex>(EMPTY_TEXT_INDEX);
  const matchesRef = useRef<FindMatch[]>([]);
  const activeRef = useRef(-1);
  const queryRef = useRef(query);
  /** The document changed since the index was built. */
  const staleRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Clamp the active match, paint, and optionally move the reader to it. The only place highlights
   *  and React state are written, so a navigation and a rebuild take the same path. */
  const apply = useCallback((reveal: boolean) => {
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

    if (reveal && activeRange) scrollRangeIntoView(activeRange);

    setResults((previous) =>
      previous.count === count &&
      previous.active === active &&
      previous.truncated === index.truncated
        ? previous
        : { count, active, truncated: index.truncated },
    );
  }, []);

  /** Re-run the query against the index already in hand. */
  const search = useCallback(
    (reveal: boolean, fromViewport: boolean) => {
      const index = indexRef.current;
      const matches = findMatches(index, queryRef.current);
      matchesRef.current = matches;
      if (fromViewport) {
        activeRef.current = firstMatchFromViewport(index, matches);
      }
      apply(reveal);
    },
    [apply],
  );

  /** Flatten the document again, then re-run the query. */
  const rebuild = useCallback(
    (reveal: boolean, fromViewport: boolean) => {
      staleRef.current = false;
      const scope = scopeRef.current;
      indexRef.current = scope
        ? buildTextIndex(scope as unknown as FindElementLike)
        : EMPTY_TEXT_INDEX;
      search(reveal, fromViewport);
    },
    [search],
  );

  // Open: take the index and watch for changes. Closing tears all of it down, highlights included.
  useEffect(() => {
    const scope = resolveFindScope();
    scopeRef.current = scope;
    rebuild(false, true);

    // The thread mounts its tail first and widens over the next few frames, so a search against the
    // document as found would miss everything above the fold. This asks for the rest and re-indexes
    // once it has painted; the bar is usable throughout.
    let live = true;
    void completeProgressiveMounts().then(() => {
      if (!live) return;
      rebuild(false, queryRef.current.length === 0);
    });

    let observer: MutationObserver | null = null;
    if (scope && typeof MutationObserver !== "undefined") {
      observer = new MutationObserver((records) => {
        // The bar floats inside the region it searches, so its own counter re-rendering is a
        // mutation of the scope. Without this it re-indexes every time the match count changes.
        if (!records.some(mutatesSearchableText)) return;
        staleRef.current = true;
        // Nothing to re-run against an empty query, so streaming into an unused bar only sets a flag.
        if (queryRef.current.length === 0) return;
        // Already scheduled: the interval is the floor, so a burst costs one rebuild, not one each.
        if (timerRef.current !== null) return;
        timerRef.current = setTimeout(() => {
          timerRef.current = null;
          if (!staleRef.current) return;
          // No reveal: a reply arrived, the reader did not ask to move. The ordinal is kept, which
          // is right for the append-only growth streaming produces.
          rebuild(false, false);
        }, REINDEX_INTERVAL_MS);
      });
      observer.observe(scope, {
        childList: true,
        subtree: true,
        characterData: true,
        // Switching between two workspaces the shell keeps alive adds and removes nothing: it flips
        // `inert` on one panel and off the other. A filter, not `attributes: true`, because `class`
        // changes on every hover.
        attributes: true,
        attributeFilter: [
          "inert",
          "hidden",
          "aria-hidden",
          FIND_SKIP_ATTRIBUTE,
        ],
      });
    }

    return () => {
      live = false;
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
  }, [rebuild]);

  // A new query starts from the reader's position, re-indexing first if the document moved on.
  useEffect(() => {
    queryRef.current = query;
    if (staleRef.current) rebuild(true, true);
    else search(true, true);
  }, [query, rebuild, search]);

  const step = useCallback(
    (delta: number) => {
      const count = matchesRef.current.length;
      if (count === 0) return;
      // Wraps, so the walk never dead-ends.
      activeRef.current = (activeRef.current + delta + count) % count;
      apply(true);
    },
    [apply],
  );

  const next = useCallback(() => step(1), [step]);
  const previous = useCallback(() => step(-1), [step]);

  return { ...results, next, previous };
}
