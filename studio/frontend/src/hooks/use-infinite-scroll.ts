// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";

/**
 * Wires an IntersectionObserver-driven sentinel for infinite scroll, with a
 * ResizeObserver+MutationObserver fallback that auto-fetches whenever the
 * scroll container's content doesn't yet overflow.
 *
 * Both fallback observers can fire many times per scroll burst; we coalesce
 * them into one read per animation frame so we only hit `scrollHeight`
 * (a forced-layout property) once per frame. The downstream `fetchMore` is
 * additionally rate-limited at the data-source layer, which is where the
 * real concurrency hazard (parallel pulls on the same iterator) lives.
 *
 * The `signal` parameter is an arbitrary value the caller updates whenever
 * the data layer has produced new state — typically `results.length`. Passing
 * it as a dep guarantees we re-evaluate the fit check after a fetch returns,
 * even when the page-level filter rejected every new row (in which case the
 * DOM doesn't change and the MutationObserver wouldn't fire on its own).
 *
 * The auto-fire fallback is also bounded by `MAX_CONSECUTIVE_AUTOFIRES`: a
 * hard backstop on no-overflow fires that never trips in normal use (callers
 * pause auto-loading far sooner) but caps a runaway sweep of the full remote
 * listing if that pause logic ever regresses. The counter resets on overflow,
 * on a fresh enable, and when the list shrinks (a new search).
 */
const MAX_CONSECUTIVE_AUTOFIRES = 40;

export function useInfiniteScroll(
  fetchMore: () => void,
  signal: number,
  enabled = true,
) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const fetchMoreRef = useRef(fetchMore);
  useEffect(() => {
    fetchMoreRef.current = fetchMore;
  }, [fetchMore]);

  const autoFireCountRef = useRef(0);
  const prevSignalRef = useRef(signal);
  const wasEnabledRef = useRef(false);

  // IntersectionObserver: fires when the sentinel scrolls into view.
  useEffect(() => {
    if (!enabled) return;
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            fetchMoreRef.current();
          }
        }
      },
      {
        threshold: 0,
        root: scrollRef.current,
        rootMargin: "200px 0px",
      },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [enabled, signal]);

  // Auto-fire fallback: keep requesting batches while the scroll container
  // doesn't overflow, including the case where every fetched row was filtered
  // out by the page (so itemCount stalls but we still need more raw data).
  useEffect(() => {
    if (!enabled) {
      wasEnabledRef.current = false;
      return;
    }
    // A fresh enable (new search, or the user resuming after a paused sweep)
    // or a shrinking list (results reset) clears the backstop so legitimate
    // loading can refill the viewport.
    if (!wasEnabledRef.current || signal < prevSignalRef.current) {
      autoFireCountRef.current = 0;
    }
    wasEnabledRef.current = true;
    prevSignalRef.current = signal;

    const root = scrollRef.current;
    if (!root) return;

    const tryFire = () => {
      const sentinel = sentinelRef.current;
      if (!sentinel?.isConnected) return;
      if (root.scrollHeight > root.clientHeight + 4) {
        autoFireCountRef.current = 0;
        return;
      }
      if (autoFireCountRef.current >= MAX_CONSECUTIVE_AUTOFIRES) return;
      autoFireCountRef.current += 1;
      fetchMoreRef.current();
    };

    let frame: number | null = null;
    const schedule = () => {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        tryFire();
      });
    };

    schedule();

    const ro = new ResizeObserver(schedule);
    ro.observe(root);

    const mo = new MutationObserver(schedule);
    mo.observe(root, { childList: true, subtree: true });

    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      ro.disconnect();
      mo.disconnect();
    };
  }, [enabled, signal]);

  return { scrollRef, sentinelRef };
}
