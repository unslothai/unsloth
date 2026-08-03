// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

/** Never let one panel eat more than this share of a narrow window. */
const MAX_VIEWPORT_FRACTION = 0.4;

export type PanelWidthStore = {
  /** Clamps to what the current viewport allows. */
  clamp: (px: number) => number;
  useWidth: () => {
    width: number;
    max: number;
    /** The uncapped stored preference. */
    stored: number;
    setWidth: (value: number) => void;
    resetWidth: () => void;
  };
};

/**
 * A persisted, viewport-aware width for a draggable panel. The preference is
 * stored whole and an effective width is derived from it, so narrowing the
 * window shrinks the panel without losing what the user picked.
 */
export function createPanelWidthStore({
  key,
  min,
  max,
  fallback,
}: {
  key: string;
  min: number;
  max: number;
  fallback: number;
}): PanelWidthStore {
  function maxWidth(): number {
    if (typeof window === "undefined") return max;
    // The floor wins on a narrow window; collapsing is the escape.
    return Math.max(min, Math.min(max, window.innerWidth * MAX_VIEWPORT_FRACTION));
  }

  /** Clamps to the absolute range, ignoring the viewport. */
  function clampStored(px: number): number {
    if (!Number.isFinite(px)) return fallback;
    return Math.min(max, Math.max(min, Math.round(px)));
  }

  function clamp(px: number): number {
    return Math.min(maxWidth(), clampStored(px));
  }

  function load(): number {
    if (typeof window === "undefined") return fallback;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null) return fallback;
      return clampStored(Number.parseFloat(raw));
    } catch {
      return fallback;
    }
  }

  let storedWidth = load();
  let effectiveWidth = clamp(storedWidth);
  let effectiveMax = maxWidth();
  const listeners = new Set<() => void>();

  let lastStored = storedWidth;

  function recompute() {
    const nextWidth = clamp(storedWidth);
    const nextMax = maxWidth();
    if (
      nextWidth === effectiveWidth &&
      nextMax === effectiveMax &&
      storedWidth === lastStored
    ) {
      return;
    }
    effectiveWidth = nextWidth;
    effectiveMax = nextMax;
    lastStored = storedWidth;
    listeners.forEach((cb) => cb());
  }

  function subscribe(cb: () => void) {
    // With no subscribers there is no resize listener, so the cache can be
    // stale after a resize on a route that hides every panel. Refresh first;
    // useSyncExternalStore re-reads the snapshot right after subscribing.
    recompute();
    listeners.add(cb);
    if (typeof window === "undefined") {
      return () => listeners.delete(cb);
    }
    // Keep tabs in sync, same as the pin flag.
    const onStorage = (e: StorageEvent) => {
      if (e.key === key || e.key === null) {
        storedWidth = load();
        effectiveWidth = clamp(storedWidth);
        effectiveMax = maxWidth();
        cb();
      }
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener("resize", recompute);
    return () => {
      listeners.delete(cb);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("resize", recompute);
    };
  }

  function setWidthGlobal(next: number) {
    const stored = clampStored(next);
    if (stored !== storedWidth) {
      storedWidth = stored;
      try {
        window.localStorage.setItem(key, String(stored));
      } catch {}
    }
    recompute();
  }

  function useWidth() {
    const width = useSyncExternalStore(subscribe, () => effectiveWidth, () => fallback);
    // What the viewport actually allows right now, for aria-valuemax.
    const panelMax = useSyncExternalStore(subscribe, () => effectiveMax, () => max);
    // The uncapped preference, so a capped drag can avoid lowering it.
    const preference = useSyncExternalStore(subscribe, () => storedWidth, () => fallback);
    const setWidth = useCallback((value: number) => setWidthGlobal(value), []);
    const resetWidth = useCallback(() => setWidthGlobal(fallback), []);
    return { width, max: panelMax, stored: preference, setWidth, resetWidth };
  }

  return { clamp, useWidth };
}
