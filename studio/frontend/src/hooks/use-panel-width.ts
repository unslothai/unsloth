// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";
import { layoutScale, subscribeLayoutScale } from "../lib/layout-scale.ts";

const MAX_VIEWPORT_FRACTION = 0.4;

export type PanelWidthStore = {
  clamp: (px: number) => number;
  useWidth: () => {
    width: number;
    max: number;
    /** Browser interface scale; render at width * scale. */
    scale: number;
    stored: number;
    setWidth: (value: number) => void;
    resetWidth: () => void;
  };
};

/** Widths are layout px rendered times `scale`. The preference is stored whole and an effective
 * width derived, so narrowing the window does not lose the user's pick. */
export function createPanelWidthStore({
  key,
  min,
  max,
  fallback,
  maxViewportFraction = MAX_VIEWPORT_FRACTION,
}: {
  key: string;
  min: number;
  max: number;
  fallback: number;
  maxViewportFraction?: number;
}): PanelWidthStore {
  function maxWidth(): number {
    if (typeof window === "undefined") return max;
    // The floor wins on a narrow window; collapsing is the escape.
    return Math.max(
      min,
      Math.min(max, (window.innerWidth / layoutScale()) * maxViewportFraction),
    );
  }

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
    // With no subscribers there is no resize listener, so the cache may be stale; refresh first.
    recompute();
    listeners.add(cb);
    if (typeof window === "undefined") {
      return () => listeners.delete(cb);
    }
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
    const unsubscribeScale = subscribeLayoutScale(recompute);
    return () => {
      listeners.delete(cb);
      unsubscribeScale();
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
    // For aria-valuemax.
    const panelMax = useSyncExternalStore(subscribe, () => effectiveMax, () => max);
    // Uncapped, so a capped drag can avoid lowering it.
    const preference = useSyncExternalStore(subscribe, () => storedWidth, () => fallback);
    const scale = useSyncExternalStore(subscribeLayoutScale, layoutScale, () => 1);
    const setWidth = useCallback((value: number) => setWidthGlobal(value), []);
    const resetWidth = useCallback(() => setWidthGlobal(fallback), []);
    return { width, max: panelMax, scale, stored: preference, setWidth, resetWidth };
  }

  return { clamp, useWidth };
}
