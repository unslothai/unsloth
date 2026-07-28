// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

const WIDTH_KEY = "sidebar_width";

/** The previous fixed 17.5rem, at a 16px root font size. */
export const SIDEBAR_WIDTH_DEFAULT = 280;
/** Header lockup plus the search and collapse buttons need ~258px. */
export const SIDEBAR_WIDTH_MIN = 264;
export const SIDEBAR_WIDTH_MAX = 480;
const SIDEBAR_MAX_VIEWPORT_FRACTION = 0.4;

function maxWidth(): number {
  if (typeof window === "undefined") return SIDEBAR_WIDTH_MAX;
  const viewportCap = window.innerWidth * SIDEBAR_MAX_VIEWPORT_FRACTION;
  // The floor wins on a narrow window; collapsing is the escape.
  return Math.max(SIDEBAR_WIDTH_MIN, Math.min(SIDEBAR_WIDTH_MAX, viewportCap));
}

/** Clamps to the absolute range, ignoring the viewport. */
function clampStored(px: number): number {
  if (!Number.isFinite(px)) return SIDEBAR_WIDTH_DEFAULT;
  return Math.min(
    SIDEBAR_WIDTH_MAX,
    Math.max(SIDEBAR_WIDTH_MIN, Math.round(px)),
  );
}

/** Clamps to what the current viewport allows. */
export function clampSidebarWidth(px: number): number {
  return Math.min(maxWidth(), clampStored(px));
}

function loadWidth(): number {
  if (typeof window === "undefined") return SIDEBAR_WIDTH_DEFAULT;
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    if (raw === null) return SIDEBAR_WIDTH_DEFAULT;
    return clampStored(Number.parseFloat(raw));
  } catch {
    return SIDEBAR_WIDTH_DEFAULT;
  }
}

// The preference is stored whole; `effective` is what the viewport allows.
// Narrowing the window shrinks the sidebar without losing the preference.
let storedWidth = loadWidth();
let effectiveWidth = clampSidebarWidth(storedWidth);
const listeners = new Set<() => void>();

function recompute() {
  const next = clampSidebarWidth(storedWidth);
  if (next === effectiveWidth) return;
  effectiveWidth = next;
  listeners.forEach((cb) => cb());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  // Keep tabs in sync, same as the pin flag.
  const onStorage = (e: StorageEvent) => {
    if (e.key === WIDTH_KEY || e.key === null) {
      storedWidth = loadWidth();
      effectiveWidth = clampSidebarWidth(storedWidth);
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
      window.localStorage.setItem(WIDTH_KEY, String(stored));
    } catch {}
  }
  recompute();
}

export function useSidebarWidth() {
  const width = useSyncExternalStore(
    subscribe,
    () => effectiveWidth,
    () => SIDEBAR_WIDTH_DEFAULT,
  );

  const setWidth = useCallback((value: number) => setWidthGlobal(value), []);
  const resetWidth = useCallback(
    () => setWidthGlobal(SIDEBAR_WIDTH_DEFAULT),
    [],
  );

  return { width, setWidth, resetWidth };
}
