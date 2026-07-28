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

export function clampSidebarWidth(px: number): number {
  if (!Number.isFinite(px)) return SIDEBAR_WIDTH_DEFAULT;
  return Math.min(maxWidth(), Math.max(SIDEBAR_WIDTH_MIN, Math.round(px)));
}

function loadWidth(): number {
  if (typeof window === "undefined") return SIDEBAR_WIDTH_DEFAULT;
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    if (raw === null) return SIDEBAR_WIDTH_DEFAULT;
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return SIDEBAR_WIDTH_DEFAULT;
    return clampSidebarWidth(parsed);
  } catch {
    return SIDEBAR_WIDTH_DEFAULT;
  }
}

let widthValue = loadWidth();
const listeners = new Set<() => void>();

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  // Keep tabs in sync, same as the pin flag.
  const onStorage = (e: StorageEvent) => {
    if (e.key === WIDTH_KEY || e.key === null) {
      widthValue = loadWidth();
      cb();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}

function setWidthGlobal(next: number) {
  const clamped = clampSidebarWidth(next);
  if (clamped === widthValue) return;
  widthValue = clamped;
  try {
    window.localStorage.setItem(WIDTH_KEY, String(clamped));
  } catch {}
  listeners.forEach((cb) => cb());
}

export function useSidebarWidth() {
  const width = useSyncExternalStore(
    subscribe,
    () => widthValue,
    () => SIDEBAR_WIDTH_DEFAULT,
  );

  const setWidth = useCallback((value: number) => setWidthGlobal(value), []);
  const resetWidth = useCallback(
    () => setWidthGlobal(SIDEBAR_WIDTH_DEFAULT),
    [],
  );

  return { width, setWidth, resetWidth };
}
