// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

type Side = "top" | "right" | "bottom" | "left";
export type CollisionPadding = number | Partial<Record<Side, number>>;

// Height of the desktop titlebar, which paints over body-portaled surfaces. 0 in a browser.
let chromeTop: number | null = null;
const listeners = new Set<() => void>();

function readChromeTop(): number {
  if (
    typeof document === "undefined" ||
    typeof getComputedStyle !== "function"
  ) {
    return 0;
  }
  const value = Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue(
      "--studio-window-chrome-top",
    ),
  );
  return Number.isFinite(value) && value > 0 ? value : 0;
}

export function getWindowChromeTop(): number {
  chromeTop ??= readChromeTop();
  return chromeTop;
}

/** Re-read after --studio-window-chrome-top or the value it references changes. */
export function refreshWindowChromeTop(): void {
  const next = readChromeTop();
  if (next === chromeTop) {
    return;
  }
  chromeTop = next;
  for (const listener of listeners) {
    listener();
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useWindowChromeTop(): number {
  return useSyncExternalStore(subscribe, getWindowChromeTop, () => 0);
}

/** Adds the titlebar to the top collision padding so popups size and flip below it. */
export function clearOfWindowChrome(
  padding: CollisionPadding | undefined,
  top: number,
): CollisionPadding | undefined {
  if (top <= 0) {
    return padding;
  }
  if (typeof padding === "object") {
    return { ...padding, top: (padding.top ?? 0) + top };
  }
  const base = padding ?? 0;
  return { top: base + top, right: base, bottom: base, left: base };
}

export function useWindowChromeCollisionPadding(
  padding: CollisionPadding | undefined,
): CollisionPadding | undefined {
  return clearOfWindowChrome(padding, useWindowChromeTop());
}
