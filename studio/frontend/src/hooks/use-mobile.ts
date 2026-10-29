// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useSyncExternalStore } from "react";

const MOBILE_BREAKPOINT = 768;
const MEDIA_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

let lastMobile = false;

// Full screen resizes the viewport; swapping layouts then would unmount the full screen element.
export function isMobileViewport(): boolean {
  if (typeof window === "undefined") return false;
  if (!document.fullscreenElement) lastMobile = window.matchMedia(MEDIA_QUERY).matches;
  return lastMobile;
}

function subscribe(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(MEDIA_QUERY);
  mql.addEventListener("change", callback);
  document.addEventListener("fullscreenchange", callback);
  return () => {
    mql.removeEventListener("change", callback);
    document.removeEventListener("fullscreenchange", callback);
  };
}

export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, isMobileViewport, () => false);
}

/** Never in the desktop app, where a narrowed window is still a desktop window. */
export function useIsMobileShell(): boolean {
  return useIsMobile() && !isTauri;
}

const COMPACT_QUERY = "(max-width: 1023px)";

function getCompactSnapshot(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia(COMPACT_QUERY).matches;
}

function subscribeCompact(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(COMPACT_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

export function useIsCompact(): boolean {
  return useSyncExternalStore(subscribeCompact, getCompactSnapshot, () => false);
}
