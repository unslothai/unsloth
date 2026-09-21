// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useSyncExternalStore } from "react";

const MOBILE_BREAKPOINT = 768;
const MEDIA_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

function getSnapshot(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia(MEDIA_QUERY).matches;
}

function subscribe(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(MEDIA_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

/**
 * A viewport too narrow to put a panel beside the content. Every platform: a
 * submenu or a settings panel that only has room to overlay has to overlay.
 */
export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

// any-pointer, not pointer: a laptop with a touchscreen and a trackpad still has a finger.
const COARSE_POINTER_QUERY = "(any-pointer: coarse)";

function getCoarseSnapshot(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia(COARSE_POINTER_QUERY).matches;
}

function subscribeCoarse(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(COARSE_POINTER_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

/** A touch screen is present, where the browser never starts a native drag from it. */
export function useIsCoarsePointer(): boolean {
  return useSyncExternalStore(subscribeCoarse, getCoarseSnapshot, () => false);
}

/**
 * Whether to swap in the mobile shell: a sheet sidebar over a dimmed page, its
 * own width, its own header. Never in the desktop app, where a narrowed window
 * is still a desktop window, and restyling the shell around it is not the same
 * as fitting it.
 */
export function useIsMobileShell(): boolean {
  return useIsMobile() && !isTauri;
}
