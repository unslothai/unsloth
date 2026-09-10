// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useSyncExternalStore } from "react";

const MOBILE_BREAKPOINT = 768;
const MEDIA_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

// A narrowed desktop window is still a desktop window. The mobile shell has its
// own sidebar (a sheet over a dimmed page) and header, so crossing the
// breakpoint by dragging the window edge would restyle the app, not fit it.
const FOLLOWS_BREAKPOINT = !isTauri;

function getSnapshot(): boolean {
  if (typeof window === "undefined" || !FOLLOWS_BREAKPOINT) return false;
  return window.matchMedia(MEDIA_QUERY).matches;
}

function subscribe(callback: () => void): () => void {
  if (typeof window === "undefined" || !FOLLOWS_BREAKPOINT) return () => {};
  const mql = window.matchMedia(MEDIA_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}
