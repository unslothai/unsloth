// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

const PINNED_KEY = "sidebar_pinned";
const WIDE_QUERY = "(min-width: 1024px)";

function loadPinned(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const raw = window.localStorage.getItem(PINNED_KEY);
    // Default to unpinned below lg.
    if (raw === null) return window.matchMedia(WIDE_QUERY).matches;
    return raw === "true";
  } catch {
    return true;
  }
}

let pinnedValue = loadPinned();
const listeners = new Set<() => void>();

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  const onStorage = (e: StorageEvent) => {
    if (e.key === PINNED_KEY || e.key === null) {
      pinnedValue = loadPinned();
      cb();
    }
  };
  // Re-derive the width default across lg. A stored choice still wins.
  const wide = window.matchMedia(WIDE_QUERY);
  const onWidth = () => {
    pinnedValue = loadPinned();
    cb();
  };
  window.addEventListener("storage", onStorage);
  wide.addEventListener("change", onWidth);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
    wide.removeEventListener("change", onWidth);
  };
}

function setPinnedGlobal(next: boolean) {
  pinnedValue = next;
  try {
    window.localStorage.setItem(PINNED_KEY, String(next));
  } catch {}
  listeners.forEach((cb) => cb());
}

export function useSidebarPin() {
  const pinned = useSyncExternalStore(
    subscribe,
    () => pinnedValue,
    () => false,
  );

  const setPinned = useCallback((value: boolean) => setPinnedGlobal(value), []);
  const togglePinned = useCallback(() => setPinnedGlobal(!pinnedValue), []);

  return { pinned, setPinned, togglePinned };
}
