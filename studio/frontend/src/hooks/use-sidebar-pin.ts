// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

const PINNED_KEY = "sidebar_pinned";

function loadPinned(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const raw = window.localStorage.getItem(PINNED_KEY);
    if (raw === null) return true;
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
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
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
