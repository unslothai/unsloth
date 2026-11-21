// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";

const PINNED_KEY = "sidebar_pinned";
const WIDE_QUERY = "(min-width: 1024px)";

function readStored(): boolean | null {
  try {
    const raw = window.localStorage.getItem(PINNED_KEY);
    return raw === null ? null : raw === "true";
  } catch {
    return null;
  }
}

// Read at import time, so a throw would fail every importer.
function wideQuery(): MediaQueryList | null {
  return typeof window.matchMedia === "function" ? window.matchMedia(WIDE_QUERY) : null;
}

function loadPinned(): boolean {
  if (typeof window === "undefined") return true;
  // Unpinned below lg; with no width to read, keep the desktop default.
  return readStored() ?? wideQuery()?.matches ?? true;
}

let pinnedValue = loadPinned();
// Once the user toggles, width changes never override it.
let chosen = false;
// A tour override: never persisted or counted as a choice.
let held = false;
let beforeHold = pinnedValue;
const listeners = new Set<() => void>();

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  const onStorage = (e: StorageEvent) => {
    if (e.key === PINNED_KEY || e.key === null) {
      // Another tab set or cleared it (Reset all local preferences).
      chosen = readStored() !== null;
      if (held) beforeHold = loadPinned();
      else pinnedValue = loadPinned();
      cb();
    }
  };
  const wide = wideQuery();
  const onWidth = () => {
    if (chosen || held) return;
    pinnedValue = loadPinned();
    cb();
  };
  window.addEventListener("storage", onStorage);
  wide?.addEventListener("change", onWidth);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
    wide?.removeEventListener("change", onWidth);
  };
}

function setPinnedGlobal(next: boolean) {
  chosen = true;
  held = false;
  pinnedValue = next;
  try {
    window.localStorage.setItem(PINNED_KEY, String(next));
  } catch {}
  listeners.forEach((cb) => cb());
}

export function holdSidebarPinned() {
  if (!held) beforeHold = pinnedValue;
  held = true;
  pinnedValue = true;
  listeners.forEach((cb) => cb());
}

export function releaseSidebarPinned() {
  if (!held) return;
  held = false;
  pinnedValue = chosen ? beforeHold : loadPinned();
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
