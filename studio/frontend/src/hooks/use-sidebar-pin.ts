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

function loadPinned(): boolean {
  if (typeof window === "undefined") return true;
  // Default to unpinned below lg, also when storage is unavailable.
  return readStored() ?? window.matchMedia(WIDE_QUERY).matches;
}

let pinnedValue = loadPinned();
// Set once the user toggles, so a width change never overrides it.
let chosen = false;
// A tour override: shown, but never persisted or counted as a choice.
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
      pinnedValue = loadPinned();
      cb();
    }
  };
  // Re-derive the width default across lg unless the user has chosen.
  const wide = window.matchMedia(WIDE_QUERY);
  const onWidth = () => {
    if (chosen || held) return;
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
  chosen = true;
  held = false;
  pinnedValue = next;
  try {
    window.localStorage.setItem(PINNED_KEY, String(next));
  } catch {}
  listeners.forEach((cb) => cb());
}

/** Pin for a tour without touching the stored preference. */
export function holdSidebarPinned() {
  if (!held) beforeHold = pinnedValue;
  held = true;
  pinnedValue = true;
  listeners.forEach((cb) => cb());
}

/** End the hold: the user's choice if any, else the default for the current width. */
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
