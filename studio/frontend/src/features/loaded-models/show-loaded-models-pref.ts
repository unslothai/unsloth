// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the corner indicator may appear. Off by default; only an explicit
// "true" (Settings -> General -> Notifications) enables it. An older explicit
// "false" still reads as off, so anyone who already turned it down stays that
// way. Tri-state on purpose: see setShowLoadedModels.

import { useSyncExternalStore } from "react";

/** Every key this feature owns, so "Reset all local preferences" clears them. */
export const LOADED_MODELS_PREFERENCE_KEYS = {
  show: "unsloth_show_loaded_models_indicator",
  collapsed: "unsloth_loaded_models_collapsed",
  position: "unsloth_loaded_models_position",
  dismissed: "unsloth_loaded_models_dismissed",
} as const;

const STORAGE_KEY = LOADED_MODELS_PREFERENCE_KEYS.show;
const DISMISSED_KEY = LOADED_MODELS_PREFERENCE_KEYS.dismissed;

const listeners = new Set<() => void>();

/** Both flags feed the same subscribers, since both decide whether it shows. */
function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getShowLoadedModels(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

export function setShowLoadedModels(show: boolean): void {
  try {
    // Both values written, never removed: "false" is the one an older reader
    // also treats as off, so a pre-update tab does not flip the card back on
    // through the storage event. Absent still means off here, so the default
    // is unaffected.
    localStorage.setItem(STORAGE_KEY, show ? "true" : "false");
  } catch {
    // storage unavailable
  }
  notify();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Sync toggles made in another tab.
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY || event.key === DISMISSED_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useShowLoadedModels(): boolean {
  return useSyncExternalStore(subscribe, getShowLoadedModels);
}

/**
 * Closed with the card's own X, which is not the same as switched off in
 * Settings. Kept apart on purpose: the next model load reopens a card the user
 * waved away, and must not reopen one they turned off.
 *
 * Stored rather than held in memory so a reload does not bring back a card that
 * was closed, the same as the collapsed state.
 */
export function getLoadedModelsDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISSED_KEY) === "true";
  } catch {
    return false;
  }
}

export function setLoadedModelsDismissed(dismissed: boolean): void {
  // Nothing to announce when it already reads that way, and the reopen path
  // runs on every load start, so this would otherwise re-render the app's whole
  // overlay stack for each one.
  if (getLoadedModelsDismissed() === dismissed) {
    return;
  }
  try {
    if (dismissed) {
      localStorage.setItem(DISMISSED_KEY, "true");
    } else {
      // Removed rather than stored "false" so the default stays open.
      localStorage.removeItem(DISMISSED_KEY);
    }
  } catch {
    // storage unavailable
  }
  notify();
}

export function useLoadedModelsDismissed(): boolean {
  return useSyncExternalStore(subscribe, getLoadedModelsDismissed);
}
