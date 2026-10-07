// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Off by default; only explicit "true" enables it. Tri-state on purpose: see setShowLoadedModels.

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
    // Write "false", never remove: older tabs also read "false" as off, so the storage event cannot flip it on.
    localStorage.setItem(STORAGE_KEY, show ? "true" : "false");
  } catch {
    // storage unavailable
  }
  notify();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
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

/** Distinct from the Settings toggle: the next load reopens a dismissed card, not a disabled one. */
export function getLoadedModelsDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISSED_KEY) === "true";
  } catch {
    return false;
  }
}

export function setLoadedModelsDismissed(dismissed: boolean): void {
  // Load starts run this every time, so skip a no-op write to avoid re-rendering the overlay stack.
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
