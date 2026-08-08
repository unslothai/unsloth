// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the corner indicator may appear. On by default; only an explicit
// "false" (Settings -> General -> Notifications) disables it. Same shape as the
// llama.cpp update banner pref.

import { useSyncExternalStore } from "react";

/** Every key this feature owns, so "Reset all local preferences" clears them. */
export const LOADED_MODELS_PREFERENCE_KEYS = {
  show: "unsloth_show_loaded_models_indicator",
  collapsed: "unsloth_loaded_models_collapsed",
  position: "unsloth_loaded_models_position",
  size: "unsloth_loaded_models_size",
} as const;

const STORAGE_KEY = LOADED_MODELS_PREFERENCE_KEYS.show;

const listeners = new Set<() => void>();

export function getShowLoadedModels(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

export function setShowLoadedModels(show: boolean): void {
  try {
    if (show) {
      // Remove rather than store "true" so the default stays on.
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, "false");
    }
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Sync toggles made in another tab.
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener();
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
