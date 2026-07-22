// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Whether the llama.cpp update banner may appear. On by default; only an
// explicit "false" (Settings -> General -> Notifications) disables it.
const STORAGE_KEY = "unsloth_show_llama_update_banner";

const listeners = new Set<() => void>();

export function getShowLlamaUpdateBanner(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

export function setShowLlamaUpdateBanner(show: boolean): void {
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

export function useShowLlamaUpdateBanner(): boolean {
  return useSyncExternalStore(subscribe, getShowLlamaUpdateBanner);
}
