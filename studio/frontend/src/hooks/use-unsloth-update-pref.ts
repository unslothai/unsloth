// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Whether Unsloth itself may check for a new app/backend release automatically.
// Desktop and web checks run independently, so they share one explicit opt-out
// here. Manual checks remain available from Settings -> About.
const STORAGE_KEY = "unsloth_show_unsloth_update_banner";

const listeners = new Set<() => void>();

function readPref(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

function writePref(show: boolean): void {
  try {
    // Remove rather than store "true" so the default stays on.
    if (show) {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, "false");
    }
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) {
    listener();
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) {
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function getShowUnslothUpdateBanner(): boolean {
  return readPref();
}

export function setShowUnslothUpdateBanner(show: boolean): void {
  writePref(show);
}

export function useShowUnslothUpdateBanner(): boolean {
  return useSyncExternalStore(subscribe, getShowUnslothUpdateBanner);
}
