// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Whether an update banner may appear, per component. On by default; only an
// explicit "false" (Settings -> General -> Notifications) disables it. A switch
// each: the two components ship on their own schedules.
const LLAMA_STORAGE_KEY = "unsloth_show_llama_update_banner";
const WHISPER_STORAGE_KEY = "unsloth_show_whisper_update_banner";

const listeners = new Set<() => void>();

function readPref(key: string): boolean {
  try {
    return localStorage.getItem(key) !== "false";
  } catch {
    return true;
  }
}

function writePref(key: string, show: boolean): void {
  try {
    if (show) {
      // Remove rather than store "true" so the default stays on.
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, "false");
    }
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) listener();
}

// One listener set for both keys: a subscriber re-reads its own getter, so a
// notification it did not need costs it a comparison.
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Sync toggles made in another tab.
  const onStorage = (event: StorageEvent) => {
    if (event.key === LLAMA_STORAGE_KEY || event.key === WHISPER_STORAGE_KEY) {
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function getShowLlamaUpdateBanner(): boolean {
  return readPref(LLAMA_STORAGE_KEY);
}

export function setShowLlamaUpdateBanner(show: boolean): void {
  writePref(LLAMA_STORAGE_KEY, show);
}

export function useShowLlamaUpdateBanner(): boolean {
  return useSyncExternalStore(subscribe, getShowLlamaUpdateBanner);
}

export function getShowWhisperUpdateBanner(): boolean {
  return readPref(WHISPER_STORAGE_KEY);
}

export function setShowWhisperUpdateBanner(show: boolean): void {
  writePref(WHISPER_STORAGE_KEY, show);
}

export function useShowWhisperUpdateBanner(): boolean {
  return useSyncExternalStore(subscribe, getShowWhisperUpdateBanner);
}
