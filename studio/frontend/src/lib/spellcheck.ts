// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Off sets spellcheck="false" on <html>, which every field inherits (none opts back in).
export const SPELLCHECK_STORAGE_KEY = "unsloth_spellcheck";

const listeners = new Set<() => void>();

export function getSpellCheck(): boolean {
  try {
    return localStorage.getItem(SPELLCHECK_STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

function apply(): void {
  const root = document.documentElement;
  if (getSpellCheck()) {
    // Remove rather than set "true", so an untouched install keeps the engine default.
    root.removeAttribute("spellcheck");
  } else {
    root.setAttribute("spellcheck", "false");
  }
}

export function setSpellCheck(enabled: boolean): void {
  try {
    if (enabled) {
      localStorage.removeItem(SPELLCHECK_STORAGE_KEY);
    } else {
      localStorage.setItem(SPELLCHECK_STORAGE_KEY, "false");
    }
  } catch {
    // storage unavailable
  }
  apply();
  for (const listener of listeners) {
    listener();
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useSpellCheck(): boolean {
  return useSyncExternalStore(subscribe, getSpellCheck);
}

/** Apply the stored choice now and follow toggles made in another tab. */
export function watchSpellCheck(win: Window): void {
  apply();
  win.addEventListener("storage", (event) => {
    if (event.key === SPELLCHECK_STORAGE_KEY || event.key === null) {
      apply();
      for (const listener of listeners) {
        listener();
      }
    }
  });
}
