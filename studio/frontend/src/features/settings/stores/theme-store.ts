// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "theme";

function readStoredTheme(): Theme {
  if (typeof globalThis === "undefined") return "system";
  let stored: string | null = null;
  try {
    stored = globalThis.localStorage.getItem(STORAGE_KEY);
  } catch {
    return "system";
  }
  if (stored === "light" || stored === "dark" || stored === "system")
    return stored;
  return "system";
}

function systemPrefersDark(): boolean {
  if (typeof globalThis === "undefined") return false;
  return globalThis.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolveTheme(theme: Theme): ResolvedTheme {
  if (theme === "system") return systemPrefersDark() ? "dark" : "light";
  return theme;
}

function applyToDocument(resolved: ResolvedTheme) {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

const listeners = new Set<() => void>();
function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof globalThis === "undefined") {
    return () => listeners.delete(cb);
  }
  const mq = globalThis.matchMedia("(prefers-color-scheme: dark)");
  const syncTheme = () => {
    applyToDocument(resolveTheme(readStoredTheme()));
    cb();
  };
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY || e.key === null) syncTheme();
  };
  mq.addEventListener("change", syncTheme);
  globalThis.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    mq.removeEventListener("change", syncTheme);
    globalThis.removeEventListener("storage", onStorage);
  };
}

function getSnapshot(): Theme {
  return readStoredTheme();
}

function getServerSnapshot(): Theme {
  return "system";
}

/**
 * Single source of truth for setting the theme. The initial DOM class is
 * applied by an inline script in index.html (to avoid FOUC); this function
 * keeps the class, localStorage, and React subscribers in sync after that.
 */
export function setTheme(next: Theme): void {
  if (typeof globalThis === "undefined") return;
  try {
    globalThis.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // ignore storage failures
  }
  applyToDocument(resolveTheme(next));
  for (const cb of listeners) cb();
}

export function useTheme(): {
  theme: Theme;
  resolved: ResolvedTheme;
  setTheme: (next: Theme) => void;
} {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const resolved = resolveTheme(theme);
  return { theme, resolved, setTheme };
}
