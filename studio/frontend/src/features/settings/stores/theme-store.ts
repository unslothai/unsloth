// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "theme";

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "system";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return "system";
  }
  if (stored === "light" || stored === "dark" || stored === "system") return stored;
  return "system";
}

function systemPrefersDark(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
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
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const syncTheme = () => {
    applyToDocument(resolveTheme(readStoredTheme()));
    cb();
  };
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY || e.key === null) syncTheme();
  };
  mq.addEventListener("change", syncTheme);
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    mq.removeEventListener("change", syncTheme);
    window.removeEventListener("storage", onStorage);
  };
}

function getSnapshot(): Theme {
  return readStoredTheme();
}

function getServerSnapshot(): Theme {
  return "system";
}

/**
 * Single source of truth for setting the theme. All writers (the Settings
 * dialog's segmented control AND the sidebar dropdown's animated toggler)
 * must route through this so the DOM class, localStorage, and React
 * subscribers stay in sync.
 */
export function setTheme(next: Theme): void {
  if (typeof window === "undefined") return;
  // Persist "system" explicitly so next-themes (mounted with
  // defaultTheme="light") doesn't clobber the choice on reload.
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // ignore storage failures
  }
  applyToDocument(resolveTheme(next));
  listeners.forEach((cb) => cb());
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
