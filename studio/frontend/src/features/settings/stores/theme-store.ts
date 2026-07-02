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
  // Keep "dark"/"light" mutually exclusive: next-themes (via Sonner) adds
  // "light" on first mount, so without this toggle we'd get "light dark".
  const cl = document.documentElement.classList;
  cl.toggle("dark", resolved === "dark");
  cl.toggle("light", resolved === "light");
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
  // Apply on mount so this store is the single source of truth for the DOM
  // class. Without this, initial paint depends solely on next-themes, which on
  // a fresh origin (e.g. a new Cloudflare --secure link with empty
  // localStorage) falls back to its own default and shows light while the
  // control still reads "system".
  applyToDocument(resolveTheme(readStoredTheme()));
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
 * Single source of truth for setting the theme. All writers (Settings dialog
 * control, sidebar dropdown toggler) route through this so the DOM class,
 * localStorage, and React subscribers stay in sync.
 */
export function setTheme(next: Theme): void {
  if (typeof window === "undefined") return;
  // Persist "system" explicitly so next-themes doesn't clobber the choice on
  // reload.
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
