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
    stored = globalThis.localStorage.getItem(STORAGE_KEY);
  } catch {
    return "system";
  }
  if (stored === "light" || stored === "dark" || stored === "system")
    return stored;
  return "system";
}

function systemPrefersDark(): boolean {
  if (typeof window === "undefined") return false;
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
  if (typeof window === "undefined") {
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

type ThemeSnapshot = `${Theme}:${ResolvedTheme}`;

function getSnapshot(): ThemeSnapshot {
  const theme = readStoredTheme();
  return `${theme}:${resolveTheme(theme)}`;
}

function getServerSnapshot(): ThemeSnapshot {
  return "system:light";
}

/**
 * Single source of truth for setting the theme. The initial DOM class is
 * applied by public/theme-init.js (to avoid FOUC and stay within Tauri's
 * default-src 'self' CSP); this function keeps the class, localStorage,
 * and React subscribers in sync after that.
 */
export function setTheme(next: Theme): void {
  if (typeof window === "undefined") return;
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
  const snapshot = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );
  const [theme, resolved] = snapshot.split(":") as [Theme, ResolvedTheme];
  return { theme, resolved, setTheme };
}
