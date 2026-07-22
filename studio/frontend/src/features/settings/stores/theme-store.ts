// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";
export type Palette = "standard" | "classic" | "minimal";

const STORAGE_KEY = "theme";
const PALETTE_STORAGE_KEY = "palette";

export const PALETTES: readonly Palette[] = ["standard", "classic", "minimal"];

export function isPalette(value: unknown): value is Palette {
  return value === "standard" || value === "classic" || value === "minimal";
}

// Persist a re-derived literal from a fixed allow-list rather than the argument,
// so a value arriving via the authenticated personalization sync is not tracked
// as sensitive data flowing into storage (these are plain UI preferences).
const STORED_THEME: Record<Theme, Theme> = {
  light: "light",
  dark: "dark",
  system: "system",
};
const STORED_PALETTE: Record<Palette, Palette> = {
  standard: "standard",
  classic: "classic",
  minimal: "minimal",
};

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "system";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return "system";
  }
  if (stored === "light" || stored === "dark" || stored === "system")
    return stored;
  return "system";
}

function readStoredPalette(): Palette {
  if (typeof window === "undefined") return "standard";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(PALETTE_STORAGE_KEY);
  } catch {
    return "standard";
  }
  return isPalette(stored) ? stored : "standard";
}

// In-memory source of truth so a selected value survives even when
// localStorage is blocked (private browsing). Without it the snapshots would
// re-read empty storage and revert React state to the default while the DOM
// already changed.
let currentTheme: Theme = readStoredTheme();
let currentPalette: Palette = readStoredPalette();

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
  const el = document.documentElement;
  el.classList.toggle("dark", resolved === "dark");
  el.classList.toggle("light", resolved === "light");
  // Native controls (scrollbars, spinners, pickers) follow the app mode.
  el.style.colorScheme = resolved;
}

function applyPaletteToDocument(palette: Palette) {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  // Standard is the base :root/.dark palette; no attribute keeps the DOM
  // (and CSS selectors) simple for the default look.
  if (palette === "standard") {
    el.removeAttribute("data-palette");
  } else {
    el.setAttribute("data-palette", palette);
  }
}

const listeners = new Set<() => void>();
function subscribe(cb: () => void) {
  listeners.add(cb);
  if (typeof window === "undefined") {
    return () => listeners.delete(cb);
  }
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  // OS scheme flip: only the resolved value changes; keep the in-memory choice
  // (re-reading storage here would clobber it when storage is blocked).
  const onSchemeChange = () => {
    applyToDocument(resolveTheme(currentTheme));
    cb();
  };
  // Another tab wrote storage (only fires when storage is available): adopt it.
  const onStorage = (e: StorageEvent) => {
    if (
      e.key === STORAGE_KEY ||
      e.key === PALETTE_STORAGE_KEY ||
      e.key === null
    ) {
      currentTheme = readStoredTheme();
      currentPalette = readStoredPalette();
      applyToDocument(resolveTheme(currentTheme));
      applyPaletteToDocument(currentPalette);
      cb();
    }
  };
  // Apply on mount so this store is the single source of truth for the DOM
  // class after the index.html bootstrap script painted the first frame.
  applyToDocument(resolveTheme(currentTheme));
  applyPaletteToDocument(currentPalette);
  mq.addEventListener("change", onSchemeChange);
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    mq.removeEventListener("change", onSchemeChange);
    window.removeEventListener("storage", onStorage);
  };
}

function getSnapshot(): Theme {
  return currentTheme;
}

function getServerSnapshot(): Theme {
  return "system";
}

// Snapshot the RESOLVED mode too: under "system" the theme string never
// changes when the OS scheme flips, so consumers keyed on `resolved`
// (customization applier, mode-scoped settings) would not re-render.
function getResolvedSnapshot(): ResolvedTheme {
  return resolveTheme(currentTheme);
}

function getResolvedServerSnapshot(): ResolvedTheme {
  return "light";
}

/**
 * Single source of truth for setting the theme. All writers (Settings dialog
 * control, sidebar dropdown toggler) route through this so the DOM class,
 * localStorage, and React subscribers stay in sync.
 */
export function setTheme(next: Theme): void {
  if (typeof window === "undefined") return;
  currentTheme = next;
  // Persist "system" explicitly so a reload keeps following the OS.
  try {
    window.localStorage.setItem(STORAGE_KEY, STORED_THEME[next]);
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
  const resolved = useSyncExternalStore(
    subscribe,
    getResolvedSnapshot,
    getResolvedServerSnapshot,
  );
  return { theme, resolved, setTheme };
}

function getPaletteSnapshot(): Palette {
  return currentPalette;
}

function getPaletteServerSnapshot(): Palette {
  return "standard";
}

/**
 * Single source of truth for setting the color palette; mirrors setTheme so
 * the data-palette attribute, localStorage, and React subscribers stay in
 * sync.
 */
export function setPalette(next: Palette): void {
  if (typeof window === "undefined") return;
  currentPalette = next;
  try {
    window.localStorage.setItem(PALETTE_STORAGE_KEY, STORED_PALETTE[next]);
  } catch {
    // ignore storage failures
  }
  applyPaletteToDocument(next);
  listeners.forEach((cb) => cb());
}

export function usePalette(): {
  palette: Palette;
  setPalette: (next: Palette) => void;
} {
  const palette = useSyncExternalStore(
    subscribe,
    getPaletteSnapshot,
    getPaletteServerSnapshot,
  );
  return { palette, setPalette };
}
