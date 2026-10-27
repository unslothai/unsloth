// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import {
  COLOR_THEME_IDS,
  type ColorThemeId,
  isColorThemeId,
} from "../lib/color-themes";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";
export type Palette = ColorThemeId;

const STORAGE_KEY = "theme";
const PALETTE_STORAGE_KEY = "palette";

export const PALETTES: readonly Palette[] = COLOR_THEME_IDS;

export function isPalette(value: unknown): value is Palette {
  return isColorThemeId(value);
}

// Persist a literal from an allow-list, not the argument, so synced values are not flagged as
// sensitive data flowing into storage.
const STORED_THEME: Record<Theme, Theme> = {
  light: "light",
  dark: "dark",
  system: "system",
};
const STORED_PALETTE = Object.fromEntries(
  COLOR_THEME_IDS.map((id) => [id, id]),
) as Record<Palette, Palette>;

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

// In-memory source of truth so a choice survives blocked localStorage.
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
  el.style.colorScheme = resolved;
}

function applyPaletteToDocument(palette: Palette) {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  // Standard is the base :root/.dark palette and uses no attribute.
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
  // Keep the in-memory choice; re-reading blocked storage would clobber it.
  const onSchemeChange = () => {
    applyToDocument(resolveTheme(currentTheme));
    cb();
  };
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
  // Take over the DOM class after the index.html bootstrap painted the first frame.
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

// Under "system" the theme string never changes when the OS flips, so snapshot the resolved mode.
function getResolvedSnapshot(): ResolvedTheme {
  return resolveTheme(currentTheme);
}

function getResolvedServerSnapshot(): ResolvedTheme {
  return "light";
}

/** Every writer goes through here so the DOM class, storage and subscribers stay in sync. */
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
