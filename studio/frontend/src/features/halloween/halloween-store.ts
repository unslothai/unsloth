// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type ColorThemeId,
  HALLOWEEN_THEME_IDS,
  type HalloweenThemeId,
  isColorThemeId,
  isHalloweenThemeId,
} from "../settings/lib/color-themes.ts";
import {
  getPalette,
  setPalette,
  usePalette,
} from "../settings/stores/theme-store.ts";

export const HALLOWEEN_STORAGE_KEY = "unsloth_halloween";

/** Local time, so the season ends at the user's own midnight on November 1. */
export function isHalloweenSeason(now: Date = new Date()): boolean {
  return now.getMonth() === 9;
}

type HalloweenState = {
  spookyGreetings: boolean;
  // Year the user hid the sidebar row; it stays hidden for the rest of that October.
  hiddenYear: number | null;
  restorePalette: ColorThemeId | null;
  setSpookyGreetings: (on: boolean) => void;
};

export const useHalloweenStore = create<HalloweenState>()(
  persist(
    (set) => ({
      spookyGreetings: false,
      hiddenYear: null,
      restorePalette: null,
      setSpookyGreetings: (spookyGreetings) => set({ spookyGreetings }),
    }),
    {
      name: HALLOWEEN_STORAGE_KEY,
      partialize: ({ spookyGreetings, hiddenYear, restorePalette }) => ({
        spookyGreetings,
        hiddenYear,
        restorePalette,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<HalloweenState>;
        return {
          ...current,
          spookyGreetings: p.spookyGreetings === true,
          hiddenYear: typeof p.hiddenYear === "number" ? p.hiddenYear : null,
          restorePalette:
            isColorThemeId(p.restorePalette) &&
            !isHalloweenThemeId(p.restorePalette)
              ? p.restorePalette
              : null,
        };
      },
    },
  ),
);

/** Remembers the theme in use before the first Halloween pick, so the season can hand it back. */
export function pickHalloweenTheme(id: HalloweenThemeId): void {
  const current = getPalette();
  if (!isHalloweenThemeId(current)) {
    useHalloweenStore.setState({ restorePalette: current });
  }
  setPalette(id);
}

export function pickRandomHalloweenTheme(): HalloweenThemeId {
  const current = getPalette();
  const choices = HALLOWEEN_THEME_IDS.filter((id) => id !== current);
  const id = choices[Math.floor(Math.random() * choices.length)];
  pickHalloweenTheme(id);
  return id;
}

function restoreNormalTheme(): void {
  if (isHalloweenThemeId(getPalette())) {
    setPalette(useHalloweenStore.getState().restorePalette ?? "standard");
  }
  useHalloweenStore.setState({ restorePalette: null });
}

export function removeHalloween(now: Date = new Date()): void {
  restoreNormalTheme();
  useHalloweenStore.setState({
    spookyGreetings: false,
    hiddenYear: now.getFullYear(),
  });
}

export function hideTrickOrTreatRow(now: Date = new Date()): void {
  useHalloweenStore.setState({ hiddenYear: now.getFullYear() });
}

export function showTrickOrTreatRow(): void {
  useHalloweenStore.setState({ hiddenYear: null });
}

export function useTrickOrTreatVisible(): boolean {
  const hiddenYear = useHalloweenStore((s) => s.hiddenYear);
  const now = new Date();
  return isHalloweenSeason(now) && hiddenYear !== now.getFullYear();
}

export function useSpookyGreetings(): boolean {
  const on = useHalloweenStore((s) => s.spookyGreetings);
  return on && isHalloweenSeason();
}

/** Outside October, hand back the pre-Halloween theme, including one a synced device re-applies. */
export function useHalloweenSeasonEnd(): void {
  const { palette } = usePalette();
  useEffect(() => {
    if (!isHalloweenSeason() && isHalloweenThemeId(palette)) {
      restoreNormalTheme();
    }
  }, [palette]);
}
