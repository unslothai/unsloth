// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { syncAcrossWindows } from "../../lib/sync-across-windows.ts";
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

/** Local time: the season ends at the user's own midnight. */
export function isHalloweenSeason(now: Date = new Date()): boolean {
  return now.getMonth() === 9;
}

export function msUntilSeasonChange(now: Date = new Date()): number {
  const year = now.getFullYear();
  const next = isHalloweenSeason(now)
    ? new Date(year, 10, 1)
    : new Date(now.getMonth() < 9 ? year : year + 1, 9, 1);
  return next.getTime() - now.getTime();
}

// setTimeout overflows past ~24.8 days and sleep stalls timers, so re-arm in bounded steps.
const SEASON_CHECK_MAX_MS = 6 * 60 * 60 * 1000;

type HalloweenState = {
  spookyGreetings: boolean;
  decorations: boolean;
  // Year the row was hidden; it stays hidden for that October.
  hiddenYear: number | null;
  restorePalette: ColorThemeId | null;
  setSpookyGreetings: (on: boolean) => void;
  setDecorations: (on: boolean) => void;
};

export const useHalloweenStore = create<HalloweenState>()(
  persist(
    (set) => ({
      spookyGreetings: false,
      decorations: true,
      hiddenYear: null,
      restorePalette: null,
      setSpookyGreetings: (spookyGreetings) => set({ spookyGreetings }),
      setDecorations: (decorations) => set({ decorations }),
    }),
    {
      name: HALLOWEEN_STORAGE_KEY,
      partialize: ({
        spookyGreetings,
        decorations,
        hiddenYear,
        restorePalette,
      }) => ({
        spookyGreetings,
        decorations,
        hiddenYear,
        restorePalette,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<HalloweenState>;
        return {
          ...current,
          spookyGreetings: p.spookyGreetings === true,
          decorations: p.decorations !== false,
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

// Another tab can pick the theme to restore; reread it before this one consumes or rewrites it.
syncAcrossWindows(useHalloweenStore);

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

export function useSpookyDecorations(): boolean {
  const on = useHalloweenStore((s) => s.decorations);
  const { palette } = usePalette();
  return on && isHalloweenSeason() && isHalloweenThemeId(palette);
}

/** Outside October, restore the pre-Halloween theme; reads the live date, so a stale flag never undoes an October pick. */
export function useHalloweenSeasonEnd(): void {
  const { palette } = usePalette();
  const [inSeason, setInSeason] = useState(() => isHalloweenSeason());
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const check = () => {
      clearTimeout(timer);
      setInSeason(isHalloweenSeason());
      timer = setTimeout(
        check,
        Math.min(msUntilSeasonChange(), SEASON_CHECK_MAX_MS),
      );
    };
    check();
    window.addEventListener("focus", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", check);
      document.removeEventListener("visibilitychange", check);
    };
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: inSeason only triggers the recheck at a boundary.
  useEffect(() => {
    if (!isHalloweenSeason() && isHalloweenThemeId(palette)) {
      restoreNormalTheme();
    }
  }, [palette, inSeason]);
}
