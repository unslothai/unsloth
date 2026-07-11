// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type Palette,
  type ResolvedTheme,
  usePalette,
  useTheme,
} from "../stores/theme-store";

type PreviewColors = {
  bg: string;
  sidebar: string;
  accent: string;
  text: string;
  border: string;
};

// Representative swatches per palette × resolved mode. Hardcoded (rather than
// reading the live CSS variables) so every card previews its own palette while
// only one palette is active on <html>. Sidebar tones are slightly exaggerated
// so the two surfaces stay distinguishable at thumbnail size.
const PREVIEWS: Record<Palette, Record<ResolvedTheme, PreviewColors>> = {
  standard: {
    light: {
      bg: "#fefefd",
      sidebar: "#f1f1ef",
      accent: "#17b88b",
      text: "#444444",
      border: "#e4e4e0",
    },
    dark: {
      bg: "#181818",
      sidebar: "#262626",
      accent: "#17b88b",
      text: "#b5b5b5",
      border: "#303030",
    },
  },
  classic: {
    light: {
      bg: "#ffffff",
      sidebar: "#ededed",
      accent: "#339cff",
      text: "#4b4d50",
      border: "#e6e6e6",
    },
    dark: {
      bg: "#181818",
      sidebar: "#262626",
      accent: "#4dabff",
      text: "#b5b5b5",
      border: "#303030",
    },
  },
  minimal: {
    light: {
      bg: "#ffffff",
      sidebar: "#f2f2f2",
      accent: "#171717",
      text: "#555555",
      border: "#e2e2e2",
    },
    dark: {
      bg: "#181818",
      sidebar: "#262626",
      accent: "#ededed",
      text: "#a8a8a8",
      border: "#303030",
    },
  },
};

const OPTIONS: {
  value: Palette;
  labelKey: TranslationKey;
}[] = [
  { value: "standard", labelKey: "settings.appearance.palette.standard" },
  { value: "classic", labelKey: "settings.appearance.palette.classic" },
  { value: "minimal", labelKey: "settings.appearance.palette.minimal" },
];

function PalettePreview({ colors }: { colors: PreviewColors }) {
  return (
    <div
      className="h-16 w-full overflow-hidden rounded-lg border"
      style={{ backgroundColor: colors.bg, borderColor: colors.border }}
      aria-hidden="true"
    >
      <div className="flex h-full">
        <div
          className="h-full w-1/4"
          style={{ backgroundColor: colors.sidebar }}
        />
        <div className="flex flex-1 flex-col justify-center gap-1.5 px-2.5">
          <span
            className="h-1.5 w-3/4 rounded-full"
            style={{ backgroundColor: colors.text }}
          />
          <span
            className="h-1.5 w-1/2 rounded-full opacity-60"
            style={{ backgroundColor: colors.text }}
          />
          <span
            className="h-2.5 w-8 rounded-full"
            style={{ backgroundColor: colors.accent }}
          />
        </div>
      </div>
    </div>
  );
}

export function PaletteCards() {
  const t = useT();
  const { resolved } = useTheme();
  const { palette, setPalette } = usePalette();
  return (
    <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-3">
      {OPTIONS.map((opt) => {
        const active = palette === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => setPalette(opt.value)}
            aria-pressed={active}
            className={cn(
              "flex flex-col gap-2 rounded-xl border p-2.5 text-left transition-colors",
              active
                ? "border-ring ring-1 ring-ring"
                : "border-border hover:border-ring/40",
            )}
          >
            <PalettePreview colors={PREVIEWS[opt.value][resolved]} />
            <div className="flex items-center justify-between gap-2 px-0.5">
              <span className="text-sm font-medium text-foreground">
                {t(opt.labelKey)}
              </span>
              {active && (
                <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  <HugeiconsIcon icon={Tick02Icon} className="size-3" />
                </span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
