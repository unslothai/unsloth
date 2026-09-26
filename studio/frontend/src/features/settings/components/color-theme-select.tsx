// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  COLOR_THEMES,
  FLAVOR_THEME_IDS,
  type ColorThemeId,
  UNSLOTH_THEME_IDS,
} from "../lib/color-themes";
import { usePalette, useTheme } from "../stores/theme-store";

/** "Aa" swatch in the theme's background and accent. */
function ThemeChip({
  colors,
}: {
  colors: { background: string; accent: string } | null;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-6 shrink-0 items-center justify-center rounded-full border text-[11px] font-semibold leading-none",
        colors
          ? "border-black/10 dark:border-white/15"
          : "border-border bg-muted text-muted-foreground",
      )}
      style={
        colors
          ? { backgroundColor: colors.background, color: colors.accent }
          : undefined
      }
    >
      Aa
    </span>
  );
}

export function ColorThemeSelect() {
  const t = useT();
  const { resolved } = useTheme();
  const { palette, setPalette } = usePalette();

  const nameOf = (id: ColorThemeId) =>
    COLOR_THEMES[id].name ??
    t(
      `settings.appearance.palette.${id as (typeof UNSLOTH_THEME_IDS)[number]}`,
    );

  const renderItem = (id: ColorThemeId) => (
    <DropdownMenuItem
      key={id}
      onSelect={() => setPalette(id)}
      data-palette-value={id}
      aria-checked={palette === id}
      role="menuitemradio"
    >
      <ThemeChip colors={COLOR_THEMES[id][resolved]} />
      <span className="min-w-0 flex-1 truncate">{nameOf(id)}</span>
      {palette === id ? (
        <HugeiconsIcon icon={Tick02Icon} className="size-4 text-foreground" />
      ) : null}
    </DropdownMenuItem>
  );

  const flavorActive = (FLAVOR_THEME_IDS as readonly string[]).includes(
    palette,
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("settings.appearance.palette.label")}
          className="flex h-9 w-52 cursor-pointer items-center gap-2 rounded-full border border-border bg-background pl-1.5 pr-3 text-sm outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring data-[state=open]:bg-accent/50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:focus-visible:border-transparent dark:focus-visible:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
        >
          <ThemeChip colors={COLOR_THEMES[palette][resolved]} />
          <span className="min-w-0 flex-1 truncate text-left">
            {nameOf(palette)}
          </span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" sideOffset={6} className="w-64">
        {UNSLOTH_THEME_IDS.map(renderItem)}
        <DropdownMenuSeparator />
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className="gap-2.5">
            <ThemeChip colors={null} />
            <span className="flex-1 whitespace-nowrap">
              {t("settings.appearance.palette.moreThemes")}
            </span>
            {/* Truncate the theme name, not the label. */}
            {flavorActive ? (
              <span className="min-w-0 truncate text-xs text-muted-foreground">
                {nameOf(palette)}
              </span>
            ) : null}
          </DropdownMenuSubTrigger>
          {/* Cap height and scroll an inner viewport to keep rounded corners. */}
          <DropdownMenuSubContent className="flex max-h-[min(--spacing(112),var(--radix-dropdown-menu-content-available-height))] w-56 flex-col">
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
              {FLAVOR_THEME_IDS.map(renderItem)}
            </div>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
