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
import {
  MATERIAL_LABELS,
  MATERIALS,
  nativeMaterialAvailable,
  useMaterialStore,
} from "@/features/chat-background";
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

function ThemeChip({
  colors,
}: {
  colors: { background: string; accent: string } | null;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-6 shrink-0 items-center justify-center rounded-full border text-ui-11 font-semibold leading-none",
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

function GlassChip() {
  return (
    <span
      aria-hidden="true"
      className="size-6 shrink-0 rounded-full border border-black/10 bg-[linear-gradient(135deg,rgb(255_255_255_/_0.85),rgb(255_255_255_/_0.2)_55%,rgb(255_255_255_/_0.55))] shadow-[inset_0_1px_0_rgb(255_255_255_/_0.9)] dark:border-white/20 dark:bg-[linear-gradient(135deg,rgb(255_255_255_/_0.35),rgb(255_255_255_/_0.05)_55%,rgb(255_255_255_/_0.18))]"
    />
  );
}

function GlassThemeSub() {
  const t = useT();
  const material = useMaterialStore((s) => s.material);
  const setMaterial = useMaterialStore((s) => s.setMaterial);
  const options = MATERIALS.filter(
    (m) => m !== "macos" || nativeMaterialAvailable(),
  );
  const shown = options.includes(material) ? material : "solid";
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger className="gap-2.5">
        <GlassChip />
        <span className="flex-1 whitespace-nowrap">
          {t("chatBackground.material")}
        </span>
        {shown !== "solid" && (
          <span className="text-xs text-muted-foreground">
            {t(MATERIAL_LABELS[shown])}
          </span>
        )}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className="w-60">
        {options.map((m) => (
          <DropdownMenuItem
            key={m}
            onSelect={() => setMaterial(m)}
            data-material-value={m}
            aria-checked={shown === m}
            role="menuitemradio"
          >
            <span className="min-w-0 flex-1 truncate">
              {t(MATERIAL_LABELS[m])}
            </span>
            {shown === m ? (
              <HugeiconsIcon
                icon={Tick02Icon}
                className="size-4 text-foreground"
              />
            ) : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
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

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("settings.appearance.palette.label")}
          className="flex h-control w-60 cursor-pointer items-center gap-2 rounded-full border border-border bg-background pl-1.5 pr-3 text-sm outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring data-[state=open]:bg-accent/50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:focus-visible:border-transparent dark:focus-visible:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
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
      <DropdownMenuContent align="end" sideOffset={6} className="w-60">
        {UNSLOTH_THEME_IDS.map(renderItem)}
        <GlassThemeSub />
        <DropdownMenuSeparator />
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className="gap-2.5">
            <ThemeChip colors={null} />
            <span className="flex-1 whitespace-nowrap">
              {t("settings.appearance.palette.moreThemes")}
            </span>
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="flex max-h-[min(--spacing(112),var(--radix-dropdown-menu-content-available-height))] w-64 flex-col">
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
              {FLAVOR_THEME_IDS.map(renderItem)}
            </div>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
