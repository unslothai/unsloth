// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  COLOR_THEMES,
  HALLOWEEN_THEME_IDS,
} from "@/features/settings/lib/color-themes";
import { usePalette } from "@/features/settings/stores/theme-store";
import { useT } from "@/i18n";
import {
  ChevronLeftStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { CandyIcon, Moon02Icon, SkullIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useState } from "react";
import {
  pickHalloweenTheme,
  pickRandomHalloweenTheme,
  removeHalloween,
  useHalloweenStore,
} from "./halloween-store";
import { PumpkinIcon } from "./pumpkin-icon";

function NightSky() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 400 140"
      className="pointer-events-none absolute inset-x-0 top-0 h-36 w-full text-foreground opacity-[0.14]"
      fill="currentColor"
    >
      <circle cx="70" cy="22" r="2" />
      <circle cx="318" cy="16" r="1.5" />
      <circle cx="352" cy="64" r="2" />
      <circle cx="36" cy="96" r="1.5" />
      <path d="M96 30a18 18 0 1 0 14 30a14 14 0 1 1-14-30z" />
      <path d="M300 34c4-6 10-7 14-4c2-4 6-4 8 0c4-3 10-2 14 4c-5-1-8 1-10 4c-2-2-4-2-6 0c-2-2-4-2-6 0c-2-3-6-5-14-4z" />
      <path d="M258 72c3-4 7-5 10-3c1-3 4-3 6 0c3-2 7-1 10 3c-4-1-6 1-7 3c-1-1-3-1-4 0c-1-1-3-1-4 0c-2-2-4-3-11-3z" />
    </svg>
  );
}

function Row({
  icon,
  label,
  description,
  onClick,
  trailing,
}: {
  icon: typeof CandyIcon;
  label: string;
  description?: string;
  onClick: () => void;
  trailing: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex min-h-[calc(64px*var(--ui-space-scale,1))] w-full cursor-pointer items-center gap-4 rounded-2xl px-4 py-2.5 text-left outline-none transition-colors hover:bg-accent focus-visible:bg-accent"
    >
      <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-primary/12 text-primary">
        <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="font-medium text-ui-14p5 text-foreground">
          {label}
        </span>
        {description && (
          <span className="text-muted-foreground text-xs">{description}</span>
        )}
      </span>
      <span className="flex w-11 shrink-0 justify-end">{trailing}</span>
    </button>
  );
}

// A chevron's arms carry its weight, so nudge it 1px towards the point to look centred.
function ArrowBadge() {
  return (
    <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
      <HugeiconsIcon
        icon={ChevronRightStandardIcon}
        className="size-4 translate-x-px rtl:-translate-x-px rtl:rotate-180"
      />
    </span>
  );
}

function ThemeGrid() {
  const { palette } = usePalette();
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {HALLOWEEN_THEME_IDS.map((id) => {
        // The dark accent in both modes: the light ones are deepened for contrast and read muddy.
        const { name, dark: colors } = COLOR_THEMES[id];
        const selected = palette === id;
        return (
          <button
            key={id}
            type="button"
            data-palette-value={id}
            aria-pressed={selected}
            onClick={() => pickHalloweenTheme(id)}
            className={cn(
              "flex min-w-0 cursor-pointer items-center gap-2 rounded-full border py-1.5 pr-3 pl-1.5 text-left outline-none transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring",
              selected ? "border-primary ring-1 ring-primary" : "border-border",
            )}
          >
            <span
              aria-hidden="true"
              className="size-7 shrink-0 rounded-full border border-black/10 dark:border-white/15"
              style={{
                background: `linear-gradient(135deg, ${colors.accent} 30%, color-mix(in srgb, ${colors.accent} 55%, black) 100%)`,
              }}
            />
            <span className="min-w-0 truncate whitespace-nowrap font-medium text-sm">
              {name}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function TrickOrTreatDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const [view, setView] = useState<"home" | "themes">("home");
  const spookyGreetings = useHalloweenStore((s) => s.spookyGreetings);
  const setSpookyGreetings = useHalloweenStore((s) => s.setSpookyGreetings);

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next);
        if (!next) {
          setView("home");
        }
      }}
    >
      <DialogContent
        data-testid="trick-or-treat-dialog"
        className="gap-0 overflow-hidden px-0 pt-0 pb-0 sm:max-w-[min(calc(560px*var(--ui-space-scale,1)),calc(100%-2rem))]"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-48 bg-[radial-gradient(110%_100%_at_50%_0%,color-mix(in_oklab,var(--primary)_24%,transparent),transparent_70%)]" />
        {view === "home" && <NightSky />}
        {view === "home" ? (
          <div className="relative flex flex-col">
            <div className="flex flex-col items-center gap-2.5 px-12 pt-10 pb-7 text-center">
              <span className="mb-1 flex size-14 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-[0_0_32px_-4px_var(--primary)]">
                <HugeiconsIcon
                  icon={PumpkinIcon}
                  strokeWidth={1.25}
                  className="size-9"
                />
              </span>
              <DialogTitle className="font-semibold text-2xl tracking-[-0.01em]">
                {t("halloween.title")}
              </DialogTitle>
              <DialogDescription className="max-w-sm text-muted-foreground text-sm">
                {t("halloween.description")}
              </DialogDescription>
            </div>
            <div className="flex flex-col gap-1 px-6">
              <Row
                icon={Moon02Icon}
                label={t("halloween.chooseTheme")}
                onClick={() => setView("themes")}
                trailing={<ArrowBadge />}
              />
              <Row
                icon={SkullIcon}
                label={t("halloween.spookyGreetings")}
                description={t("halloween.spookyGreetingsDescription")}
                onClick={() => setSpookyGreetings(!spookyGreetings)}
                trailing={
                  <Switch
                    aria-label={t("halloween.spookyGreetings")}
                    checked={spookyGreetings}
                    tabIndex={-1}
                    className="pointer-events-none"
                  />
                }
              />
              <Row
                icon={CandyIcon}
                label={t("halloween.surprise")}
                description={t("halloween.surpriseDescription")}
                onClick={() => {
                  pickRandomHalloweenTheme();
                  setView("themes");
                }}
                trailing={<ArrowBadge />}
              />
            </div>
            <p className="px-12 pt-6 pb-10 text-center text-muted-foreground text-xs italic">
              {t("halloween.footer")}{" "}
              <button
                type="button"
                data-testid="trick-or-treat-remove"
                onClick={() => {
                  removeHalloween();
                  onOpenChange(false);
                }}
                className="cursor-pointer text-primary underline-offset-2 hover:underline focus-visible:underline focus-visible:outline-none"
              >
                {t("halloween.removeNow")}
              </button>
            </p>
          </div>
        ) : (
          <div className="relative flex flex-col gap-5 px-6 pt-6 pb-7">
            <div className="flex items-center gap-2 pr-8">
              <button
                type="button"
                aria-label={t("halloween.back")}
                onClick={() => setView("home")}
                className="flex size-8 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:bg-accent focus-visible:outline-none"
              >
                <HugeiconsIcon
                  icon={ChevronLeftStandardIcon}
                  className="size-4 -translate-x-px rtl:translate-x-px rtl:rotate-180"
                />
              </button>
              <DialogTitle className="font-semibold text-lg">
                {t("halloween.themesTitle")}
              </DialogTitle>
              <DialogDescription className="sr-only">
                {t("halloween.description")}
              </DialogDescription>
            </div>
            <ThemeGrid />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
