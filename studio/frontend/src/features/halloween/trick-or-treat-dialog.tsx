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
import {
  type ReactElement,
  type ReactNode,
  isValidElement,
  useState,
} from "react";
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

// Fluent Emoji "spider web" (high contrast), MIT (github.com/microsoft/fluentui-emoji).
function SpiderWebIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="2 2 28 28"
      fill="currentColor"
      className="size-[calc(22px*var(--ui-space-scale,1))]"
    >
      <path d="M29.93 16.29c.02-.03.05-.07.06-.11v-.01c.01-.02.01-.04.02-.06c0-.02.01-.04.01-.07v-.08c0-.02-.01-.04-.01-.06v-.01c0-.02-.01-.04-.02-.05v-.01l-.03-.06c-.01-.02-.02-.03-.03-.05l-.01-.01c-.01-.02-.03-.03-.04-.05a.2.2 0 0 0-.05-.04l-.01-.01a.2.2 0 0 1-.06-.04c-2.58-1.39-4-3.3-4.2-5.67c-.16-1.81.46-3.23.46-3.24v-.01c.01-.02.02-.04.02-.06v-.01c.01-.02.01-.04.01-.06v-.16a.3.3 0 0 0-.03-.11s0-.01-.01-.01a.2.2 0 0 0-.03-.05c-.01-.02-.03-.04-.04-.05v-.01l-.01-.01l-.02-.02l-.02-.02l-.01-.01c-.02-.01-.03-.03-.05-.04c0 0-.01 0-.01-.01c-.01 0-.03-.01-.04-.02c0 0-.01 0-.01-.01c-.02-.01-.04-.02-.06-.02h-.01c-.02-.01-.04-.01-.06-.01h-.17c-.02 0-.04.01-.06.01c-2.81.84-5.16.5-6.98-1.04c-1.38-1.16-1.95-2.6-1.95-2.61l-.03-.06c-.01-.02-.02-.03-.03-.05l-.01-.01c-.01-.02-.03-.03-.04-.05l-.04-.04l-.01-.01c-.03-.02-.07-.05-.11-.06h-.01c-.02-.01-.04-.01-.06-.02c-.04-.01-.06-.02-.08-.02h-.09c-.02 0-.04.01-.06.01h-.01c-.02 0-.03.01-.05.02c0 0-.01 0-.01.01l-.06.03h-.01c-.02.01-.03.02-.05.03l-.01.01c-.02.01-.03.03-.05.04l-.04.04l-.01.01c-.01.02-.02.04-.03.05c-1.39 2.58-3.3 4-5.67 4.2C8.11 6.63 6.69 6 6.68 6h-.01l-.06-.03H6.6c-.02 0-.04-.01-.05-.01h-.16c-.04.01-.08.02-.12.04c0 0-.01 0-.01.01c-.02 0-.04.01-.05.02c-.03.02-.04.03-.06.04h-.01l-.01.01c0 .01 0 .02-.01.02c-.01.01-.01.02-.02.02l-.01.01c-.01.02-.03.04-.04.06c0 0 0 .01-.01.01c0 .01-.01.03-.02.04c0 0 0 .01-.01.01c0 .02-.01.04-.02.07c0 .02-.01.04-.01.06v.15c0 .02.01.04.01.06c.84 2.81.5 5.16-1.04 6.98c-1.16 1.38-2.59 1.94-2.62 1.95l-.06.03h-.01c-.02.01-.03.02-.05.03l-.01.01c-.02.01-.03.03-.05.04l-.01.01c-.01.01-.02.03-.04.04l-.01.01c-.02.03-.04.07-.06.11v.01c-.01.02-.01.04-.02.06c0 .02-.01.04-.01.06v.08c0 .02.01.04.01.06v.01c0 .02.01.04.02.05v.01l.03.06c.01.02.02.03.03.05l.01.01c.01.02.03.03.04.05c.01.01.03.03.05.04l.01.01c.02.01.04.02.06.04c2.58 1.39 4 3.3 4.2 5.67c.18 1.84-.44 3.26-.44 3.27v.01c-.01.02-.02.04-.02.06v.01c-.01.02-.01.04-.01.06v.16c.01.04.02.08.04.12c0 0 0 .01.01.01c.01.02.02.04.03.05c.01.02.03.04.04.05v.01l.01.01l.02.02l.02.02l.01.01c.02.01.03.03.05.04c0 0 .01 0 .01.01c.02.01.03.02.05.03c0 0 .01 0 .01.01c.02.01.04.02.06.02h.01c.02.01.04.01.06.01h.15c.02 0 .04-.01.06-.01c2.81-.85 5.16-.5 6.98 1.04c1.38 1.16 1.95 2.6 1.95 2.61l.03.06c.01.02.02.03.03.05l.01.01c.01.02.03.03.04.05l.04.04l.01.01c.03.02.07.05.11.06h.01c.02.01.04.01.06.02c.02 0 .04.01.06.01h.12c.02 0 .03-.01.05-.01h.01a.4.4 0 0 0 .11-.05l.01-.01c.01-.01.03-.02.04-.03l.01-.01c.02-.01.03-.03.04-.04l.01-.01c.01-.01.02-.02.03-.04l.01-.01c.01-.02.02-.03.03-.05c1.39-2.58 3.3-4 5.67-4.2c1.8-.16 3.22.46 3.23.47c.02.01.04.02.06.02h.01c.02.01.04.01.06.01h.17c.04-.01.08-.02.12-.04c0 0 .01 0 .01-.01c.02-.01.04-.02.05-.03c.02-.01.04-.03.05-.04h.01l.01-.01l.02-.02l.02-.02l.01-.01c.01-.02.03-.03.04-.05c0 0 0-.01.01-.01c.01-.02.02-.03.03-.05c0 0 0-.01.01-.01c.01-.02.02-.04.02-.06c.01-.02.01-.04.01-.06v-.15c0-.02-.01-.04-.01-.06c-.84-2.81-.5-5.16 1.04-6.98c1.16-1.38 2.6-1.95 2.61-1.95l.06-.03h.01c.02-.01.03-.02.05-.03l.01-.01c.02-.01.03-.03.05-.04l.04-.04c.01-.04.01-.04.02-.05m-2.13-.79h-4.15c-2.84-1.6-2.08-4.05-1.9-4.52l2.86-2.86c-.08.53-.11 1.15-.06 1.81c.21 2.61 1.68 4.39 3.25 5.57m-16.3 3.86c-.07-.82-.39-1.9-1.41-2.86h4.73l-3.3 3.3c0-.14-.01-.29-.02-.44m5.02-9.23c.1.1.21.2.32.3c.6.51 1.53 1.02 2.85 1.02h.17l-3.34 3.34zm-1 4.66l-3.3-3.3c.14 0 .29 0 .44-.01c.82-.07 1.9-.39 2.86-1.41zm-.71.71h-4.66c.1-.1.2-.21.3-.32c.53-.63 1.07-1.61 1.02-3.02zm.71 1.71v4.67c-.1-.1-.21-.2-.32-.3c-.6-.51-1.53-1.02-2.85-1.02h-.17zm1 0l3.3 3.3c-.14 0-.29 0-.44.01c-.82.07-1.9.39-2.86 1.41zm.71-.71h4.67c-.1.1-.2.21-.3.32c-.53.63-1.07 1.61-1.02 3.02zm0-1l3.3-3.3c0 .14 0 .29.01.44c.07.82.39 1.9 1.41 2.86zm.53-9.83c1.99 1.7 4.29 1.92 6.24 1.64l-2.94 2.94c-3.14.88-4.33-1.39-4.54-1.85V4.35c.32.43.73.89 1.24 1.32m-7.81 1.8c2.61-.21 4.39-1.68 5.57-3.25v4.15c-1.6 2.84-4.05 2.08-4.52 1.9L8.14 7.41c.53.07 1.14.11 1.81.06m-4.26 6.79c1.7-1.99 1.92-4.29 1.64-6.24l2.94 2.94c.88 3.13-1.39 4.33-1.85 4.54H4.37c.43-.32.89-.73 1.32-1.24M4.24 16.5h4.15c2.84 1.6 2.08 4.05 1.9 4.52l-2.86 2.86c.08-.53.11-1.15.06-1.81c-.21-2.61-1.68-4.39-3.25-5.57m10.04 9.83c-1.54-1.31-3.27-1.74-4.87-1.74c-.47 0-.93.04-1.37.1l2.94-2.94c3.14-.88 4.33 1.39 4.54 1.85v4.05c-.32-.43-.73-.89-1.24-1.32m7.81-1.8c-2.61.21-4.39 1.68-5.57 3.25v-4.15c1.6-2.84 4.05-2.08 4.52-1.9l2.86 2.86c-.53-.07-1.14-.11-1.81-.06m4.26-6.79c-1.7 1.99-1.92 4.29-1.64 6.24l-2.94-2.94c-.88-3.14 1.39-4.33 1.85-4.54h4.05c-.43.32-.89.73-1.32 1.24" />
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
  icon: typeof CandyIcon | ReactElement;
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
        {isValidElement(icon) ? (
          icon
        ) : (
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
        )}
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

const SWATCH_OVERRIDE: Partial<
  Record<(typeof HALLOWEEN_THEME_IDS)[number], string>
> = {
  "trick-or-treat":
    "linear-gradient(135deg, #ffc400 28%, #b35cff 68%, color-mix(in srgb, #b35cff 75%, black) 100%)",
  gothic:
    "linear-gradient(135deg, #a78bfa 28%, #e0335e 72%, color-mix(in srgb, #e0335e 75%, black) 100%)",
  candycorn:
    "linear-gradient(170deg, #fff6e6 12%, #ff8f1f 42%, #ff8f1f 55%, #ffd84a 85%)",
};

const swatchBackground = (accent: string) =>
  `linear-gradient(135deg, ${accent} 30%, color-mix(in srgb, ${accent} 55%, black) 100%)`;

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
              className="size-7 shrink-0 rounded-full"
              style={{
                background:
                  SWATCH_OVERRIDE[id] ?? swatchBackground(colors.accent),
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
  const decorations = useHalloweenStore((s) => s.decorations);
  const setDecorations = useHalloweenStore((s) => s.setDecorations);

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
        className="gap-0 overflow-x-hidden px-0 pt-0 pb-0 sm:max-w-[min(calc(560px*var(--ui-space-scale,1)),calc(100%-2rem))]"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-48 bg-[radial-gradient(110%_100%_at_50%_0%,color-mix(in_oklab,var(--primary)_24%,transparent),transparent_70%)]" />
        {view === "home" && <NightSky />}
        {view === "home" ? (
          <div className="relative flex flex-col">
            <div className="flex flex-col items-center gap-2.5 px-12 pt-10 pb-7 text-center">
              <span className="mb-1 flex size-14 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-[0_0_32px_-4px_var(--primary)]">
                <HugeiconsIcon icon={PumpkinIcon} className="size-9" />
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
                icon={<SpiderWebIcon />}
                label={t("halloween.decorations")}
                description={t("halloween.decorationsDescription")}
                onClick={() => setDecorations(!decorations)}
                trailing={
                  <Switch
                    aria-label={t("halloween.decorations")}
                    checked={decorations}
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
