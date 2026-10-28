// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { ThemeChip } from "@/features/settings/components/color-theme-select";
import { SettingsRow } from "@/features/settings/components/settings-row";
import {
  COLOR_THEMES,
  HALLOWEEN_THEME_IDS,
} from "@/features/settings/lib/color-themes";
import { usePalette, useTheme } from "@/features/settings/stores/theme-store";
import { useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  pickHalloweenTheme,
  pickRandomHalloweenTheme,
  removeHalloween,
  useHalloweenStore,
} from "./halloween-store";

function ThemeGrid() {
  const t = useT();
  const { resolved } = useTheme();
  const { palette } = usePalette();
  return (
    <div className="flex flex-col gap-2.5">
      <span className="font-medium text-foreground text-sm">
        {t("halloween.chooseTheme")}
      </span>
      <div className="grid grid-cols-2 gap-2">
        {HALLOWEEN_THEME_IDS.map((id) => {
          const selected = palette === id;
          return (
            <button
              key={id}
              type="button"
              data-palette-value={id}
              aria-pressed={selected}
              onClick={() => pickHalloweenTheme(id)}
              className={cn(
                "flex h-control cursor-pointer items-center gap-2 rounded-full border border-border bg-background pr-3 pl-1.5 text-left text-sm outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus-visible:border-transparent dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
                selected &&
                  "bg-accent/50 dark:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
              )}
            >
              <ThemeChip colors={COLOR_THEMES[id][resolved]} />
              <span className="min-w-0 flex-1 truncate">
                {COLOR_THEMES[id].name}
              </span>
              {selected ? (
                <HugeiconsIcon
                  icon={Tick02Icon}
                  className="size-4 shrink-0 text-foreground"
                />
              ) : null}
            </button>
          );
        })}
      </div>
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
  const spookyGreetings = useHalloweenStore((s) => s.spookyGreetings);
  const setSpookyGreetings = useHalloweenStore((s) => s.setSpookyGreetings);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        data-testid="trick-or-treat-dialog"
        className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md"
      >
        <DialogHeader>
          <DialogTitle>{t("halloween.title")}</DialogTitle>
          <DialogDescription>{t("halloween.description")}</DialogDescription>
        </DialogHeader>
        <ThemeGrid />
        <SettingsRow
          label={t("halloween.spookyGreetings")}
          description={t("halloween.spookyGreetingsDescription")}
          className="py-0"
        >
          <Switch
            aria-label={t("halloween.spookyGreetings")}
            checked={spookyGreetings}
            onCheckedChange={setSpookyGreetings}
          />
        </SettingsRow>
        <p className="text-muted-foreground text-xs">{t("halloween.footer")}</p>
        <DialogFooter>
          <Button
            variant="outline"
            data-testid="trick-or-treat-remove"
            onClick={() => {
              removeHalloween();
              onOpenChange(false);
            }}
          >
            {t("halloween.removeNow")}
          </Button>
          <Button onClick={() => pickRandomHalloweenTheme()}>
            {t("halloween.surprise")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
