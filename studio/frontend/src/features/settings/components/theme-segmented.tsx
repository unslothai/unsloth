// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useT, type TranslationKey } from "@/i18n";
import {
  LaptopIcon,
  Moon02Icon,
  Sun02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { useTheme, type Theme } from "../stores/theme-store";

const OPTIONS: {
  value: Theme;
  labelKey: TranslationKey;
  icon: typeof Sun02Icon;
}[] = [
  { value: "light", labelKey: "settings.appearance.theme.light", icon: Sun02Icon },
  { value: "dark", labelKey: "settings.appearance.theme.dark", icon: Moon02Icon },
  { value: "system", labelKey: "settings.appearance.theme.system", icon: LaptopIcon },
];

export function ThemeSegmented() {
  const t = useT();
  const { theme, setTheme } = useTheme();
  const reduced = useReducedMotion();
  return (
    <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
      {OPTIONS.map((opt) => {
        const active = theme === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => setTheme(opt.value)}
            aria-pressed={active}
            className={cn(
              "relative flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="theme-pill"
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            )}
            <HugeiconsIcon icon={opt.icon} className="relative z-10 size-3.5" />
            <span className="relative z-10">{t(opt.labelKey)}</span>
          </button>
        );
      })}
    </div>
  );
}
