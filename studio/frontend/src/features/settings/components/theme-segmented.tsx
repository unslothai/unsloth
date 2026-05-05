// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  LaptopIcon,
  Moon02Icon,
  Sun02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { useTheme, type Theme } from "../stores/theme-store";

const OPTIONS: { value: Theme; label: string; icon: typeof Sun02Icon }[] = [
  { value: "light", label: "Light", icon: Sun02Icon },
  { value: "dark", label: "Dark", icon: Moon02Icon },
  { value: "system", label: "System", icon: LaptopIcon },
];

export function ThemeSegmented() {
  const { theme, setTheme } = useTheme();
  const reduced = useReducedMotion();
  return (
    <div className="inline-flex items-center rounded-md border border-border bg-muted/30 p-0.5">
      {OPTIONS.map((opt) => {
        const active = theme === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => setTheme(opt.value)}
            aria-pressed={active}
            className={cn(
              "relative flex h-7 items-center gap-1.5 rounded px-2.5 text-xs font-medium transition-colors",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="theme-pill"
                className="absolute inset-0 rounded bg-background shadow-border"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            )}
            <HugeiconsIcon icon={opt.icon} className="relative z-10 size-3.5" />
            <span className="relative z-10">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
