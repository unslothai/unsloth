// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { type ReactNode, useId } from "react";

export interface PillSegmentedOption<T extends string> {
  value: T;
  label: ReactNode;
  icon?: Parameters<typeof HugeiconsIcon>[0]["icon"];
}

/**
 * Segmented control whose segments are each as wide as their own label, with
 * the selected pill following the active one. Prefer `SegmentedControl` when
 * the segments should divide a track evenly instead.
 */
export function PillSegmented<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  disabled,
}: {
  value: T;
  options: readonly PillSegmentedOption<T>[];
  onChange: (value: T) => void;
  ariaLabel: string;
  disabled?: boolean;
}) {
  const reduced = useReducedMotion();
  // Per instance: a shared id would animate the pill between two controls.
  const pill = useId();
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={cn(
        "hub-tab-toggle inline-flex h-8 items-center rounded-full",
        disabled && "opacity-50",
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={active}
            disabled={disabled}
            className={cn(
              "relative flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors",
              disabled && "cursor-not-allowed",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId={pill}
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            )}
            {option.icon ? (
              <HugeiconsIcon
                icon={option.icon}
                className="relative z-10 size-3.5"
              />
            ) : null}
            <span className="relative z-10">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
