// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export interface PillTab {
  value: string;
  label: string;
  icon?: ReactNode;
}

/** Segmented pill toggle reusing the Hub's .hub-tab-toggle styling (extended in
 * hub.css to also match .unsloth-model-selector-menu). Keeps tab roles for
 * keyboard nav. */
export function PillTabs({
  tabs,
  value,
  onValueChange,
  ariaLabel,
  className,
  compact = false,
  fit = false,
}: {
  tabs: PillTab[];
  value: string;
  onValueChange: (value: string) => void;
  ariaLabel: string;
  className?: string;
  compact?: boolean;
  /** Size each tab to its label instead of equal widths. The active tab carries
   * the pill background directly (the toggle never animates). */
  fit?: boolean;
}) {
  const activeIndex = Math.max(
    0,
    tabs.findIndex((tab) => tab.value === value),
  );
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "hub-menu-trigger hub-tab-toggle relative inline-flex items-center rounded-full",
        compact ? "h-7" : "h-9",
        // Don't stretch to fill a flex-column parent (the popover) in fit mode,
        // and never compress so the last tab keeps its padding.
        fit && "w-fit max-w-full shrink-0 self-start",
        className,
      )}
    >
      {!fit && (
        <span
          aria-hidden="true"
          style={{
            width: `${100 / tabs.length}%`,
            transform: `translateX(${activeIndex * 100}%)`,
          }}
          className="hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 rounded-full transition-transform duration-200 ease-out"
        />
      )}
      {tabs.map((tab, index) => (
        <button
          key={tab.value}
          type="button"
          role="tab"
          aria-selected={value === tab.value}
          // Roving tabindex: only the active tab is in the tab order; Arrow
          // Left/Right move between tabs (WAI-ARIA tablist pattern). ArrowDown
          // is left to bubble so the picker's "enter the list" handler still runs.
          tabIndex={value === tab.value ? 0 : -1}
          onKeyDown={(e) => {
            if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
            e.preventDefault();
            const next =
              (index + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) %
              tabs.length;
            onValueChange(tabs[next].value);
            e.currentTarget.parentElement
              ?.querySelectorAll<HTMLElement>('button[role="tab"]')
              .item(next)
              ?.focus();
          }}
          onClick={() => onValueChange(tab.value)}
          className={cn(
            "relative z-10 inline-flex items-center justify-center gap-1.5 rounded-full transition-colors",
            fit ? "shrink-0" : "min-w-0 flex-1",
            compact ? "h-7 px-2.5 text-[0.6875rem]" : "h-9 px-3 text-[0.78125rem]",
            value === tab.value
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
            // The active tab carries the pill; pin its hover bg so an
            // already-selected tab shows no hover change.
            fit &&
              value === tab.value &&
              "hub-tab-toggle-pill hover:!bg-[var(--background)] dark:hover:!bg-[color-mix(in_srgb,var(--foreground)_10%,transparent)]",
          )}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
}
