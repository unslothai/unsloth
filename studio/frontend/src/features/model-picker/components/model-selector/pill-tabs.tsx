// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export interface PillTab {
  value: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
}

export function PillTabs({
  tabs,
  value,
  onValueChange,
  ariaLabel,
  className,
  compact = false,
  fit = false,
  disabled = false,
  dataTour,
}: {
  tabs: PillTab[];
  value: string;
  onValueChange: (value: string) => void;
  ariaLabel: string;
  className?: string;
  compact?: boolean;
  dataTour?: string;
  disabled?: boolean;
  /** Size each tab to its label instead of equal widths; the active tab carries the pill. */
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
      data-tour={dataTour}
      className={cn(
        "hub-menu-trigger hub-tab-toggle relative inline-flex items-center rounded-full",
        compact ? "h-7" : "h-(--picker-control-h)",
        // Do not stretch in a flex-column parent, and never compress so the last tab keeps its padding.
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
          // Roving tabindex (WAI-ARIA tablist); ArrowDown bubbles to the picker's enter-the-list handler.
          tabIndex={value === tab.value ? 0 : -1}
          disabled={disabled || tab.disabled}
          onKeyDown={(e) => {
            if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
            e.preventDefault();
            const step = e.key === "ArrowRight" ? 1 : -1;
            let next = (index + step + tabs.length) % tabs.length;
            while (tabs[next].disabled && next !== index) {
              next = (next + step + tabs.length) % tabs.length;
            }
            if (next === index) return;
            onValueChange(tabs[next].value);
            e.currentTarget.parentElement
              ?.querySelectorAll<HTMLElement>('button[role="tab"]')
              .item(next)
              ?.focus();
          }}
          onClick={() => onValueChange(tab.value)}
          className={cn(
            "relative z-10 inline-flex items-center justify-center gap-1.5 rounded-full transition-colors",
            (disabled || tab.disabled) && "cursor-not-allowed opacity-50",
            fit ? "min-w-0 shrink" : "min-w-0 flex-1",
            compact
              ? "h-7 px-2.5 text-ui-11"
              : "h-(--picker-control-h) px-3 text-ui-12p5",
            value === tab.value
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
            // The active tab carries the pill; its hover lives in hub.css (needs a dark-mode variant).
            fit && value === tab.value && "hub-tab-toggle-pill",
          )}
        >
          {tab.icon}
          <span className="min-w-0 truncate">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
