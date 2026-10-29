// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { cn } from "@/lib/utils";
import { type ReactNode, useId } from "react";

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
  /** Size each tab to its label instead of equal widths. Tabs only shrink
   * when their combined intrinsic width exceeds the available space. */
  fit?: boolean;
}) {
  const pillId = useId();
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
            "relative inline-flex items-center justify-center gap-1.5 rounded-full transition-colors",
            (disabled || tab.disabled) && "cursor-not-allowed opacity-50",
            fit ? "min-w-0 shrink" : "min-w-0 flex-1",
            compact
              ? "h-7 px-2.5 text-ui-11"
              : "h-(--picker-control-h) px-3 text-ui-12p5",
            value === tab.value
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {value === tab.value && <SegmentedPill group={`${pillId}-pill`} />}
          {tab.icon}
          {/* Consumers target the label by this marker; a bare `> span`
              would also match the pill. */}
          <span data-pill-tab-label className="min-w-0 truncate">
            {tab.label}
          </span>
        </button>
      ))}
    </div>
  );
}
