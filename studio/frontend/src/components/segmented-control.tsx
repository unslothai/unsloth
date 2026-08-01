// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { type CSSProperties, type ReactNode, useId } from "react";

export interface SegmentedControlOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

export function SegmentedControlIndicator({
  activeIndex,
  optionCount,
  className,
  width = `${100 / optionCount}%`,
}: {
  activeIndex: number;
  optionCount: number;
  className?: string;
  width?: CSSProperties["width"];
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "hub-tab-toggle-pill pointer-events-none absolute start-0 rounded-full transition-transform duration-200 ease-out [transform:translateX(calc(var(--segmented-control-index)*100%))] rtl:[transform:translateX(calc(var(--segmented-control-index)*-100%))]",
        className,
      )}
      style={
        {
          "--segmented-control-index": activeIndex,
          width,
        } as CSSProperties
      }
    />
  );
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  className,
}: {
  value: T;
  options: readonly [
    SegmentedControlOption<T>,
    SegmentedControlOption<T>,
    ...SegmentedControlOption<T>[],
  ];
  onValueChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
}) {
  const name = useId();
  const activeIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );

  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "hub-tab-toggle relative inline-flex h-9 w-full min-w-0 items-stretch rounded-full",
        className,
      )}
    >
      <SegmentedControlIndicator
        activeIndex={activeIndex}
        optionCount={options.length}
        className="inset-y-0"
      />
      {options.map((option) => {
        const active = option.value === value;
        return (
          <label
            key={option.value}
            className={cn(
              "relative z-10 flex h-full min-w-0 flex-1 cursor-pointer rounded-full",
              option.disabled && "cursor-not-allowed opacity-50",
            )}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={active}
              disabled={option.disabled}
              onChange={() => onValueChange(option.value)}
              className="peer sr-only"
            />
            <span
              className={cn(
                "inline-flex h-full w-full select-none items-center justify-center rounded-full px-3 text-ui-12p5 font-medium transition-colors peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-inset peer-focus-visible:ring-ring",
                active
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {option.label}
            </span>
          </label>
        );
      })}
    </div>
  );
}
