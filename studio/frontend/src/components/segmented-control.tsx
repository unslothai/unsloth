// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { type ReactNode, useId, useLayoutEffect, useRef } from "react";
import {
  type SegmentedSize,
  segmentedSegmentLabel,
  segmentedTrackHeight,
} from "./segmented-control-styles";
import { watchSegmentedPill } from "./segmented-pill-slide";

interface SegmentedControlOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

const HIDDEN = { display: "none" } as const;

/** Pieces in the order segmented-pill-slide reads them: shadow, start cap, strip, end cap. */
const SLIDE_PIECES = [
  "left-0 origin-left rounded-full",
  "left-0 rounded-l-full",
  "origin-left",
  "right-0 rounded-r-full",
];

/**
 * Render the pill only inside the active (relative) segment with a control-unique group id; the
 * track must isolate (.hub-tab-toggle does) and segments must not, or the pill covers labels
 */
export function SegmentedPill({
  group,
  className,
}: {
  group: string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    if (!ref.current) return;
    return watchSegmentedPill(group, ref.current);
  }, [group]);
  return (
    <span
      ref={ref}
      aria-hidden="true"
      className={cn(
        "hub-tab-toggle-pill pointer-events-none absolute inset-0 -z-10 rounded-full",
        className,
      )}
    >
      {/* Shown only while sliding between segments of different widths. */}
      {SLIDE_PIECES.map((piece) => (
        <span
          key={piece}
          className={cn(
            "hub-tab-toggle-pill absolute inset-y-0",
            piece,
            className,
          )}
          style={HIDDEN}
        />
      ))}
    </span>
  );
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  size = "default",
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
  size?: SegmentedSize;
  className?: string;
}) {
  const name = useId();
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "hub-tab-toggle relative inline-flex w-full min-w-0 items-stretch rounded-full",
        segmentedTrackHeight[size],
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <label
            key={option.value}
            className={cn(
              "relative flex h-full min-w-0 flex-1 cursor-pointer rounded-full",
              option.disabled && "cursor-not-allowed opacity-50",
            )}
          >
            {active && <SegmentedPill group={`${name}-pill`} />}
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
                "inline-flex h-full w-full select-none items-center justify-center rounded-full font-medium transition-colors peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-inset peer-focus-visible:ring-ring",
                segmentedSegmentLabel,
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
