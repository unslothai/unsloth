// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Progress as ProgressPrimitive } from "radix-ui";
import type * as React from "react";

import { cn } from "@/lib/utils";

function Progress({
  className,
  indicatorClassName,
  indeterminate = false,
  value,
  ...props
}: React.ComponentProps<typeof ProgressPrimitive.Root> & {
  indicatorClassName?: string;
  /** Work of unknown length: sweep a segment across the track instead of filling it. */
  indeterminate?: boolean;
}) {
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      className={cn(
        "bg-foreground/[0.06] h-3 rounded-4xl relative flex w-full items-center overflow-x-hidden",
        className,
      )}
      {...props}
      // Radix reads a missing value as the indeterminate state, which is what assistive
      // tech should announce here, so it is dropped rather than passed through as 0.
      value={indeterminate ? undefined : value}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          "bg-control-accent",
          // One branch owns the width: tailwind-merge keeps both `size-full` and `w-1/3`,
          // which would leave the sweeping segment full-width on a utility-order change.
          indeterminate
            ? "h-full w-1/3 loading-bar-slide"
            : "size-full flex-1 transition-all",
          indicatorClassName,
        )}
        style={
          indeterminate
            ? undefined
            : { transform: `translateX(-${100 - (value || 0)}%)` }
        }
      />
    </ProgressPrimitive.Root>
  );
}

export { Progress };
