


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
  // work of unknown length: sweep a segment across the track instead of filling it.
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
      // radix reads a missing value as the indeterminate state, which is what belongs here.
      value={indeterminate ? undefined : value}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          "bg-control-accent",
          // one branch owns the width: tailwind-merge keeps both `size-full` and `w-1/3`.
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
