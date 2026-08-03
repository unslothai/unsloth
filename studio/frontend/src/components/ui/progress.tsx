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
  indeterminate?: boolean;
}) {
  return (
    <ProgressPrimitive.Root
      {...props}
      data-slot="progress"
      className={cn(
        "bg-foreground/[0.06] h-3 rounded-4xl relative flex w-full items-center overflow-x-hidden",
        className,
      )}
      value={indeterminate ? undefined : value}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          "bg-control-accent size-full flex-1 transition-all",
          indeterminate && "w-1/3 flex-none loading-bar-slide",
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
