// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch as SwitchPrimitive } from "radix-ui";
import type * as React from "react";

import { cn } from "@/lib/utils";

function Switch({
  className,
  size = "default",
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root> & {
  size?: "sm" | "default";
}) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      data-size={size}
      className={cn(
        // Thumb travel below is width - thumb - 2 * inset; keep the inset a whole pixel, or the thumb rounds off-centre.
        "peer group/switch relative inline-flex shrink-0 items-center rounded-full data-[size=default]:p-[calc(2px*var(--ui-space-scale,1))] data-[size=sm]:p-[calc(2px*var(--ui-space-scale,1))] outline-none transition-[background-color,box-shadow] duration-200 ease-out after:absolute after:-inset-x-3 after:-inset-y-2",
        "data-[size=default]:h-[calc(20px*var(--ui-space-scale,1))] data-[size=default]:w-[calc(34px*var(--ui-space-scale,1))] data-[size=sm]:h-[calc(16px*var(--ui-space-scale,1))] data-[size=sm]:w-[calc(26px*var(--ui-space-scale,1))]",
        "data-checked:bg-control-accent data-unchecked:bg-[color-mix(in_oklab,var(--foreground)_9%,transparent)] dark:data-unchecked:bg-[color-mix(in_oklab,var(--foreground)_11%,transparent)] hover:data-unchecked:bg-[color-mix(in_oklab,var(--foreground)_13%,transparent)] dark:hover:data-unchecked:bg-[color-mix(in_oklab,var(--foreground)_14%,transparent)] hover:data-checked:bg-[color-mix(in_oklab,var(--control-accent),black_6%)]",
        "focus-visible:ring-[3px] focus-visible:ring-ring/40 aria-invalid:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 data-disabled:cursor-not-allowed data-disabled:opacity-50",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="pointer-events-none block rounded-full bg-white transition-transform duration-200 ease-out group-data-[size=default]/switch:size-[calc(16px*var(--ui-space-scale,1))] group-data-[size=sm]/switch:size-[calc(12px*var(--ui-space-scale,1))] data-unchecked:translate-x-0 group-data-[size=default]/switch:data-checked:translate-x-[calc(14px*var(--ui-space-scale,1))] group-data-[size=sm]/switch:data-checked:translate-x-[calc(10px*var(--ui-space-scale,1))] data-checked:bg-control-accent-foreground motion-reduce:transition-none"
      />
    </SwitchPrimitive.Root>
  );
}

export { Switch };
