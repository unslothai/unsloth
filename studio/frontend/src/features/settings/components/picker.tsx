// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared chrome for the settings model pickers (dictation model, refining
// model). Keeping the trigger and row styles in one place is what makes the two
// pickers look and behave the same.

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ComponentProps } from "react";

/** Matches the app's `SelectTrigger` so a picker and a select read as one family. */
export const pickerTriggerClass =
  "border-border bg-background hover:bg-accent/50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] focus-visible:border-ring dark:focus-visible:border-transparent dark:focus-visible:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))] flex h-8 w-full cursor-pointer items-center justify-between gap-1.5 rounded-full border px-3.5 text-sm outline-none transition-colors";

export function PickerTrigger({
  className,
  children,
  ...props
}: ComponentProps<"button">) {
  return (
    <button
      type="button"
      className={cn(pickerTriggerClass, className)}
      {...props}
    >
      <span className="flex min-w-0 items-center gap-1.5 truncate">
        {children}
      </span>
      <HugeiconsIcon
        icon={ChevronDownStandardIcon}
        strokeWidth={2}
        className="text-muted-foreground pointer-events-none size-4 shrink-0"
      />
    </button>
  );
}

export function PickerRow({
  selected,
  className,
  children,
  ...props
}: ComponentProps<"button"> & { selected?: boolean }) {
  return (
    <button
      type="button"
      aria-selected={selected}
      className={cn(
        "flex w-full items-center justify-between gap-3 rounded-full px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-muted",
        selected && "bg-accent font-medium",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
