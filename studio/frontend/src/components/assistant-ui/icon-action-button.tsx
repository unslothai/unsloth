// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { ComponentProps, ReactNode } from "react";

/**
 * A transcript toolbar button: Copy, Download and the like, above a code cell or beside a
 * thinking trace's header. Icon only, because these sit in narrow headers next to each other
 * and the word only repeats what the glyph already says. The label is not dropped: it is the
 * accessible name and it is what the tooltip shows on hover.
 */
export function IconActionButton({
  label,
  children,
  ...props
}: ComponentProps<"button"> & { label: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          className="inline-flex items-center justify-center rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          aria-label={label}
          {...props}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
