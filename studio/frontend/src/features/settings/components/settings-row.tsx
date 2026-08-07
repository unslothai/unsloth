// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export function SettingsRow({
  label,
  description,
  icon,
  children,
  destructive,
  className,
  alignTop,
}: {
  label: string;
  description?: ReactNode;
  icon?: ReactNode;
  children?: ReactNode;
  destructive?: boolean;
  className?: string;
  /** Top-align the control instead of centering it, for tall descriptions. */
  alignTop?: boolean;
}) {
  return (
    <div
      data-settings-label={label}
      className={cn(
        // Controls are fixed-width and shrink-0, so an unwrapped row starves the
        // label. justify-end right-aligns a wrapped control without breaking
        // items-stretch for flex-col callers.
        "flex flex-wrap justify-end gap-x-6 gap-y-2 py-3",
        alignTop ? "items-start" : "items-center",
        destructive && "border-t border-border/60 mt-2 pt-4",
        className,
      )}
    >
      <div
        className={cn(
          // Widest floor that leaves already-fitting rows unchanged.
          "flex min-w-[11rem] flex-1 basis-0 gap-2.5",
          alignTop ? "items-start" : "items-center",
        )}
      >
        {icon ? (
          <span className="flex shrink-0 items-center text-foreground">
            {icon}
          </span>
        ) : null}
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">{label}</span>
          {description ? (
            <span className="text-xs text-muted-foreground leading-snug">
              {description}
            </span>
          ) : null}
        </div>
      </div>
      {children ? (
        <div
          className={cn(
            "flex max-w-full shrink-0",
            // Line the control up with the first description line, not the label.
            alignTop ? "items-start pt-[21px]" : "items-center",
          )}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}
