// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";

export function SettingsRow({
  label,
  description,
  hint,
  icon,
  children,
  destructive,
  className,
  below,
}: {
  label: string;
  description?: ReactNode;
  /** Shown on hover; plain text because it doubles as the trigger's aria-label. */
  hint?: string;
  icon?: ReactNode;
  children?: ReactNode;
  destructive?: boolean;
  className?: string;
  below?: ReactNode;
}) {
  return (
    <div
      data-settings-label={label}
      className={cn(
        // Controls are fixed-width, so wrap to avoid starving the label.
        "flex flex-wrap items-center justify-end gap-x-6 gap-y-2 py-3",
        destructive && "border-t border-border/60 mt-2 pt-4",
        className,
      )}
    >
      <div
        className="flex min-w-[calc(11rem*var(--ui-space-scale,1))] flex-1 basis-0 items-center gap-2.5"
      >
        {icon ? (
          <span className="flex shrink-0 items-center text-foreground">
            {icon}
          </span>
        ) : null}
        <div className="flex min-w-0 w-full max-w-lg flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">
            {label}
            {hint ? (
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  {/* Focusable and labelled, so keyboard users reach the
                      text too. Matches the secure-HTTPS hint. Inline on the
                      baseline, so it follows the label's last word like a glyph. */}
                  <button
                    type="button"
                    aria-label={hint}
                    className="ml-1.5 inline-flex align-baseline rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    {/* Sized off the token, not size-3.5: the label beside it is
                        scaled by the UI font size preference, and a fixed 14px
                        glyph drifts out of proportion with it. Same curve the
                        app's other small glyphs follow. */}
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-[var(--ui-icon-size-hint)]"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-[calc(300px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
                  {hint}
                </TooltipContent>
              </Tooltip>
            ) : null}
          </span>
          {description ? (
            <span className="text-xs text-muted-foreground leading-snug">
              {description}
            </span>
          ) : null}
        </div>
      </div>
      {children ? (
        <div className="flex max-w-full shrink-0 items-center">{children}</div>
      ) : null}
      {below ? <div className="flex basis-full justify-end">{below}</div> : null}
    </div>
  );
}
