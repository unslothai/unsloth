// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useSettingsDialogStore } from "@/features/settings";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { AiSecurity03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

interface HfTokenIndicatorProps {
  /** true: chip with "HF Token" label (Train wizard); false: icon-only pill (Hub header). */
  showLabel?: boolean;
}

// Compact "set / not set" indicator for the app-wide HF token; click opens
// Settings -> General. Shared by the Hub header and Train wizard (same store).
export function HfTokenIndicator({ showLabel = false }: HfTokenIndicatorProps = {}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const openDialog = useSettingsDialogStore((s) => s.openDialog);
  const hasToken = Boolean(hfToken && hfToken.trim());

  const ariaLabel = hasToken
    ? "Hugging Face token configured"
    : "Set Hugging Face token";
  const tipText = hasToken
    ? "Token set. Allows access to private and gated repos."
    : "Set a token to access private and gated repos.";

  if (showLabel) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => openDialog("general")}
            aria-label={ariaLabel}
            className={cn(
              "hub-menu-trigger field-soft inline-flex h-9 w-full items-center justify-between gap-2 rounded-[12px] py-0 pl-1.5 pr-3 text-ui-12p5 font-medium text-foreground transition-colors",
              "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
            )}
          >
            <span className="flex min-w-0 items-center gap-2">
              <span
                aria-hidden="true"
                className={cn(
                  "inline-flex size-6 items-center justify-center rounded-full transition-colors",
                  hasToken
                    ? "bg-verified/15 text-verified"
                    : "bg-foreground/[0.07] text-muted-foreground dark:bg-white/[0.07]",
                )}
              >
                <HugeiconsIcon
                  icon={AiSecurity03Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </span>
              <span className="truncate">HF Token</span>
            </span>
            <span
              className={cn(
                "shrink-0 text-ui-11 font-normal tabular-nums",
                hasToken ? "text-verified" : "text-muted-foreground/70",
              )}
            >
              {hasToken ? "Set" : "Add"}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
          {tipText}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => openDialog("general")}
          aria-label={ariaLabel}
          className={cn(
            // Solid circle reads optically larger than the flat HTTP/Xet box, so
            // keep it 22px to sit within the row rather than bulging above it.
            "inline-flex h-[22px] w-[22px] items-center justify-center rounded-full text-ui-11p5 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            hasToken
              ? "hub-tag-soft text-muted-foreground hover:text-foreground/80"
              : "bg-destructive text-destructive-foreground hover:bg-destructive/90",
          )}
        >
          <HugeiconsIcon
            icon={AiSecurity03Icon}
            strokeWidth={1.75}
            // Shield ink leans right; nudge left to optically centre it.
            className="block size-[13px] shrink-0 -translate-x-[0.5px]"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {tipText}
      </TooltipContent>
    </Tooltip>
  );
}
