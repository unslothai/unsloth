// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { AiSecurity03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useHfTokenStore } from "../stores/hf-token-store";

interface HfTokenIndicatorProps {
  /** true: chip with "HF Token" label (Train wizard); false: icon-only pill (Hub header). */
  showLabel?: boolean;
  onOpenSettings: () => void;
}

// Compact "set / not set" indicator for the app-wide HF token; click opens
// Settings -> General. Shared by the Hub header and Train wizard (same store).
export function HfTokenIndicator({
  showLabel = false,
  onOpenSettings,
}: HfTokenIndicatorProps) {
  const t = useT();
  const hfToken = useHfTokenStore((s) => s.token);
  const hasToken = Boolean(hfToken.trim());

  const ariaLabel = hasToken
    ? t("picker.hfToken.savedAriaLabel")
    : t("picker.hfToken.addAriaLabel");
  const tipText = hasToken
    ? t("picker.hfToken.savedHint")
    : t("picker.hfToken.addHint");

  if (showLabel) {
    return (
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            onClick={onOpenSettings}
            aria-label={ariaLabel}
            className={cn(
              "hub-menu-trigger field-soft inline-flex h-9 w-full items-center justify-between gap-2 rounded-[12px] py-0 pl-1.5 pr-3 text-ui-12p5 font-medium text-foreground transition-colors",
              PICKER_FOCUS_VISIBLE_CLASS,
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
              <span className="truncate">{t("picker.hfToken.label")}</span>
            </span>
            <span
              className={cn(
                "shrink-0 text-ui-11 font-normal tabular-nums",
                hasToken ? "text-verified" : "text-muted-foreground/70",
              )}
            >
              {hasToken ? t("picker.hfToken.saved") : t("picker.hfToken.add")}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          {tipText}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={onOpenSettings}
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
