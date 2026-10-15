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
import { hfApiToken, useHfTokenStore } from "../stores/hf-token-store";

interface HfTokenIndicatorProps {
  showLabel?: boolean;
  onOpenSettings: () => void;
}

function maskHfToken(token: string): string {
  const trimmed = token.trim();
  return trimmed.length < 8 ? "••••" : `••••${trimmed.slice(-4)}`;
}

export function HfTokenIndicator({
  showLabel = false,
  onOpenSettings,
}: HfTokenIndicatorProps) {
  const t = useT();
  const hfToken = useHfTokenStore((s) => s.token);
  const hasToken = hfApiToken(hfToken) !== undefined;

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
                    : "bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] text-muted-foreground dark:bg-[rgb(255_255_255_/_calc(0.07*var(--contrast-wash-gain,1)))]",
                )}
              >
                <HugeiconsIcon
                  icon={AiSecurity03Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </span>
              <span
                className={cn(
                  "truncate",
                  hasToken
                    ? "font-mono text-verified"
                    : "text-muted-foreground",
                )}
              >
                {hasToken ? (
                  <>
                    <span className="sr-only">{t("picker.hfToken.saved")}</span>
                    {maskHfToken(hfToken)}
                  </>
                ) : (
                  t("picker.hfToken.add")
                )}
              </span>
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
            "inline-flex h-[calc(22px*var(--ui-space-scale,1))] w-[calc(22px*var(--ui-space-scale,1))] items-center justify-center rounded-full text-ui-11p5 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            hasToken
              ? "hub-tag-soft text-muted-foreground hover:text-foreground/80"
              : "bg-destructive text-destructive-foreground hover:bg-destructive/90",
          )}
        >
          <HugeiconsIcon
            icon={AiSecurity03Icon}
            strokeWidth={1.75}
            className="block size-[calc(13px*var(--ui-space-scale,1))] shrink-0 -translate-x-[0.5px]"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {tipText}
      </TooltipContent>
    </Tooltip>
  );
}
