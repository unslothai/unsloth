// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LibrariesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { LibraryTab } from "@/features/library";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowRightIcon,
} from "lucide-react";

/** Link to another page's workspace, parked past a divider so it reads as leaving. */
export function MediaPageLink({
  to,
  libraryTab,
  label,
  icon,
  tooltip,
  onNavigate,
  labelClassName,
  arrowClassName,
}: {
  to: "/images" | "/video" | "/audio" | "/library";
  libraryTab?: LibraryTab;
  label: string;
  icon: IconSvgElement;
  /** Needed on a translated page: the default prefix below is English. */
  tooltip?: string;
  onNavigate?: () => void;
  labelClassName?: string;
  arrowClassName?: string;
}) {
  const navigate = useNavigate();
  return (
    <>
      {/* first:hidden: the control to its left is conditional, and a leading divider looks stray. */}
      <span
        aria-hidden="true"
        className="mx-0.5 h-4 w-px shrink-0 bg-border/70 first:hidden"
      />
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={label}
            onClick={() => {
              onNavigate?.();
              if (to === "/library") {
                void navigate({ to, search: libraryTab ? { show: libraryTab } : {} });
              } else {
                void navigate({ to });
              }
            }}
            className="flex h-[calc(34px*var(--ui-space-scale,1))] min-w-0 items-center gap-1.5 rounded-full pl-2.5 pr-2 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HugeiconsIcon icon={icon} className="size-4 shrink-0" />
            <span className={cn("min-w-0 truncate", labelClassName)}>{label}</span>
            <ArrowRightIcon
              className={cn("size-3.5 shrink-0 opacity-60", arrowClassName)}
            />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
          {tooltip ?? `Go to ${label}`}
        </TooltipContent>
      </Tooltip>
    </>
  );
}

export function LibraryPageLink({
  tab,
  labelClassName,
  arrowClassName,
}: {
  tab: LibraryTab;
  labelClassName?: string;
  arrowClassName?: string;
}) {
  const t = useT();
  return (
    <MediaPageLink
      to="/library"
      libraryTab={tab}
      label={t("shell.navigation.library")}
      tooltip={t("studio.goToLibrary")}
      icon={LibrariesIcon}
      labelClassName={labelClassName}
      arrowClassName={arrowClassName}
    />
  );
}
