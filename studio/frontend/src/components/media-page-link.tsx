// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  FlimSlateIcon,
  Image03Icon,
  LibrariesIcon,
} from "@hugeicons/core-free-icons";
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
import { ArrowRightIcon } from "lucide-react";

export const HEADER_ICON_BUTTON =
  "flex size-[calc(30px*var(--ui-space-scale,1))] shrink-0 cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:text-white";

/** Link to another page, past a divider; `iconOnly` makes it a header icon button. */
export function MediaPageLink({
  to,
  libraryTab,
  label,
  icon,
  tooltip,
  onNavigate,
  labelClassName,
  arrowClassName,
  iconOnly = false,
  iconArrow = false,
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
  iconOnly?: boolean;
  iconArrow?: boolean;
}) {
  const navigate = useNavigate();
  const go = () => {
    onNavigate?.();
    if (to === "/library") {
      void navigate({ to, search: libraryTab ? { show: libraryTab } : {} });
    } else {
      void navigate({ to });
    }
  };
  if (iconOnly) {
    return (
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={label}
            onClick={go}
            className={cn(
              HEADER_ICON_BUTTON,
              // The square button's inset, so the icon stays put.
              iconArrow &&
                "w-auto gap-1.5 pr-2 pl-[calc((30px*var(--ui-space-scale,1)-var(--icon-size))/2)]",
            )}
          >
            <HugeiconsIcon
              icon={icon}
              strokeWidth={1.75}
              className="size-icon"
            />
            {iconArrow && (
              <>
                <span className="hidden min-w-0 truncate text-ui-13 font-medium @[50rem]:inline">
                  {label}
                </span>
                <ArrowRightIcon className="size-3.5 shrink-0 opacity-60" />
              </>
            )}
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          {tooltip ?? label}
        </TooltipContent>
      </Tooltip>
    );
  }
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
            onClick={go}
            className="flex h-[calc(34px*var(--ui-space-scale,1))] min-w-0 items-center gap-1.5 rounded-full pl-2.5 pr-2 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HugeiconsIcon icon={icon} className="size-4 shrink-0" />
            <span className={cn("min-w-0 truncate", labelClassName)}>
              {label}
            </span>
            <ArrowRightIcon
              className={cn("size-3.5 shrink-0 opacity-60", arrowClassName)}
            />
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
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
  iconOnly,
}: {
  tab: LibraryTab;
  labelClassName?: string;
  arrowClassName?: string;
  iconOnly?: boolean;
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
      iconOnly={iconOnly}
    />
  );
}

export function SiblingMediaPageLink({ to }: { to: "/images" | "/video" }) {
  const t = useT();
  const label = t(
    to === "/images" ? "shell.navigation.images" : "shell.navigation.video",
  );
  return (
    <MediaPageLink
      to={to}
      label={label}
      icon={to === "/images" ? Image03Icon : FlimSlateIcon}
      tooltip={label}
      iconOnly={true}
      iconArrow={true}
    />
  );
}
