// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** The 40px column every row in a results card starts with, so icons and play buttons line up down the card. */
export const CLIP_ICON_COLUMN = "size-[calc(40px*var(--ui-space-scale,1))]";

/** The expanded top row (player, transcript, stems): tinted, on the same 40px column and title edge as the rows. */
export const CLIPS_TOP_ROW = "grid w-full gap-2 rounded-xl bg-wash-6 px-3 py-3";

export const CLIPS_TITLE_INSET = "ps-[calc(52px*var(--ui-space-scale,1))]";

export const CLIPS_LIST =
  "hover-scrollbar mt-2 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto border-t border-wash-6 pt-2";

export const CLIPS_HEADER_LINK =
  "me-3 shrink-0 rounded-full text-ui-12 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

export function ClipsIconTile({
  icon,
  className,
}: {
  icon: IconSvgElement;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "grid shrink-0 place-items-center rounded-full bg-wash-6 text-muted-foreground",
        CLIP_ICON_COLUMN,
        className,
      )}
    >
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.5}
        className="size-[calc(18px*var(--ui-space-scale,1))]"
      />
    </span>
  );
}

export function ClipsCard({
  label,
  title,
  count,
  countMore = false,
  action,
  bare = false,
  children,
}: {
  label: string;
  title: ReactNode;
  count?: number | null;
  countMore?: boolean;
  action?: ReactNode;
  /** Nothing to list yet: no header, and the card fills the pane so its empty state sits centred. */
  bare?: boolean;
  children: ReactNode;
}) {
  if (bare) {
    return (
      <section aria-label={label} className="flex min-h-0 w-full flex-1 flex-col">
        {children}
      </section>
    );
  }
  return (
    <section
      aria-label={label}
      // No box of its own: the top row is the one tinted block, so nothing sits block-in-block.
      className="flex min-h-0 w-full flex-1 flex-col"
    >
      {/* No icon tile: the rail title beside it already carries the workflow's icon. */}
      <header className="flex items-center gap-3 px-3 pb-3">
        <div className="flex min-w-0 flex-1 items-center gap-2 text-ui-14 font-medium text-foreground">
          <h2 className="min-w-0 truncate">{title}</h2>
          {count ? (
            <span className="shrink-0 rounded-full bg-wash-8 px-2 py-0.5 text-ui-11 font-normal tabular-nums text-muted-foreground">
              {count}
              {countMore ? "+" : ""}
            </span>
          ) : null}
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}

/** The output pane with nothing in it: one icon, what will appear, how to get it, and the next step.
 *  Centred like the Images and Video canvases. */
export function ClipsCardEmpty({
  icon,
  title,
  text,
  actions,
}: {
  icon?: IconSvgElement;
  title?: ReactNode;
  text: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex min-h-[calc(240px*var(--ui-space-scale,1))] flex-1 flex-col items-center justify-center gap-3 px-3 pb-[8%] text-center text-muted-foreground">
      {icon ? (
        <HugeiconsIcon icon={icon} strokeWidth={1.5} className="size-12" aria-hidden="true" />
      ) : null}
      <div className="grid max-w-[46ch] gap-1">
        {/* The pane's heading while it has no "Clips" header of its own. */}
        {title ? (
          <h2 className="text-sm font-medium text-balance text-foreground">{title}</h2>
        ) : null}
        <p className="text-ui-13 leading-normal text-pretty">{text}</p>
      </div>
      {actions ? <div className="mt-1">{actions}</div> : null}
    </div>
  );
}
