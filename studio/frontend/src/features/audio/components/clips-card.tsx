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

/** Content under a top row's header starts at the title edge: 40px column + 12px gap. */
export const CLIPS_TITLE_INSET = "ps-[calc(52px*var(--ui-space-scale,1))]";

/** The scrolling list of earlier results, set off from the top row by a hairline. */
export const CLIPS_LIST =
  "hover-scrollbar mt-2 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto border-t border-wash-6 pt-2";

/** A quiet header link, ending on the same right edge as the top row's actions. */
export const CLIPS_HEADER_LINK =
  "me-3 shrink-0 rounded-full text-ui-12 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

/** A round 40px icon tile on the shared column. */
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

/** The results section on every Audio workflow: top-anchored, a header, then a top row and the list. */
export function ClipsCard({
  label,
  title,
  count,
  countMore = false,
  action,
  children,
}: {
  label: string;
  title: ReactNode;
  count?: number | null;
  countMore?: boolean;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={label}
      // No box of its own: the top row is the one tinted block, so nothing sits block-in-block.
      className="flex min-h-0 w-full flex-col"
    >
      {/* No icon tile: the rail title beside it already carries the workflow's icon. */}
      <header className="flex items-center gap-3 px-3 pt-2 pb-3">
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

/** The card's body before the first result: a hint and any suggestions, on the title edge. */
export function ClipsCardEmpty({
  text,
  actions,
}: {
  text: ReactNode;
  actions?: ReactNode;
}) {
  return (
    // The header's 12px inset, so the hint starts under the title.
    <div className="grid gap-3 px-3 pb-3">
      <p className="text-ui-13 leading-normal text-muted-foreground">{text}</p>
      {actions ? <div className="[&>div]:justify-start">{actions}</div> : null}
    </div>
  );
}
