// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";

/** The 40px column every row in a results card starts with, so icons and play buttons line up down the card. */
export const CLIP_ICON_COLUMN = "size-[calc(40px*var(--ui-space-scale,1))]";

/** The expanded top row (player, transcript, stems): tinted, on the same 40px column and title edge as the rows. */
export const CLIPS_TOP_ROW = "grid w-full gap-2.5 rounded-xl bg-wash-6 p-3";

export const CLIP_PLAY_BUTTON =
  "shrink-0 rounded-full border-transparent bg-background text-foreground shadow-sm hover:bg-background hover:text-foreground dark:bg-wash-12 dark:hover:bg-wash-18 [&_svg]:fill-current [&_svg.lucide-play]:ml-0.5";

export const CLIP_ROW =
  "group relative flex items-center gap-1 rounded-lg ps-3 transition-colors hover:bg-wash-3 hover:[--clip-row-duration:0] has-[[aria-haspopup=menu]:focus-visible]:[--clip-row-duration:0] has-[[aria-expanded=true]]:[--clip-row-duration:0]";

export const CLIP_ROW_TILE =
  "mx-0.5 grid size-[calc(36px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full bg-wash-4 text-muted-foreground";

export const CLIP_ROW_ICON = "size-[calc(18px*var(--ui-space-scale,1))]";

export const CLIP_ROW_SELECT =
  "flex min-w-0 flex-1 items-center gap-3 py-2.5 pe-4 ps-2 text-left pointer-coarse:pe-10";

/** The row sets the property: a style after a :has() is not allowed here (tests/has-subject-after.test.ts). */
export const CLIP_ROW_DURATION =
  "shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground opacity-[var(--clip-row-duration,1)] transition-opacity pointer-coarse:opacity-100!";

export const CLIP_ROW_MENU = "absolute inset-y-0 end-2.5 flex items-center";

export const CLIP_ROW_PIN = "start-[calc(38px*var(--ui-space-scale,1))] top-1";

export const CLIPS_TITLE_INSET = "ps-[calc(52px*var(--ui-space-scale,1))]";

export const CLIPS_LIST =
  "hover-scrollbar mt-2 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto border-t border-wash-6 pt-2";

export const CLIPS_HEADER_LINK =
  "me-1 inline-flex shrink-0 items-center gap-1.5 rounded-full text-ui-12 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

export function ClipsDeleteAll({
  noun,
  active,
  onConfirm,
}: {
  /** Plural, lower case: "clips", "separations". */
  noun: string;
  /** The page stays mounted off-route and the dialog portals to body: close it when hidden. */
  active: boolean;
  onConfirm: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!active) setOpen(false);
  }, [active]);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className={CLIPS_HEADER_LINK}>
        <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} className="size-3.5" />
        Delete all
      </button>
      <AlertDialog open={active && open} onOpenChange={(next) => (busy ? undefined : setOpen(next))}>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete all {noun}?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes all {noun} on this page. Archived {noun} are kept. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                setBusy(true);
                void onConfirm().finally(() => {
                  setBusy(false);
                  setOpen(false);
                });
              }}
            >
              {busy ? "Deleting…" : "Delete all"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

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
  /** Nothing to list yet: no header. */
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
        {title ? (
          <h2 className="text-sm font-medium text-balance text-foreground">{title}</h2>
        ) : null}
        <p className="text-ui-13/normal text-pretty">{text}</p>
      </div>
      {actions ? <div className="mt-1">{actions}</div> : null}
    </div>
  );
}
