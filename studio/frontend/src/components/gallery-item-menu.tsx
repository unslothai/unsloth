// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Archive02Icon,
  ArchiveRestoreIcon,
  Delete02Icon,
  MoreHorizontalIcon,
  PinIcon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

/**
 * Pin / archive / delete for one gallery item, shared by the Images and Video pages.
 *
 * "toolbar" sits in the glass toolbar over the preview; "overlay" is the badge that appears on a
 * filmstrip tile on hover. A tile is itself a <button>, so the overlay must be rendered as its
 * SIBLING, never a child -- nested buttons are invalid and break keyboard activation.
 */
export type GalleryItemMenuVariant = "toolbar" | "overlay";

export function GalleryItemMenu({
  pinned,
  archived,
  onTogglePin,
  onToggleArchive,
  onDelete,
  variant = "toolbar",
  noun,
  active = true,
  className,
}: {
  pinned: boolean;
  archived: boolean;
  onTogglePin: () => void;
  onToggleArchive: () => void;
  onDelete: () => void;
  variant?: GalleryItemMenuVariant;
  /** Used in the aria-label, e.g. "image" or "video". */
  noun: string;
  /** False while the page is off-tab; forces the menu shut so a portalled popup cannot outlive it. */
  active?: boolean;
  className?: string;
}) {
  // Controlled like RecipePopover: DropdownMenuContent portals to body, so the inert page wrapper
  // cannot contain it when the tab goes away.
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!active) setOpen(false);
  }, [active]);

  const overlay = variant === "overlay";
  return (
    <DropdownMenu open={active && open} onOpenChange={(o) => setOpen(active && o)}>
      <DropdownMenuTrigger asChild={true}>
        <Button
          size={overlay ? "icon-xs" : "sm"}
          variant="ghost"
          aria-label={`More actions for this ${noun}`}
          className={cn(
            // Reads over any thumbnail, whatever its colours.
            overlay &&
              "bg-background/80 text-foreground shadow-sm ring-1 ring-border backdrop-blur hover:bg-background",
            className,
          )}
        >
          <HugeiconsIcon icon={MoreHorizontalIcon} className={overlay ? "size-3.5" : "size-4"} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={onTogglePin}>
          <HugeiconsIcon icon={pinned ? PinOffIcon : PinIcon} />
          {pinned ? "Unpin" : "Pin to front"}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onToggleArchive}>
          <HugeiconsIcon icon={archived ? ArchiveRestoreIcon : Archive02Icon} />
          {archived ? "Restore from archive" : "Archive"}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={onDelete}>
          <HugeiconsIcon icon={Delete02Icon} />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
