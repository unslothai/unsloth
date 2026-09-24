// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  Folder01Icon,
  FolderAddIcon,
  FolderExportIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MessageCircleIcon, StarPointedIcon } from "@/lib/hugeicons-derived";
import { useMemo, useState } from "react";
import type { LibraryFolder } from "../api";
import { isDeletable, isFileItem, isModelItem } from "../file-kind";
import { type LibraryTarget, useLibraryActions } from "../actions-context";
import { canReveal, revealInFolder, useRevealLabel } from "../reveal";

// Sized and weighted as the sidebar's chat and project menus draw theirs.
const ICON = "size-icon";

/** Folders a target can move into, depth-first with their nesting depth. A folder is never offered
 *  itself or anything under it. */
function moveDestinations(
  folders: LibraryFolder[],
  target: LibraryTarget,
): { folder: LibraryFolder; depth: number }[] {
  const excluded = target.kind === "folder" ? target.folder.id : null;
  const byParent = new Map<string | null, LibraryFolder[]>();
  for (const folder of folders) {
    const siblings = byParent.get(folder.parentId) ?? [];
    siblings.push(folder);
    byParent.set(folder.parentId, siblings);
  }
  const out: { folder: LibraryFolder; depth: number }[] = [];
  const walk = (parentId: string | null, depth: number) => {
    const children = [...(byParent.get(parentId) ?? [])].sort((a, b) =>
      a.name.localeCompare(b.name),
    );
    for (const folder of children) {
      if (folder.id === excluded) continue;
      out.push({ folder, depth });
      walk(folder.id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}

function currentFolderId(target: LibraryTarget): string | null {
  return target.kind === "item" ? target.item.folderId : target.folder.parentId;
}

/** The ⋯ button and its menu. `overlay` is the frosted round button that floats over a card. */
export function LibraryActionsMenu({
  target,
  variant,
  className,
}: {
  target: LibraryTarget;
  variant: "overlay" | "row";
  className?: string;
}) {
  const actions = useLibraryActions();
  const [open, setOpen] = useState(false);
  const destinations = useMemo(
    () => (open ? moveDestinations(actions.folders, target) : []),
    [open, actions.folders, target],
  );
  const inFolder = currentFolderId(target);
  const item = target.kind === "item" ? target.item : null;
  const revealLabel = useRevealLabel();

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="More actions"
          data-open={open || undefined}
          onClick={(event) => event.stopPropagation()}
          className={cn(
            "flex shrink-0 items-center justify-center rounded-full outline-none transition-opacity focus-visible:opacity-100 data-open:opacity-100",
            variant === "overlay"
              ? "size-8 bg-black/45 text-white opacity-0 backdrop-blur-md hover:bg-black/60 group-hover/library-card:opacity-100"
              : "size-8 text-muted-foreground opacity-0 hover:bg-accent hover:text-foreground group-hover/library-row:opacity-100",
            className,
          )}
        >
          <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-60"
        onClick={(event) => event.stopPropagation()}
      >
        <DropdownMenuItem onSelect={() => actions.chatAbout(target)}>
          <HugeiconsIcon icon={MessageCircleIcon} strokeWidth={1.75} className={ICON} />
          {item && isModelItem(item) ? "Chat with this model" : "Chat about this"}
        </DropdownMenuItem>
        {item && (
          <DropdownMenuItem onSelect={() => actions.toggleFavorite(item)}>
            <HugeiconsIcon
              icon={StarPointedIcon}
              strokeWidth={1.75}
              className={cn(ICON, item.favorite && "[&_path]:fill-current")}
            />
            {item.favorite ? "Remove from Favorites" : "Add to Favorites"}
          </DropdownMenuItem>
        )}
        {item && isFileItem(item) && (
          <DropdownMenuItem onSelect={() => actions.download(item)}>
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className={ICON} />
            Download
          </DropdownMenuItem>
        )}
        {item && revealLabel && canReveal(item) && (
          <DropdownMenuItem onSelect={() => revealInFolder(item.id)}>
            <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className={ICON} />
            {revealLabel}
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onSelect={() => actions.rename(target)}>
          <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className={ICON} />
          Rename
        </DropdownMenuItem>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className="gap-2.5">
            <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className={ICON} />
            Add to folder
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="max-h-[min(--spacing(80),var(--radix-dropdown-menu-content-available-height))] w-56">
            <DropdownMenuItem onSelect={() => actions.moveToNewFolder(target)}>
              <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className={ICON} />
              New folder
            </DropdownMenuItem>
            {inFolder && (
              <DropdownMenuItem onSelect={() => actions.moveTo(target, null)}>
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className={ICON} />
                Library (no folder)
              </DropdownMenuItem>
            )}
            {destinations.length > 0 && <DropdownMenuSeparator />}
            {destinations.map(({ folder, depth }) => (
              <DropdownMenuItem
                key={folder.id}
                disabled={folder.id === inFolder}
                onSelect={() => actions.moveTo(target, folder.id)}
                style={{ paddingLeft: `${12 + depth * 14}px` }}
              >
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className={ICON} />
                <span className="truncate">{folder.name}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        {(!item || isDeletable(item)) && (
          <DropdownMenuItem variant="destructive" onSelect={() => actions.remove(target)}>
            <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className={ICON} />
            {target.kind === "folder" ? "Delete folder" : "Delete"}
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
