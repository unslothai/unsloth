// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowTurnBackwardIcon,
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type KeyboardEventHandler, type ReactNode, useState } from "react";

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { useProjectSubmenu } from "./project-submenu";

export interface MediaViewerActions {
  /** The white pill, e.g. Chat about this. */
  primary?: { label: string; icon: IconSvgElement; onClick: () => void; disabled?: boolean };
  onDownload?: () => void;
  onViewChat?: () => void;
  favorite?: boolean;
  onToggleFavorite?: () => void;
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
  onDelete?: () => void;
}

/**
 * One file, filling the window. Images and videos sit on black; everything else keeps the theme.
 * Shared by the Library and the Images and Video pages, so a file opens the same way everywhere.
 */
export function MediaViewer({
  open,
  onOpenChange,
  title,
  meta,
  media,
  noun,
  actions,
  extra,
  onKeyDown,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  meta?: ReactNode;
  /** Images and videos: black stage, white controls. */
  media: boolean;
  /** Used in labels and messages, e.g. "image". */
  noun: string;
  actions: MediaViewerActions;
  /** Anything else for the header, before the actions (e.g. Save). */
  extra?: ReactNode;
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  children: ReactNode;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const project = useProjectSubmenu({ noun, onAddToProject: actions.onAddToProject });
  const iconButton = cn(
    "flex size-9 shrink-0 items-center justify-center rounded-full outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
    media ? "text-white hover:bg-neutral-800 aria-expanded:bg-neutral-800" : "hover:bg-muted aria-expanded:bg-muted",
  );
  const hasMenu = Boolean(
    actions.onViewChat || actions.onToggleFavorite || actions.onAddToProject || actions.onDelete,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        onKeyDown={onKeyDown}
        className={cn(
          "left-0 top-[var(--studio-window-chrome-top,0px)] flex h-[calc(100dvh-var(--studio-window-chrome-top,0px))] w-dvw max-h-none max-w-none translate-x-0 translate-y-0 flex-col gap-0 overflow-hidden rounded-none p-0 ring-0 sm:max-w-none",
          media && "bg-black text-white",
        )}
      >
        <div className="flex items-center gap-2 py-3 pl-6 pr-4">
          <div className="min-w-0 flex-1">
            <DialogTitle className="truncate text-[15px] font-medium">{title}</DialogTitle>
            <DialogDescription
              className={cn("mt-0.5 truncate text-[13px]", media && "text-neutral-400")}
            >
              {meta}
            </DialogDescription>
          </div>
          {extra}
          {actions.primary && (
            <button
              type="button"
              disabled={actions.primary.disabled}
              onClick={actions.primary.onClick}
              className={cn(
                "mr-1 flex h-9 shrink-0 items-center gap-2 rounded-full px-4 text-sm font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
                media
                  ? "bg-white text-black hover:bg-neutral-200"
                  : "bg-foreground text-background hover:opacity-90",
              )}
            >
              <HugeiconsIcon icon={actions.primary.icon} strokeWidth={1.75} className="size-4" />
              {actions.primary.label}
            </button>
          )}
          {actions.onDownload && (
            <button
              type="button"
              aria-label="Download"
              title="Download"
              onClick={actions.onDownload}
              className={iconButton}
            >
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-5" />
            </button>
          )}
          {hasMenu && (
            <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
              <DropdownMenuTrigger asChild={true}>
                <button type="button" aria-label={`More actions for this ${noun}`} className={iconButton}>
                  <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="unsloth-plus-menu sidebar-row-menu menu-flat-destructive w-56"
              >
                {actions.onViewChat && (
                  <>
                    <DropdownMenuItem onClick={actions.onViewChat}>
                      <HugeiconsIcon icon={ArrowTurnBackwardIcon} strokeWidth={1.75} className="size-icon" />
                      View original chat
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}
                {actions.onToggleFavorite && (
                  <DropdownMenuItem onClick={actions.onToggleFavorite}>
                    <HugeiconsIcon
                      icon={StarPointedIcon}
                      strokeWidth={1.75}
                      className={cn("size-icon", actions.favorite && "[&_path]:fill-current")}
                    />
                    {actions.favorite ? "Remove from Favorites" : "Add to Favorites"}
                  </DropdownMenuItem>
                )}
                {project.submenu}
                {actions.onDelete && (
                  <DropdownMenuItem variant="destructive" onClick={actions.onDelete}>
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          <DialogClose asChild={true}>
            <button type="button" aria-label="Close" className={iconButton}>
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-5" />
            </button>
          </DialogClose>
        </div>
        <div className={cn("flex min-h-0 flex-1", media ? "px-4 pb-4" : "px-6 pb-6")}>{children}</div>
        {project.dialog}
      </DialogContent>
    </Dialog>
  );
}
