// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Archive03Icon,
  ArchiveRestoreIcon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  FolderAddIcon,
  FolderExportIcon,
  MoreVerticalIcon,
  PinIcon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { NewProjectDialog, useChatProjects } from "@/features/chat";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

/**
 * Actions for one gallery item, shared by the Images, Video and Audio pages. Matches a chat row's menu.
 *
 * "toolbar" sits in the glass toolbar over the preview; "overlay" is the badge that appears on a
 * filmstrip tile on hover; "row" is the quiet trigger at the end of a list row. A tile is itself a
 * <button>, so the menu must be rendered as its SIBLING, never a child -- nested buttons are invalid
 * and break keyboard activation.
 */
export type GalleryItemMenuVariant = "toolbar" | "overlay" | "row";

export function GalleryItemMenu({
  pinned,
  archived,
  onTogglePin,
  onToggleArchive,
  onDelete,
  onDownload,
  onAddToProject,
  leadingItems,
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
  /** One-click download of the original file. */
  onDownload?: () => void;
  /** Copies the item into a project's folder. */
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
  /** Page-specific items shown first, above a separator. */
  leadingItems?: ReactNode;
  variant?: GalleryItemMenuVariant;
  /** Used in the aria-label and messages, e.g. "image" or "video". */
  noun: string;
  /** False while the page is off-tab; forces the menu shut so a portalled popup cannot outlive it. */
  active?: boolean;
  className?: string;
}) {
  // Controlled like RecipePopover: DropdownMenuContent portals to body, so the inert page wrapper
  // cannot contain it when the tab goes away.
  const [open, setOpen] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  useEffect(() => {
    if (!active) {
      setOpen(false);
      setCreatingProject(false);
    }
  }, [active]);
  const { projects } = useChatProjects();

  async function addToProject(projectId: string, projectName: string) {
    if (!onAddToProject) return;
    try {
      const { already } = await onAddToProject(projectId);
      toast.success(already ? `Already in ${projectName}` : `Added to ${projectName}`);
    } catch (err) {
      toast.error(`Failed to add ${noun} to project`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  const overlay = variant === "overlay";
  const row = variant === "row";
  const menu = (
    <DropdownMenu open={active && open} onOpenChange={(o) => setOpen(active && o)}>
      <DropdownMenuTrigger asChild={true}>
        <Button
          size={overlay || row ? "icon-xs" : "icon-sm"}
          variant="ghost"
          aria-label={`More actions for this ${noun}`}
          className={cn(
            // Circular hover.
            "rounded-full",
            // Reads over any thumbnail, whatever its colours.
            overlay &&
              "bg-background/80 text-foreground shadow-sm ring-1 ring-border backdrop-blur hover:bg-background",
            row && "text-muted-foreground hover:text-foreground",
            className,
          )}
        >
          <HugeiconsIcon
            icon={MoreVerticalIcon}
            className={overlay || row ? "size-3.5" : "size-4"}
          />
        </Button>
      </DropdownMenuTrigger>
      {/* Same styling as a chat row's menu. */}
      <DropdownMenuContent
        align="end"
        className="unsloth-plus-menu sidebar-row-menu menu-flat-destructive w-52"
      >
        {leadingItems ? (
          <>
            {leadingItems}
            <DropdownMenuSeparator />
          </>
        ) : null}
        <DropdownMenuItem onClick={onTogglePin}>
          <HugeiconsIcon icon={pinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
          {pinned ? "Unpin" : "Pin"}
        </DropdownMenuItem>
        {onDownload ? (
          <DropdownMenuItem onClick={onDownload}>
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
            Download
          </DropdownMenuItem>
        ) : null}
        {onAddToProject ? (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-icon" />
              <span>Project</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent
              sideOffset={0}
              alignOffset={-4}
              className="unsloth-plus-menu sidebar-row-menu w-48"
            >
              {/* Actions above the rule, destinations below, as in a chat's Project menu. */}
              <DropdownMenuItem onClick={() => setCreatingProject(true)}>
                <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className="size-icon" />
                <span>New project</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              {projects.length === 0 ? (
                <DropdownMenuItem disabled={true}>No projects yet</DropdownMenuItem>
              ) : (
                projects.map((project) => (
                  <DropdownMenuItem
                    key={project.id}
                    onClick={() => void addToProject(project.id, project.name)}
                  >
                    <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                    <span className="truncate">{project.name}</span>
                  </DropdownMenuItem>
                ))
              )}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ) : null}
        <DropdownMenuItem onClick={onToggleArchive}>
          <HugeiconsIcon icon={archived ? ArchiveRestoreIcon : Archive03Icon} strokeWidth={1.75} className="size-icon" />
          {archived ? "Restore from archive" : "Archive"}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={onDelete}>
          <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  // Mounted only while open, since the overlay renders once per tile.
  const newProjectDialog = creatingProject ? (
    <NewProjectDialog
      open={true}
      onOpenChange={setCreatingProject}
      title={`Add ${noun} to new project`}
      submitLabel="Create and add"
      onCreated={(project) => addToProject(project.id, project.name)}
    />
  ) : null;

  if (!overlay && !row) {
    return (
      <>
        {menu}
        {newProjectDialog}
      </>
    );
  }
  return (
    <span
      className={cn(
        "inline-flex",
        active && open
          ? "opacity-100"
          : "opacity-0 group-hover:opacity-100 has-[button:focus-visible]:opacity-100 pointer-coarse:opacity-100",
      )}
    >
      {menu}
      {newProjectDialog}
    </span>
  );
}

/** A pinned tile's marker, which unpins on click. Shows the unpin icon on hover or focus. */
export function GalleryPinBadge({
  noun,
  onUnpin,
  className,
}: {
  noun: string;
  onUnpin: () => void;
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={`Unpin this ${noun}`}
          onClick={onUnpin}
          className={cn(
            "group/pin absolute flex items-center justify-center rounded-full bg-background/80 p-0.5 text-foreground shadow-sm ring-1 ring-border backdrop-blur outline-none transition-colors hover:bg-background focus-visible:ring-2 focus-visible:ring-ring",
            className,
          )}
        >
          <HugeiconsIcon
            icon={PinIcon}
            className="size-3 group-hover/pin:hidden group-focus-visible/pin:hidden"
          />
          <HugeiconsIcon
            icon={PinOffIcon}
            className="hidden size-3 group-hover/pin:block group-focus-visible/pin:block"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent>Unpin</TooltipContent>
    </Tooltip>
  );
}
