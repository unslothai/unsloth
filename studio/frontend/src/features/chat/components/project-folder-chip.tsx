// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import { useIsAccountOwner } from "@/features/auth";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Folder02Icon } from "@/lib/folder-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  GitBranchIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ProjectRecord } from "../types";
import {
  projectFolderState,
  shortenFolderLabel,
} from "../utils/project-folder-state";
import { OpenProjectFolderItem } from "./open-chat-folder-item";

export function ProjectFolderChip({
  project,
  onChange,
}: {
  project: ProjectRecord;
  onChange: () => void;
}) {
  const isOwner = useIsAccountOwner();
  if (!isOwner) return null;
  const folder = projectFolderState(project);

  return (
    <NonModalDropdownMenu
      side="bottom"
      align="start"
      sideOffset={6}
      className="unsloth-plus-menu w-64"
      trigger={(triggerRef) => (
        <button
          ref={triggerRef}
          type="button"
          data-testid="project-folder-chip"
          data-folder-state={
            folder.unavailable
              ? "unavailable"
              : folder.chosen
                ? "chosen"
                : "managed"
          }
          title={folder.problem ?? folder.path ?? undefined}
          className={cn(
            "inline-flex h-7 max-w-full min-w-0 items-center gap-1.5 rounded-full px-2.5 text-ui-13 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring data-[state=open]:bg-muted",
            folder.unavailable
              ? "text-amber-600 dark:text-amber-400"
              : "text-muted-foreground hover:text-foreground data-[state=open]:text-foreground",
          )}
        >
          <HugeiconsIcon
            icon={folder.unavailable ? Alert02Icon : Folder02Icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
          />
          <span
            className={cn(
              "min-w-0 truncate",
              folder.chosen && "font-mono text-ui-12",
            )}
          >
            {shortenFolderLabel(folder.label)}
          </span>
          {folder.unavailable && <span className="shrink-0">not found</span>}
          {folder.branch && (
            <span className="flex min-w-0 shrink items-center gap-1 border-l border-border pl-1.5">
              <HugeiconsIcon
                icon={GitBranchIcon}
                strokeWidth={1.75}
                className="size-3 shrink-0"
              />
              <span className="min-w-0 truncate font-mono text-ui-12">
                {folder.branch}
              </span>
            </span>
          )}
        </button>
      )}
    >
      {folder.problem && (
        <>
          <DropdownMenuLabel className="font-normal whitespace-normal text-muted-foreground">
            {folder.problem}
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
        </>
      )}
      <DropdownMenuItem
        disabled={!folder.path}
        onSelect={() => {
          if (!folder.path) return;
          void copyToClipboard(folder.path).then((copied) => {
            if (copied) toast.success("Folder path copied");
            else toast.error("Could not copy the folder path");
          });
        }}
      >
        <HugeiconsIcon
          icon={Copy01Icon}
          strokeWidth={1.75}
          className="size-icon"
        />
        <span>Copy path</span>
      </DropdownMenuItem>
      <OpenProjectFolderItem projectId={project.id} />
      <DropdownMenuItem onSelect={onChange}>
        <HugeiconsIcon
          icon={PencilEdit02Icon}
          strokeWidth={1.75}
          className="size-icon"
        />
        <span>{folder.chosen ? "Change folder…" : "Choose a folder…"}</span>
      </DropdownMenuItem>
    </NonModalDropdownMenu>
  );
}
