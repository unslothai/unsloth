// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Folder01Icon, FolderAddIcon, FolderExportIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";

import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { NewProjectDialog, useChatProjects } from "@/features/chat";
import { toast } from "@/lib/toast";

/**
 * The Project submenu and its New project dialog, for any menu that copies an item into a project.
 * Render `submenu` inside the menu and `dialog` beside it.
 */
export function useProjectSubmenu({
  noun,
  onAddToProject,
}: {
  noun: string;
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
}) {
  const [creatingProject, setCreatingProject] = useState(false);
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

  const submenu = onAddToProject ? (
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
  ) : null;

  // Mounted only while open, since the overlay renders once per tile.
  const dialog = creatingProject ? (
    <NewProjectDialog
      open={true}
      onOpenChange={setCreatingProject}
      title={`Add ${noun} to new project`}
      submitLabel="Create and add"
      onCreated={(project) => addToProject(project.id, project.name)}
    />
  ) : null;

  const close = useCallback(() => setCreatingProject(false), []);
  return { submenu, dialog, close };
}
