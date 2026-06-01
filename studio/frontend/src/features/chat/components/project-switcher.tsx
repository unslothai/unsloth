// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Folder02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";
import type { ReactElement } from "react";
import type { ProjectRecord } from "../types";

export function ProjectSwitcher({
  currentProject,
  projects,
  isLoading,
  onSelectProject,
  onViewAllProjects,
}: {
  currentProject: ProjectRecord | null;
  projects: ProjectRecord[];
  isLoading: boolean;
  onSelectProject: (projectId: string) => void;
  onViewAllProjects: () => void;
}): ReactElement {
  const showLoadingRow = isLoading && projects.length === 0;
  const showEmptyRow = !isLoading && projects.length === 0;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={
            currentProject
              ? `Project: ${currentProject.name}. Switch project`
              : "Pick a project"
          }
          className="flex shrink-0 items-center gap-1.5 rounded-[8px] -mx-1 px-1.5 py-0.5 transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={Folder02Icon}
            strokeWidth={1.75}
            className="size-icon shrink-0 text-foreground/70"
          />
          <span className="max-w-[150px] truncate font-medium text-foreground">
            {currentProject ? currentProject.name : "Projects"}
          </span>
          <ChevronDown
            strokeWidth={1.75}
            className="size-3.5 shrink-0 text-foreground/60"
            aria-hidden={true}
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="bottom"
        align="start"
        sideOffset={6}
        className="app-user-menu menu-soft-surface ring-0 min-w-56 max-w-72 max-h-72 py-2 font-heading rounded-[14px] border-0"
      >
        {showLoadingRow ? (
          <DropdownMenuItem disabled={true} className="text-muted-foreground">
            Loading…
          </DropdownMenuItem>
        ) : null}
        {showEmptyRow ? (
          <DropdownMenuItem disabled={true} className="text-muted-foreground">
            No projects yet
          </DropdownMenuItem>
        ) : null}
        {projects.map((project) => {
          const isActive = currentProject?.id === project.id;
          return (
            <DropdownMenuItem
              key={project.id}
              onSelect={() => onSelectProject(project.id)}
              className="justify-between"
            >
              <span className="flex min-w-0 items-center gap-2">
                <HugeiconsIcon
                  icon={Folder02Icon}
                  strokeWidth={1.75}
                  className="size-icon shrink-0 text-foreground/70"
                />
                <span className="truncate">{project.name}</span>
              </span>
              {isActive ? (
                <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
                  className="size-icon shrink-0 text-foreground/80"
                />
              ) : null}
            </DropdownMenuItem>
          );
        })}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onViewAllProjects}>
          View all projects
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
