// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  ArrowDown01Icon,
  Folder01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
  const label = currentProject?.name ?? (isLoading ? "Project" : "Projects");

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={
            currentProject
              ? `Project: ${currentProject.name}. Switch project`
              : isLoading
                ? "Loading project"
                : "Pick a project"
          }
          className="-mx-1 flex h-[34px] shrink-0 items-center gap-2 rounded-[10px] px-1.5 transition-colors hover:bg-[#ececec] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:hover:bg-[#2d2e32]"
        >
          <HugeiconsIcon
            icon={Folder01Icon}
            strokeWidth={1.75}
            className="size-icon shrink-0 text-foreground/70"
          />
          <span className="flex min-w-0 flex-1 items-baseline">
            <span className="min-w-0 flex max-w-[150px] flex-1 items-baseline truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
              {label}
            </span>
          </span>
          <span className="flex size-4 shrink-0 items-center justify-center">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              strokeWidth={1.75}
              className="relative top-0.5 size-3.5 text-muted-foreground"
              aria-hidden={true}
            />
          </span>
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
                  icon={Folder01Icon}
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
