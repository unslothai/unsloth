// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useChatActive } from "../runtime-provider";
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

  // Controlled so the body-portaled dropdown can't linger over another tab off-route.
  const active = useChatActive();
  const [open, setOpen] = useState(false);

  return (
    <DropdownMenu open={active && open} onOpenChange={(o) => setOpen(active && o)}>
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
          className="-mx-1 flex h-[34px] shrink-0 items-center gap-2 rounded-full pl-3 pr-2.5 transition-colors hover:bg-[#ececec] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:bg-accent"
        >
          <HugeiconsIcon
            icon={Folder01Icon}
            strokeWidth={1.75}
            className="size-icon shrink-0 text-foreground/70"
          />
          <span className="flex min-w-0 flex-1 items-baseline">
            <span className="min-w-0 flex max-w-[150px] flex-1 items-baseline truncate font-heading text-[1rem] font-medium leading-tight text-black dark:text-white">
              {label}
            </span>
          </span>
          <span className="flex size-4 shrink-0 items-center justify-center">
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-3.5 text-muted-foreground"
              aria-hidden={true}
            />
          </span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="bottom"
        align="start"
        sideOffset={0}
        className="unsloth-plus-menu ring-0 min-w-56 max-w-72 font-heading"
      >
        {/* Scroll the list here, not the container, so the rounded corners on
            the scrollbar side are not squared off. */}
        <div className="max-h-72 overflow-y-auto">
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
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
