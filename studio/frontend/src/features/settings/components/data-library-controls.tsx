// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState, type ReactNode } from "react";
import { Folder01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import {
  type LibraryFilters,
  type LibraryItem,
  type LibrarySort,
  formatLibraryDate,
  groupLibraryItems,
} from "./data-library";

const SORT_LABELS: Record<LibrarySort, string> = {
  default: "Default",
  updated: "Updated",
  created: "Created",
  oldest: "Oldest first",
  alphabetical: "Alphabetical",
};
const TYPE_LABELS: Record<string, string> = {
  all: "All chats",
  single: "Single chats",
  compare: "Compare chats",
};

export function LibraryToolbar({
  filters,
  onChange,
  placeholder,
  projects,
  disabled = false,
}: {
  filters: LibraryFilters;
  onChange: (filters: LibraryFilters) => void;
  placeholder: string;
  projects?: ReadonlyMap<string, string>;
  disabled?: boolean;
}) {
  const [projectOpen, setProjectOpen] = useState(false);
  const projectLabel =
    filters.project === "all"
      ? "All projects"
      : filters.project === "none"
        ? "No project"
        : (projects?.get(filters.project.slice(8)) ?? "Unavailable project");
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="relative min-w-48 flex-1">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute start-3 top-3 size-4 text-muted-foreground"
        />
        <Input
          type="search"
          value={filters.query}
          onChange={(event) =>
            onChange({ ...filters, query: event.target.value })
          }
          disabled={disabled}
          placeholder={placeholder}
          aria-label={placeholder}
          className="h-10 rounded-full ps-9"
        />
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            variant="outline"
            disabled={disabled}
            className="h-10 gap-2 rounded-xl"
            aria-label="Filter and sort"
          >
            {projects
              ? TYPE_LABELS[filters.type]
              : filters.sort === "default"
                ? "Sort"
                : SORT_LABELS[filters.sort]}
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              className="size-4 text-muted-foreground"
            />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          {projects && (
            <>
              <DropdownMenuLabel>Type</DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={filters.type}
                onValueChange={(type) => onChange({ ...filters, type })}
              >
                {Object.entries(TYPE_LABELS).map(([value, label]) => (
                  <DropdownMenuRadioItem key={value} value={value}>
                    {label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
            </>
          )}
          <DropdownMenuLabel>Sort by</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={filters.sort}
            onValueChange={(sort) =>
              onChange({ ...filters, sort: sort as LibrarySort })
            }
          >
            {Object.entries(SORT_LABELS)
              .filter(([value]) =>
                projects ? value !== "default" : value !== "updated",
              )
              .map(([value, label]) => (
                <DropdownMenuRadioItem key={value} value={value}>
                  {label}
                </DropdownMenuRadioItem>
              ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
      {projects && (
        <Popover open={projectOpen} onOpenChange={setProjectOpen}>
          <PopoverTrigger asChild={true}>
            <Button
              variant="outline"
              disabled={disabled}
              className="h-10 max-w-full gap-2 rounded-xl"
              aria-label="Filter by project"
            >
              <HugeiconsIcon icon={Folder01Icon} className="size-4 shrink-0" />
              <span className="max-w-40 truncate">{projectLabel}</span>
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className="size-4 shrink-0 text-muted-foreground"
              />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-64 p-1">
            <Command>
              <CommandInput
                placeholder="Search projects"
                aria-label="Search projects"
              />
              <CommandList>
                <CommandEmpty>No projects found.</CommandEmpty>
                {[
                  ["all", "All projects"],
                  ["none", "No project"],
                  ...[...projects]
                    .sort((a, b) => a[1].localeCompare(b[1]))
                    .map(([id, name]) => [`project:${id}`, name]),
                ].map(([value, label]) => (
                  <CommandItem
                    key={value}
                    value={value}
                    keywords={[label]}
                    onSelect={() => {
                      onChange({ ...filters, project: value });
                      setProjectOpen(false);
                    }}
                  >
                    {label}
                  </CommandItem>
                ))}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
}

export function LibraryRow({
  title,
  date,
  leading,
  actions,
  onOpen,
}: {
  title: string;
  date: number;
  leading?: ReactNode;
  actions?: ReactNode;
  onOpen?: () => void;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3 py-3.5 text-sm">
      {leading}
      <div className="min-w-0 flex-1">
        {onOpen ? (
          <button
            type="button"
            onClick={onOpen}
            title={title}
            className="block max-w-full truncate text-start font-medium hover:underline"
          >
            {title || "Untitled"}
          </button>
        ) : (
          <p title={title} className="truncate font-medium">
            {title || "Untitled"}
          </p>
        )}
        <p className="mt-1 text-xs text-muted-foreground">
          {formatLibraryDate(date)}
        </p>
      </div>
      {actions && (
        <div className="flex shrink-0 items-center gap-1">{actions}</div>
      )}
    </div>
  );
}

export function ChatLibraryGroups<T extends LibraryItem>({
  items,
  projects,
  children,
}: {
  items: readonly T[];
  projects: ReadonlyMap<string, string>;
  children: (item: T) => ReactNode;
}) {
  return (
    <div className="space-y-5">
      {groupLibraryItems(items, projects).map((group) => (
        <section key={group.id} className="space-y-2">
          <div className="flex items-center gap-2 px-1 text-sm">
            <HugeiconsIcon
              icon={Folder01Icon}
              className="size-4 shrink-0 text-muted-foreground"
            />
            <h3
              className="min-w-0 flex-1 truncate font-medium"
              title={group.name}
            >
              {group.name}
            </h3>
            <span className="text-xs text-muted-foreground">
              {group.items.length} {group.items.length === 1 ? "chat" : "chats"}
            </span>
          </div>
          <div className="divide-y divide-border/50 rounded-2xl border border-border/60 px-3 sm:px-4">
            {group.items.map((item) => (
              <div key={item.id}>{children(item)}</div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
