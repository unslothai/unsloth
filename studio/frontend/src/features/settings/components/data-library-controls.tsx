// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo, useState, type ReactNode } from "react";
import { useLocale, useT, type TranslationKey } from "@/i18n";
import { useLibraryProjectLabels } from "./use-library-project-labels";
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

const SORT_LABELS: Record<LibrarySort, TranslationKey> = {
  default: "settings.data.library.defaultSort",
  updated: "settings.data.library.updated",
  created: "settings.data.library.created",
  oldest: "settings.data.library.oldest",
  alphabetical: "settings.data.library.alphabetical",
};
const TYPE_LABELS: Record<string, TranslationKey> = {
  all: "settings.data.library.allChats",
  single: "settings.data.library.singleChats",
  compare: "settings.data.library.compareChats",
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
  const t = useT();
  const locale = useLocale();
  const compare = useMemo(() => new Intl.Collator(locale).compare, [locale]);
  const [projectOpen, setProjectOpen] = useState(false);
  const projectLabel =
    filters.project === "all"
      ? t("settings.data.library.allProjects")
      : filters.project === "none"
        ? t("settings.data.library.noProject")
        : (projects?.get(filters.project.slice(8)) ??
          t("settings.data.library.unavailableProject"));
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
            aria-label={t("settings.data.library.filterSort")}
          >
            {projects
              ? t(TYPE_LABELS[filters.type])
              : filters.sort === "default"
                ? t("settings.data.library.sort")
                : t(SORT_LABELS[filters.sort])}
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              className="size-4 text-muted-foreground"
            />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          {projects && (
            <>
              <DropdownMenuLabel>
                {t("settings.data.library.type")}
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={filters.type}
                onValueChange={(type) => onChange({ ...filters, type })}
              >
                {Object.entries(TYPE_LABELS).map(([value, label]) => (
                  <DropdownMenuRadioItem key={value} value={value}>
                    {t(label)}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
            </>
          )}
          <DropdownMenuLabel>
            {t("settings.data.library.sortBy")}
          </DropdownMenuLabel>
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
                  {t(label)}
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
              aria-label={t("settings.data.library.filterProject")}
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
                placeholder={t("settings.data.library.searchProjects")}
                aria-label={t("settings.data.library.searchProjects")}
              />
              <CommandList>
                <CommandEmpty>
                  {t("settings.data.library.noProjects")}
                </CommandEmpty>
                {[
                  ["all", t("settings.data.library.allProjects")],
                  ["none", t("settings.data.library.noProject")],
                  ...[...projects]
                    .sort((a, b) => compare(a[1], b[1]))
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
  const t = useT();
  const locale = useLocale();
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
            {title || t("settings.data.library.untitled")}
          </button>
        ) : (
          <p title={title} className="truncate font-medium">
            {title || t("settings.data.library.untitled")}
          </p>
        )}
        <p className="mt-1 text-xs text-muted-foreground">
          {formatLibraryDate(date, locale)}
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
  const t = useT();
  const labels = useLibraryProjectLabels();
  return (
    <div className="space-y-5">
      {groupLibraryItems(items, projects, labels).map((group) => (
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
              {t(
                group.items.length === 1
                  ? "settings.data.library.oneChat"
                  : "settings.data.library.chatCount",
                { count: group.items.length },
              )}
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
